import random
import threading
import math
import traceback
import gc
from typing import Iterator, Any, Literal

import torch
import torch.optim as optim
from loguru import logger
from pydantic import BaseModel, Field
import settings
from utils.vector_utils import add_artificial_gradients, flatten_optimizer_state, reconstruct_optimizer_state
from transformers import PreTrainedTokenizer, AutoTokenizer

from miner.api_client import APIClient
from model.loaders import load_model_split, load_dataloader
from settings import (
    BATCH_SIZE,
    DATASET_NAME,
    DEVICE,
    HF_TOKEN,
    LEARNING_RATE,
    MOCK,
    MODEL_SPLITS,
    PACK_SAMPLES,
    SEQUENCE_LENGTH,
    TOKENIZER_NAME,
    WEIGHT_DECAY,
    MODEL_CFG,
    BITTENSOR,
)
import bittensor as bt
from bittensor_wallet.mock import get_mock_wallet

from utils.metagraph_syncer import MetagraphSyncer
from utils.bt_utils import get_subtensor, NotRegisteredError
from utils.partitions import ChunkData, Partition
from utils.s3_interactions import download_metadata, download_weights_or_optimizer_state


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 64).to(torch.bfloat16)
        self.layer2 = torch.nn.Linear(64, 32).to(torch.bfloat16)
        self.layer3 = torch.nn.Linear(32, 100).to(torch.bfloat16)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x, {}

    def backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # Pass in activation_grads to backward() to avoid implicit scalar gradient error
        output_activations.backward(activation_grads)

    def parameters(self):
        return [p for p in super().parameters()]


class BaseNeuron(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    hotkey: str | None = None
    MAX_ACTIVATION_CACHE_SIZE: int = settings.MAX_ACTIVATION_CACHE_SIZE
    STEPS_UNTIL_ALL_REDUCE: int = 16
    STEPS_UNTIL_SYNC: int = 16 * 4
    completed_optim_steps: int = 0
    saved_forward_activations: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = Field(default_factory=dict)
    layer: int | None = None
    processed_forward_activations: list[str] = Field(default_factory=list)
    processed_backward_activations: list[str] = Field(default_factory=list)
    backwards_since_reduce: int = 0
    backwards_since_sync: int = 0
    forwards_since_reduce: int = 0
    weights: torch.Tensor | None = None
    model: torch.nn.Module | None = None
    total_model_params: int | None = None
    tokenizer: PreTrainedTokenizer | None = None
    optimizer: optim.Optimizer | None = None
    lr_scheduler: optim.lr_scheduler.LRScheduler | None = None
    vocab_size: int | None = None
    eos_token_id: int | None = None
    dataloader: Iterator[torch.Tensor] | None = None
    weight_version: str | None = None
    api_client: APIClient | None = None
    wallet: bt.wallet | None = None
    subtensor: bt.subtensor | None = None
    metagraph: bt.metagraph | None = None
    config: bt.config | None = None
    netuid: int | None = None
    lock: Any | None = None
    metagraph_syncer: MetagraphSyncer | None = None
    wandb_initialized: bool = False
    uid: int | None = None
    wallet_name: str | None = None
    wallet_hotkey: str | None = None

    def _clean_gpu_memory(self):
        logger.debug(f"🗑️ Cleaning GPU memory for {self.hotkey[:8]} 🗑️")

        # Explicitly delete attributes that hold large tensors
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            logger.debug("🗑️ Deleted self.model 🗑️")

        if hasattr(self, "optimizer") and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
            logger.debug("🗑️ Deleted self.optimizer 🗑️")

        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            del self.lr_scheduler
            self.lr_scheduler = None
            logger.debug("🗑️ Deleted self.lr_scheduler 🗑️")

        if hasattr(self, "weights") and self.weights is not None:
            del self.weights
            self.weights = None
            logger.debug("🗑️ Deleted self.weights 🗑️")

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            logger.debug(f"GPU memory cleaned. Allocated memory: {allocated_memory:.2f}GB")

    def _clear_cache_entry(self, activation_uid: str):
        """Safely removes an entry from the activation cache and cleans up tensors."""
        if activation_uid in self.saved_forward_activations:
            # Pop the data from the cache
            cached_data = self.saved_forward_activations.pop(activation_uid, None)

            if cached_data:
                # Unpack and explicitly delete to release references, helping the GC
                input_act, output_act, state, _ = cached_data
                del input_act
                del output_act
                del state
                del cached_data

            logger.debug(f"🗑️ Removed and cleaned activation {activation_uid} from cache 🗑️")

    def _clear_saved_forward_activations(self):
        """Clears the saved forward activations cache for all activations."""
        for activation_uid in list(self.saved_forward_activations.keys()):
            self._clear_cache_entry(activation_uid=activation_uid)

    async def _download_chunk(
        self,
        data_path: str,
        metadata_path: str,
        chunk_id: int | str,
        data_type: Literal["weights", "optimizer_state"],
    ) -> tuple[torch.Tensor, dict]:
        """Download a chunk from the database for a miner that will be used for butterfly all reduce merging.

        Args:
            weights_path (str): The path to the weights
            metadata_path (str): The path to the metadata
            chunk_id (int | str): The chunk id
        """

        if isinstance(chunk_id, str):
            assert chunk_id == "all"

        # cache this (but its tiny)
        metadata: dict[str, Any] = download_metadata(metadata_path)

        # get the chunk in the metadata with the correct chunk_id
        available_chunk_ids = list(int(k) for k in metadata["sections"].keys())
        if chunk_id == "all":
            chunk_start_idx = metadata["sections"][str(min(available_chunk_ids))]["start_idx"]
            chunk_end_idx = metadata["sections"][str(max(available_chunk_ids))]["end_idx"]
            chunk_start_byte = metadata["sections"][str(min(available_chunk_ids))]["start_byte"]
            chunk_end_byte = metadata["sections"][str(max(available_chunk_ids))]["end_byte"]
        else:
            chunk = metadata["sections"][str(chunk_id)]
            chunk_start_idx = chunk["start_idx"]
            chunk_end_idx = chunk["end_idx"]
            chunk_start_byte = chunk["start_byte"]
            chunk_end_byte = chunk["end_byte"]

        chunk_data = ChunkData(
            chunk_start_idx=chunk_start_idx,
            chunk_end_idx=chunk_end_idx,
            chunk_start_byte=chunk_start_byte,
            chunk_end_byte=chunk_end_byte,
            chunk_dtype=metadata["tensor"]["dtype"].split(".")[-1],
            chunk_length=chunk_end_idx - chunk_start_idx,
        )

        partition = Partition(
            layer=self.layer,
            chunk_number=chunk_id,
            weight_path=data_path,
            weight_metadata_path=metadata_path,
            miner_hotkey=self.hotkey,
            weight_data=chunk_data if data_type == "weights" else ChunkData(),
            optimizer_state_data=chunk_data if data_type == "optimizer_state" else ChunkData(),
        )
        # only download the chunk we need: we want to form an s3 query which includes the start and end indices
        logger.debug(f"DOWNLOADING CHUNK FOR MERGING: {partition}")
        weights = download_weights_or_optimizer_state(data_path, partition=partition, data_type=data_type)
        return weights, metadata

    async def download_weights(self):
        # Now the miners have to download the shards of the final merged layer weights. These weights exist across multiple files and are 1D byte strings
        try:
            logger.debug(f"Downloading weights for layer {self.layer} for miner {self.hotkey[:8]}")
            merged_partitions: list[Partition] = await self.api_client.get_layer_weights(layer=self.layer)

            # Allocate memory to the full 1d tensor
            new_weights = torch.nn.utils.parameters_to_vector(self.model.parameters())

            # Set random gradients
            add_artificial_gradients(self.model)

            # Take a step to populate internal state
            self.optimizer.step()
            self.optimizer.zero_grad()
            flat_tensor, tensor_shapes, state_dict = flatten_optimizer_state(self.optimizer)
            # Convert to numpy array
            new_optimizer_state = flat_tensor  # .to(torch.float16).detach().cpu().numpy(force=True)

            for partition in merged_partitions:
                logger.info(
                    f"Downloading shard {partition.weight_path!r} and metadata {partition.weight_metadata_path!r}"
                )
                weight_shard = download_weights_or_optimizer_state(
                    partition.weight_path, partition=partition, data_type="weights"
                )
                shard_optimizer_state = download_weights_or_optimizer_state(
                    partition.optimizer_state_path, partition=partition, data_type="optimizer_state"
                )

                new_weights[partition.weight_data.chunk_start_idx : partition.weight_data.chunk_end_idx] = weight_shard
                new_optimizer_state[
                    partition.optimizer_state_data.chunk_start_idx : partition.optimizer_state_data.chunk_end_idx
                ] = shard_optimizer_state

            # assign weights to self.model
            # reshape thecomplete 1D tensor into the appropriate shape
            self.weights = new_weights
            new_optimizer_state = reconstruct_optimizer_state(
                new_optimizer_state, tensor_shapes, state_dict, self.optimizer
            )
            torch.nn.utils.vector_to_parameters(new_weights, self.model.parameters())
            logger.debug(f"Successfully applied weights to model for layer {self.layer}")
            return new_weights, new_optimizer_state
        except Exception as e:
            logger.exception(f"Error downloading weights: {e}")
            raise

    async def initialize(self):
        """Async initialization that must be called after construction"""
        await self._init_bittensor()
        # Initialize API client
        self.api_client = APIClient(wallet=self.wallet)
        await self.api_client.__aenter__()
        # Network setup
        self.netuid = settings.netuid
        self.subtensor = get_subtensor()
        self.lock = threading.RLock()

        self.metagraph_syncer = MetagraphSyncer(subtensor=self.subtensor)
        self.metagraph_syncer.do_initial_sync()
        self.metagraph_syncer.start()

        if self.metagraph_syncer and self.netuid is not None:
            # Get initial metagraph
            self.metagraph = self.metagraph_syncer.get_metagraph(self.netuid)
            # Register listener for metagraph updates
            logger.info(f"Neuron initialized with metagraph syncing for uid {self.uid} on netuid {self.netuid}")
        if BITTENSOR:
            try:
                self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            except ValueError:
                raise NotRegisteredError(f"Hotkey {self.wallet.hotkey.ss58_address} not registered on {self.netuid}")
        else:
            self.uid = random.randint(0, 255)

        self.hotkey = self.wallet.hotkey.ss58_address
        logger.info(f"Initialized with wallet: {self.wallet}")

    async def _init_bittensor(self):
        if not BITTENSOR:
            logger.warning("Bittensor is not enabled, using mock wallet")
            self.wallet = get_mock_wallet()
            self.api_client = APIClient(wallet=self.wallet)
            await self.api_client.__aenter__()
            logger.info(f"Initialized with mock wallet: {self.wallet}")
            return
        else:
            logger.info(
                f"Initializing Bittensor components with wallet name: {settings.wallet_name} and hotkey: {settings.wallet_hotkey} on network: {settings.network} and netuid: {settings.netuid}"
            )
            if self.wallet_name is None or self.wallet_hotkey is None:
                self.wallet_name = settings.wallet_name
                self.wallet_hotkey = settings.wallet_hotkey
                self.wallet = bt.wallet(name=self.wallet_name, hotkey=self.wallet_hotkey)
            else:
                self.wallet = bt.wallet(name=self.wallet_name, hotkey=self.wallet_hotkey)
            self.api_client = APIClient(wallet=self.wallet)
            await self.api_client.__aenter__()
            logger.info(
                f"Bittensor initialized with metagraph: {self.metagraph} and hotkey: {self.wallet.hotkey} in {self.wallet} "
            )

    async def _forward(self, input_activations: torch.Tensor):
        if self.layer is not None and self.layer > 0:
            input_activations.requires_grad_(True)
        self.model.to(DEVICE)
        output_activations = self.model(input_activations.to(DEVICE))
        return output_activations

    async def _backward(
        self,
        output_activations: torch.Tensor,
        activation_grads: torch.Tensor,
        state: dict,
    ):
        # If this is the last layer, then output_activations is the loss
        if self.layer == settings.N_LAYERS - 1:
            try:
                output_activations.backward()
            except RuntimeError as e:
                logger.error(f"Error during backward step: {e}")
                traceback.print_exc()
                raise
        else:
            try:
                self.model.backward(output_activations, activation_grads, state)
            except RuntimeError as e:
                logger.error(f"Error during backward step: {e}")
                traceback.print_exc()
                raise
        if self.backwards_since_reduce >= settings.LOCAL_OPTIMIZER_STEPS:
            # Clip the gradients
            await self.clip_gradients()

    async def clip_gradients(self):
        if self.total_model_params is None:
            self.total_model_params = sum(p.numel() for p in self.model.parameters())

        split_grad_norm = settings.GRAD_CLIP_NORM * math.sqrt(
            self.total_model_params / settings.MODEL_CFG["total_global_params"]
        )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), split_grad_norm)

    async def _load_model(self):
        if MOCK:
            logger.info("Mock mode enabled - loading mock model")
            self.model = MockModel()
            self.model.train()
            return
        logger.info(f"MODEL_SPLITS: {MODEL_SPLITS}, LAYER: {self.layer}")
        logger.info(f"Loading model from {MODEL_CFG['model_name']} with split {MODEL_SPLITS[self.layer]}")

        try:
            self.model = load_model_split(
                model_cfg=MODEL_CFG,
                model_split=MODEL_SPLITS[self.layer],
                device=DEVICE,
                seed=42,
            )
            # put the model in train mode
            self.model.train()

            # forward pass to populate bottleneck decoder in the case where
            # the bottleneck dynamically changes it size based on the input data.
            if self.layer > 0:
                logger.success(f"Populating bottleneck decoder for layer {self.layer}")
                self.model.forward(
                    torch.zeros(
                        settings.BATCH_SIZE,
                        settings.SEQUENCE_LENGTH,
                        MODEL_CFG["bottleneck_dim"] or MODEL_CFG["emb_dim"],
                    ).to(DEVICE)
                )

        except ValueError as e:
            logger.exception(f"{e}")
        except Exception as e:
            logger.exception(f"Error loading model: {e}")

        # log the number of parameters
        logger.info(f"Number of parameters in the model: {sum(p.numel() for p in self.model.parameters()) / 1e9}B")

    async def _load_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        add_artificial_gradients(self.model)
        self.optimizer.step()
        self.optimizer.zero_grad()

        logger.info(f"Loaded optimizer with learning rate {LEARNING_RATE} and weight decay {WEIGHT_DECAY}")

    async def _load_lr_scheduler(self):
        """
        Setting the learning rate schedulers

        TODO: scheduler milestone should be probably dictated by the orchestrator
        """
        # -------------------------------------------------------------
        # ─── hyper-parameters from settings.py ───────────────────────
        # -------------------------------------------------------------
        warm_steps = settings.LR_WARMUP_STEPS  # e.g. 2_500
        plateau_steps = settings.LR_CONST_STEPS  # e.g. 5_000 (0 → none)
        total_steps = settings.TOTAL_TRAIN_STEPS  # e.g. 125_000
        tail_frac = settings.LR_TAIL_STEPS_FRAC  # 0.02  (=  2 %)
        start_fac = settings.LR_WARMUP_START_FACTOR  # 0.002
        final_fac = settings.LR_FINAL_FACTOR  # 0.10  (= 10 %)

        tail_steps = int(total_steps * tail_frac)
        decay_steps = total_steps - warm_steps - plateau_steps - tail_steps
        assert decay_steps > 0, "decay phase length would be negative"

        # -------------------------------------------------------------
        # 0) linear warm-up 0 → LRpeak
        # -------------------------------------------------------------
        sched_warm = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_fac,
            end_factor=1.0,
            total_iters=warm_steps,
        )

        # -------------------------------------------------------------
        # 1) constant plateau at LRpeak (optional)
        # -------------------------------------------------------------
        if plateau_steps:
            sched_plateau = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)
        else:
            sched_plateau = None

        # -------------------------------------------------------------
        # 2) cosine decay LRpeak → final_fac·LRpeak
        # -------------------------------------------------------------
        def cos_decay(step):
            p = step / decay_steps
            return final_fac + (1.0 - final_fac) * 0.5 * (1 + math.cos(math.pi * p))

        sched_decay = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cos_decay)

        # -------------------------------------------------------------
        # 3) tail anneal: cosine from final_fac·LRpeak → 0
        # -------------------------------------------------------------
        def cos_tail(step):
            p = step / tail_steps
            return final_fac * 0.5 * (1 + math.cos(math.pi * p))  # goes to 0

        sched_tail = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cos_tail)

        # -------------------------------------------------------------
        # build SequentialLR
        # -------------------------------------------------------------
        scheds = [sched_warm]
        milestones = [warm_steps]

        if sched_plateau is not None:
            scheds.append(sched_plateau)
            milestones.append(milestones[-1] + plateau_steps)

        scheds.append(sched_decay)
        milestones.append(milestones[-1] + decay_steps)

        print(f"scheds: {scheds}")
        print(f"milestones: {milestones}")
        print(
            f"total_steps: {total_steps}, decay_steps: {decay_steps}, warm_steps: {warm_steps}, plateau_steps: {plateau_steps}, tail_steps: {tail_steps}"
        )
        scheds.append(sched_tail)  # final cosine-to-zero phase

        self.lr_scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=scheds, milestones=milestones)

        logger.info(
            f"LR schedule: warm-up {warm_steps} → plateau {plateau_steps} → "
            f"cosine {decay_steps} → cosine-to-zero tail {tail_steps} steps."
        )

    async def _load_lr_scheduler_2(self):
        """
        Here are the stages of this scheduler:
        0. linear warm-up 0 → 1 × LRpeak
        1. constant plateau at LRpeak (optional)
        2. macro-cosine × micro-saw-tooth
        3. tail cosine to zero
        """
        # ─── hyper-parameters from settings.py ────────────────────────────
        warm_steps = settings.LR_WARMUP_STEPS  # e.g. 3_500
        plateau_steps = settings.LR_CONST_STEPS  # e.g.   500
        total_steps = settings.TOTAL_TRAIN_STEPS  # e.g. 100_000
        tail_frac = settings.LR_TAIL_STEPS_FRAC  # 0.02 (2 %)
        start_fac = settings.LR_WARMUP_START_FACTOR  # 0.002
        final_fac = settings.LR_FINAL_FACTOR  # 0.10
        cycle_length = settings.LR_SAW_CYCLE_LENGTH  # e.g. 10_000
        # if you prefer "N cycles", set cycle_length = decay_steps // N

        tail_steps = int(total_steps * tail_frac)
        decay_steps = total_steps - warm_steps - plateau_steps - tail_steps
        assert decay_steps > 0, "decay phase would be zero/negative"

        # ─── phase-0: linear warm-up 0 → 1 × LRpeak ───────────────────────
        sched_warm = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=start_fac,
            end_factor=1.0,
            total_iters=warm_steps,
        )

        # ─── phase-1: constant plateau at LRpeak (optional) ───────────────
        sched_plateau = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0) if plateau_steps else None

        # ─── phase-2: macro-cosine × micro-saw-tooth  ────────────────────
        def combined_lambda(step):
            """
            step counts from 0 … decay_steps-1 inside the decay phase
            return LR multiplier ∈ [0, 1]
            """
            # ----- macro envelope  LRpeak → final_fac·LRpeak --------------
            macro_p = step / decay_steps
            macro = final_fac + (1.0 - final_fac) * 0.5 * (1 + math.cos(math.pi * macro_p))

            # ----- micro cosine-restart 1 → 0.1 → 1 every cycle_length ----
            cycle_p = (step % cycle_length) / cycle_length
            micro = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * cycle_p))
            # micro ∈ [0.1, 1]

            return macro * micro  # overall multiplier

        sched_saw = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=combined_lambda)

        # ─── phase-3: tail cosine to zero ─────────────────────────────────
        def tail_lambda(step):
            p = step / tail_steps
            return final_fac * 0.5 * (1 + math.cos(math.pi * p))  # ↘ 0

        sched_tail = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=tail_lambda)

        # ─── stitch phases together ──────────────────────────────────────
        schedulers = [sched_warm]
        milestones = [warm_steps]

        if sched_plateau:
            schedulers.append(sched_plateau)
            milestones.append(milestones[-1] + plateau_steps)

        schedulers += [sched_saw, sched_tail]
        milestones += [milestones[-1] + decay_steps]  # (= total)

        self.lr_scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=schedulers, milestones=milestones
        )

        logger.info(
            f"LR schedule\n"
            f"  warm-up   : 0–{warm_steps-1}\n"
            f"  plateau   : {warm_steps}–{warm_steps+plateau_steps-1}\n"
            f"  saw-tooth : {milestones[-2]-decay_steps}–{milestones[-2]-1} "
            f"(cycle_length={cycle_length})\n"
            f"  tail      : {milestones[-2]}–{total_steps-1}"
        )

    async def _load_tokenizer(self):
        if MOCK:
            logger.info("Mock mode enabled - skipping tokenizer load")
            return

        logger.info(f"Loading tokenizer from {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)

    async def _load_dataloader(self):
        if MOCK:
            logger.info("Mock mode enabled - skipping dataloader load")
            return

        logger.info(
            f"Loading dataloader from {DATASET_NAME} "
            f"with batch size {BATCH_SIZE} "
            f"and sequence length {SEQUENCE_LENGTH} "
            f"and pack_samples set to: {PACK_SAMPLES}"
        )
        self.dataloader = load_dataloader(
            dataset_name=DATASET_NAME,
            tokenizer=self.tokenizer,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            pack_samples=PACK_SAMPLES,
        )

        logger.info("Loaded dataloader.")

    async def _load_vocab_info(self):
        if MOCK:
            logger.info("Mock mode enabled - using mock vocab info")
            self.vocab_size = 100
            self.eos_token_id = 1
            return

        if self.tokenizer is None:
            # The tokenizer is a local variable, so it will garbage collected
            # after this function is called. This happens when the miner is
            # in last stage of the swarm, since we don't need the tokenizer
            # anymore.
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        else:
            tokenizer = self.tokenizer

        self.vocab_size = len(tokenizer)
        self.eos_token_id = tokenizer.eos_token_id
        logger.info(f"loaded vocab info: vocab size | {self.vocab_size} | EOS token id | {self.eos_token_id}")

    async def _load_data(self):
        if MOCK:
            mock_data = torch.randn(100, 100).to(DEVICE).to(torch.bfloat16)
            logger.info(f"Generated mock data sample of shape {mock_data.shape}")
            return mock_data

        data_sample = next(self.dataloader)
        logger.info(f"Loaded data sample of shape {data_sample.shape}")

        return data_sample.to(DEVICE)

    async def local_all_reduce(self):
        logger.info(f"{self.hotkey} updating weights after {self.backwards_since_reduce} steps")
        # if not settings.MOCK:
        logger.warning(f"{self.hotkey} is stepping")

        self.optimizer.step()
        self.lr_scheduler.step()
        logger.info(f"{self.uid} learning rate: {self.optimizer.param_groups[0]['lr']}")
        self.optimizer.zero_grad()
        self.completed_optim_steps += 1

        self.backwards_since_reduce = 0
        self.forwards_since_reduce = 0
        self.processed_forward_activations = []
        self.processed_backward_activations = []
        self.saved_forward_activations = {}

        # Log GPU memory after weight update
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            logger.debug(f"GPU memory after local all reduce: {allocated:.2f}GB")

    @property
    def block(self):
        return self.subtensor.get_current_block(self.wallet.hotkey)
