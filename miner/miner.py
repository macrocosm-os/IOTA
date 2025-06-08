import asyncio
import time
import uuid
import json
import io
import gc
from typing import Literal, Any, Optional

import torch
from loguru import logger
import wandb
from aiohttp import web

import model.utils as model_utils
from base.base_neuron import BaseNeuron
from miner.api_client import APIClient
import settings
from utils.s3_interactions import (
    download_weights_or_optimizer_state,
    download_activation,
    upload_to_bucket,
    smart_upload_via_orchestrator_async,
)
from utils.shared_states import MergingPhase
from utils.partitions import ChunkData, Partition
from utils.vector_utils import flatten_optimizer_state
from orchestrator.serializers import SubmittedWeights
from storage.serializers import ActivationResponse


WAIT_TIME = 5 if settings.MOCK else 15


class Miner(BaseNeuron):
    TIMEOUT: int = settings.TIMEOUT
    epoch: int = 0
    health_app_runner: Optional[web.AppRunner] = None
    health_site: Optional[web.TCPSite] = None
    reregister_needed: bool = True
    training: bool = True

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

    def _clean_gpu_memory(self):
        """Overrides the base neuron's cleanup to include miner-specific caches."""
        logger.debug("🗑️ Running miner-specific GPU memory cleanup. 🗑️")

        # Clear activation cache
        if hasattr(self, "saved_forward_activations"):
            logger.debug(f"Clearing {len(self.saved_forward_activations)} cached activations.")

        self._clear_saved_forward_activations()

        # Clear processed activation lists
        if hasattr(self, "processed_forward_activations"):
            self.processed_forward_activations.clear()
            logger.debug("Cleared processed forward activations list.")

        if hasattr(self, "processed_backward_activations"):
            self.processed_backward_activations.clear()
            logger.debug("Cleared processed backward activations list.")

        # Call parent cleanup
        super()._clean_gpu_memory()

    @property
    async def out_of_cache(self):
        if ooc := len(self.saved_forward_activations) >= self.MAX_ACTIVATION_CACHE_SIZE:
            logger.debug(
                f"Miner {self.hotkey} cache full with {len(self.saved_forward_activations)} activations: {self.saved_forward_activations.keys()}"
            )

            # Clean up inactive activations
            activations_to_remove = []
            for activation_uid in list(self.saved_forward_activations.keys()):
                try:
                    if self.api_client and not await self.api_client.is_activation_active(
                        layer=self.layer, activation_uid=activation_uid
                    ):
                        activations_to_remove.append(activation_uid)
                except Exception as e:
                    logger.warning(f"Error checking activation {activation_uid} status: {e}")
                    # If we can't check, keep the activation to be safe

            # Remove inactive activations
            for activation_uid in activations_to_remove:
                if activation_uid in self.saved_forward_activations:
                    # Clean up tensors before removing from cache
                    cached_data = self.saved_forward_activations[activation_uid]
                    del cached_data  # This will help with garbage collection
                    del self.saved_forward_activations[activation_uid]
                    logger.debug(f"Removed inactive activation {activation_uid} from cache")

            # Update out_of_cache status after cleanup
            ooc = len(self.saved_forward_activations) >= self.MAX_ACTIVATION_CACHE_SIZE

            logger.debug(
                f"Miner {self.hotkey} cache status: {len(self.saved_forward_activations)}/{self.MAX_ACTIVATION_CACHE_SIZE} activations cached, out_of_cache: {ooc}"
            )
        return ooc

    @classmethod
    async def create(cls, wallet_name: str, wallet_hotkey: str, timeout: int, n_layers: int):
        miner = cls(
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            TIMEOUT=timeout,
            N_LAYERS=n_layers,
        )
        await miner.initialize()
        return miner

    @property
    def has_layer(self):
        return self.layer is not None

    async def load_model(self):
        """
        Loads the model, weights, optimizer, tokenizer, dataloader, and vocab info
        """
        try:
            # Ensure previous model artifacts are cleared before loading a new one
            self._clean_gpu_memory()

            # Check GPU memory before loading model
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory before model load: {allocated_memory:.2f}GB / {total_memory:.2f}GB")

                if allocated_memory > total_memory * 0.8:  # If more than 80% already used
                    logger.warning(f"High GPU memory usage detected before model load: {allocated_memory:.2f}GB")

            # Load the model
            await self._load_model()
            await self._load_optimizer()
            await self._load_lr_scheduler_2()

            # Load the tokenizer if this is the first
            if self.layer == 0:
                await self._load_tokenizer()
                await self._load_dataloader()

            # If this is the first or last stage, get the vocab info
            if self.layer == 0 or self.layer == settings.N_LAYERS - 1:
                await self._load_vocab_info()

            # Final memory check after loading
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory after model load: {allocated_memory:.2f}GB")

        except Exception as e:
            logger.exception(f"Error loading model: {e}")
            raise

    async def _start_health_server(self):
        """Starts the aiohttp web server for healthchecks."""
        app = web.Application()

        async def health_handler(request):
            return web.json_response(
                {
                    "status": "healthy",
                    "hotkey": getattr(self, "hotkey", "N/A"),
                    "layer": getattr(self, "layer", "N/A"),
                    "uid": getattr(self, "uid", "N/A"),
                    "registered": getattr(self, "reregister_needed", True) is False,
                    "timestamp": time.time(),
                }
            )

        app.router.add_get(settings.MINER_HEALTH_ENDPOINT, health_handler)

        self.health_app_runner = web.AppRunner(app)
        await self.health_app_runner.setup()

        self.health_site = web.TCPSite(self.health_app_runner, settings.MINER_HEALTH_HOST, settings.MINER_HEALTH_PORT)
        if settings.LAUNCH_HEALTH:
            await self.health_site.start()
            logger.info(
                f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API started on "
                f"http://{settings.MINER_HEALTH_HOST}:{settings.MINER_HEALTH_PORT}{settings.MINER_HEALTH_ENDPOINT}"
            )

    async def _stop_health_server(self):
        """Stops the aiohttp web server for healthchecks."""
        if self.health_site:
            await self.health_site.stop()
            logger.info(f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API site stopped.")
            self.health_site = None
        if self.health_app_runner:
            await self.health_app_runner.cleanup()
            logger.info(f"Miner {getattr(self, 'hotkey', 'N/A')} healthcheck API runner cleaned up.")
            self.health_app_runner = None

    async def run(self):
        logger.info(f"🚀 Starting miner {self.hotkey[:8]} | Timeout: {self.TIMEOUT}s")

        self.reregister_needed = True

        try:
            # Start the healthcheck server
            if settings.LAUNCH_HEALTH:
                await self._start_health_server()
                logger.info(f"🏥 Health server started for miner {self.hotkey[:8]}")
            else:
                logger.warning(
                    "⚠️ Miner healthcheck API not configured in settings (MINER_HEALTH_PORT missing). Skipping."
                )

            start = time.time()

            while not settings.MOCK or time.time() - start < self.TIMEOUT:
                try:
                    if (
                        self.api_client.failed_api_request
                        and not await self.api_client.health_check()
                        and not self.reregister_needed
                    ):
                        logger.warning(f"🏥 Health check failed for miner {self.hotkey[:8]}, reregistering")
                        self.reregister_needed = True
                        await asyncio.sleep(10)
                        continue

                    # If any of the requests failed during the loop, failed_api_request is set to True, which will trigger the miner to do a health check and reregister if needed
                    self.api_client.failed_api_request = False

                    if not self.api_client:
                        logger.debug(f"🔗 Miner {self.hotkey[:8]} initializing API client")
                        self.api_client = APIClient(wallet=self.wallet)
                        await self.api_client.__aenter__()

                    if self.reregister_needed:
                        try:
                            logger.info(f"🔄 Reregistering miner {self.hotkey[:8]}")
                            await self.register()

                            await self.load_model()

                            self.reregister_needed = False

                            continue
                        except Exception as e:
                            logger.error(f"❌ Error reregistering miner {self.uid}: {e}")
                            continue

                    if self.layer is None:
                        try:
                            response = await self.api_client.request_layer()
                            self.layer = response.layer
                            logger.info(f"🎯 Miner {self.uid} assigned to layer {self.layer}")

                            # Load the model
                            await self._load_model()

                            # if there are global weights, download them
                            try:
                                weights_path = await self.api_client.get_layer_weights(self.layer)
                            except Exception as e:
                                logger.warning(f"⚠️ No weights found for layer {self.layer}, skipping")
                                weights_path = None

                            if weights_path:
                                logger.info(f"📥 Loading existing weights for layer {self.layer}")
                                self.weights = download_weights_or_optimizer_state(weights_path)
                                assert isinstance(
                                    self.weights, torch.Tensor
                                ), f"Weights must on {weights_path} be a torch.Tensor but are {type(self.weights)}"
                                # assign weights to self.model
                                torch.nn.utils.vector_to_parameters(self.weights, self.model.parameters())
                                logger.info(f"✅ Loaded existing weights for layer {self.layer}")
                            else:
                                # if there are no global weights, generate random weights
                                self.weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
                                logger.info(f"🎲 Generated random weights for layer {self.layer}")
                            logger.debug(f"Miner {self.uid} has weights: {self.weights}")
                            await self._load_optimizer()
                            await self._load_lr_scheduler_2()

                            # Load the tokenizer if this is the first
                            if self.layer == 0:
                                await self._load_tokenizer()
                                await self._load_dataloader()

                            # If this is the first or last stage, get the vocab info
                            if self.layer == 0 or self.layer == settings.N_LAYERS - 1:
                                await self._load_vocab_info()

                        except Exception as e:
                            logger.error(f"❌ Error loading model for miner {self.hotkey[:8]}: {e}")
                            continue

                    else:
                        # Final memory check after loading
                        if torch.cuda.is_available():
                            allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                            logger.debug(f"💾 GPU memory: {allocated_memory:.2f}GB")

                        if settings.N_LAYERS == 1:
                            await self.local_step()
                        else:
                            await self.step()

                    if self.reregister_needed:
                        await asyncio.sleep(10)
                    await self.print_status()

                except Exception as e:
                    logger.exception(f"❌ Main loop failed for miner {self.hotkey[:8]}: {e}")

            raise Exception(
                f"⏰ Miner {self.hotkey[:8]} timed out on run loop: Timeout: {self.TIMEOUT} and time: {time.time() - start}"
            )

        except Exception as e:
            logger.exception(f"❌ Miner {self.hotkey[:8]} failed: {e}")
        finally:
            # Stop the healthcheck server
            if settings.MINER_HEALTH_PORT:
                await self._stop_health_server()
                logger.info(f"🏥 Health server stopped for miner {self.hotkey[:8]}")

            if self.api_client:
                await self.api_client.__aexit__(None, None, None)

            # Final memory cleanup only on shutdown
            self._clean_gpu_memory()

    async def print_status(self):
        logger.debug(f"📊 Miner {self.hotkey[:8]} status | Layer: {self.layer} | Epoch: {self.epoch}")
        # Status is now managed through API calls, so we don't need to access orchestrator directly

    async def step(self):
        # self.saved_forward_activations
        # Check if we're merging
        if self.saved_forward_activations:
            for activation_uid, activation_data in list(self.saved_forward_activations.items()):
                upload_time = activation_data[-1]
                if upload_time < time.time() - settings.ACTIVATION_CACHE_TIMEOUT:
                    del self.saved_forward_activations[activation_uid]
                    logger.warning(
                        f"🗑️ Removed activation {activation_uid} from miner {self.hotkey[:8]} cache due to timeout"
                    )

        if not self.training:
            result = await self.api_client.merge_info(layer=self.layer)
            logger.info(f"🔄 Miner {self.hotkey[:8]} entering merging phase: {result}")

            # Clear cache before weight syncing
            self._clear_saved_forward_activations()

            try:
                await self.sync_weights(num_sections=int(result["num_sections"]))
                self.training = True
                return
            except Exception as e:
                logger.exception(f"❌ Error syncing weights for miner {self.hotkey[:8]}: {e}")
                await asyncio.sleep(WAIT_TIME)
                raise

        if (
            self.backwards_since_reduce >= settings.LOCAL_OPTIMIZER_STEPS
            and settings.LOCAL_OPTIMIZER_STEPS < settings.GLOBAL_OPTIMIZER_STEPS
        ):
            logger.info(
                f"🔄 Performing local all-reduce for miner {self.hotkey[:8]} | Steps: {self.backwards_since_reduce}"
            )
            await self.local_all_reduce()
            self._clear_saved_forward_activations()

            self.backwards_since_reduce = 0
            return

        if not self.has_layer:
            raise Exception("Layer is not set")

        activation_response: ActivationResponse = await self.api_client.get_random_activation()
        if activation_response.reason == "not training":
            logger.warning(
                f"⏸️ Miner {self.hotkey[:8]} on layer {self.layer} is merging based on get_random_activation reason, skipping forward step"
            )
            await asyncio.sleep(1)
            self.training = False
            return

        if activation_response.direction is not None:
            if activation_response.direction == "forward":
                return await self.forward(activation=activation_response)
            elif activation_response.direction == "backward":
                return await self.backward(activation=activation_response)

        if self.layer == 0:
            if activation_response.reason == "out_of_cache":
                logger.info(f"⏸️ Miner {self.hotkey[:8]} on layer {self.layer} is out of cache, skipping forward step")
                await asyncio.sleep(1)
                return
            return await self.forward()

        logger.debug(f"⏸️ Miner {self.hotkey[:8]} on layer {self.layer} is idle, no activations are ready... waiting")
        await asyncio.sleep(1)

    async def local_step(self):
        if self.layer is None:
            raise Exception("Layer is not set")

        logger.info(
            f"🔄 Performing local step for single-layer training | Miner: {self.hotkey[:8]} | Layer: {self.layer}"
        )

        input_activations = await self._load_data()
        activation_uid = str(uuid.uuid4())

        output_activations, state = self.model(input_activations)

        """
        if 'bottlenecks' in state and self.completed_optim_steps % 5000 == 0:
            for i, bottleneck_x in enumerate(state['bottlenecks']):
                bottleneck_x.retain_grad()
                bottleneck_x_path = create_activation_path(
                    uid=str(self.completed_optim_steps).zfill(10), layer=i, direction="forward"
                )
                upload(bottleneck_x_path, bottleneck_x.detach().clone())
        """
        loss = model_utils.compute_loss(
            logits=output_activations,
            targets=input_activations,
            vocab_size=self.vocab_size,
            pad_token_id=self.eos_token_id,
            pack=settings.PACK_SAMPLES,
        )
        loss.backward()

        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), settings.GRAD_CLIP_NORM)

        logger.info(f"📊 Local step loss: {loss.item():.6f} | Activation: {activation_uid} | Miner: {self.hotkey[:8]}")

        if settings.USE_WANDB:
            metrics = {"loss": float(loss.item())}
            await self._log_wandb(metrics)

        """
        if 'bottlenecks' in state and self.completed_optim_steps % 5000 == 0:
            for i, bottleneck_x in enumerate(state['bottlenecks']):
                # Save the activation gradients
                #if bottleneck_x.grad is not None:
                bottleneck_grad_path = create_activation_path(
                    uid=str(self.completed_optim_steps).zfill(10), layer=i, direction="backward"
                )
                upload(bottleneck_grad_path, bottleneck_x.grad.detach().clone())

        """

        await self.api_client.report_loss(activation_uid, float(loss.item()))
        await self.local_all_reduce()

        logger.info(f"✅ Local step completed | Miner: {self.hotkey[:8]}")

    async def register(self):
        """
        Registers the miner with the orchestrator and returns the layer assigned to the miner
        """
        logger.info(f"🔗 Registering miner {self.hotkey[:8]} with orchestrator...")
        self.layer: int = await self.api_client.register()
        logger.info(f"✅ Successfully registered miner {self.hotkey[:8]} | Assigned to layer: {self.layer}")

    async def await_orchestrator_status(self, status: MergingPhase):
        while True:
            logger.info(f"⏳ Miner {self.hotkey[:8]} in epoch {self.epoch} waiting for orchestrator status: {status}")
            status_response = await self.api_client.merge_info(layer=self.layer)
            if status_response.get("status") == status.value:
                logger.info(f"✅ Orchestrator status reached: {status}")
                break
            await asyncio.sleep(1.5 if settings.MOCK else 10)

    async def sync_weights(self, num_sections: int):
        """Syncs the weights of the miner with the orchestrator.

        Args:
            num_sections (int): The number of sections to sync.

        Raises:
            Exception: If the miner is not registered on the metagraph.
        """
        if not self.has_layer:
            logger.warning(f"⚠️ Miner {self.hotkey[:8]} cannot sync weights: layer not assigned")
            return

        flattened_optimizer_state = None
        weights = None

        try:
            logger.info(
                f"🔄 Starting weight sync for miner {self.hotkey[:8]} | Layer: {self.layer} | Epoch: {self.epoch} | Steps since sync: {self.backwards_since_sync}"
            )

            # Clear cache before weight syncing
            self._clear_saved_forward_activations()

            # If local optimizer steps are smaller than global optimizer steps, we already handle them in step()
            if settings.LOCAL_OPTIMIZER_STEPS >= settings.GLOBAL_OPTIMIZER_STEPS:
                await self.local_all_reduce()
                self.backwards_since_reduce = 0

            flattened_optimizer_state, _, _ = flatten_optimizer_state(self.optimizer)
            weights = torch.nn.utils.parameters_to_vector(self.model.parameters())

            logger.info(f"📤 Uploading weights and optimizer state | Sections: {num_sections}")
            (
                weight_path,
                weight_metadata_path,
                optimizer_state_path,
                optimizer_state_metadata_path,
            ) = await self.upload_weights_with_metadata_and_optimizer_state(
                flattened_weights=weights,
                flattened_optimizer_state=flattened_optimizer_state,
                layer=self.layer,
                miner_hotkey=self.hotkey,
                num_sections=num_sections,
                epoch=self.epoch,
            )
            logger.debug(
                f"Epoch {self.epoch}: Uploading weights to {weight_path} and metadata to {weight_metadata_path} and optimizer state to {optimizer_state_path} and metadata to {optimizer_state_metadata_path}"
            )
            logger.debug(f"EPOCH {self.epoch}: UPLOADING WEIGHTS: {self.hotkey} | {self.layer} | {weights}")

            logger.info("📤 Notifying orchestrator of uploaded weights...")
            success = (
                await self.api_client.notify_weights_uploaded(
                    weights_path=weight_path,
                    metadata_path=weight_metadata_path,
                    optimizer_state_path=optimizer_state_path,
                    optimizer_state_metadata_path=optimizer_state_metadata_path,
                )
            )["success"]
            if success != "success":
                logger.error(f"❌ Miner {self.hotkey[:8]} failed to upload weights to orchestrator")
                raise Exception(f"Miner {self.hotkey[:8]} failed to upload weights to orchestrator. Error: {success}")

            logger.info("✅ Successfully notified orchestrator of weight upload")

            # TODO: start downloading other miners parititons as they become available
            # TODO: make a placeholder for the polling process
            await self.await_orchestrator_status(status=MergingPhase.MINERS_MERGING_PARTITIONS)

            logger.info("📥 Getting weight partition info for merging...")
            information_packets, partition_ids = await self.api_client.weight_partition_info()
            information_packets = [SubmittedWeights(**packet) for packet in information_packets]

            # download the partitons, apply the merge (simple avg) and upload the merged weights
            logger.info(f"🔄 Merging {len(information_packets)} weight partitions...")
            partitions: list[Partition] = await self._merge_models(
                information_packets=information_packets,
                partition_ids=partition_ids,
            )

            logger.info("📤 Notifying orchestrator of merged partitions...")
            logger.debug(
                f"Miner {self.hotkey[:8]} notifying orchestrator of merged partitions: {partitions}, epoch {self.epoch}"
            )
            await self.api_client.notify_merged_partitions_uploaded(partitions=partitions)
            logger.info("✅ Successfully uploaded merged partitions")

            await self.await_orchestrator_status(status=MergingPhase.IS_TRAINING)

            # Once merging is complete, we download the layer weights and update the model
            logger.info("📥 Downloading merged weights...")
            self.weights, self.optimizer = await self.download_weights()
            logger.info("✅ Successfully downloaded merged weights")
            logger.warning(f"WEIGHTS DOWNLOADED: {self.weights}")

            self.backwards_since_sync = 0
            self.epoch += 1

            logger.info(f"✅ Weight sync completed | Epoch: {self.epoch} | Miner: {self.hotkey[:8]}")

        finally:
            # Guaranteed cleanup of large temporary tensors and GPU cache.
            # This block executes regardless of whether the try block succeeds or fails.
            logger.info("🗑️ Entering guaranteed cleanup block for sync_weights.")
            try:
                if flattened_optimizer_state is not None:
                    del flattened_optimizer_state
                if weights is not None:
                    del weights

                gc.collect()
                torch.cuda.empty_cache()

                logger.info("🗑️ Forced cleanup of temporary tensors and GPU cache completed.")
            except NameError:
                # This can happen if the function errored out before the tensors were created.
                pass

    async def _merge_models(
        self, information_packets: list[SubmittedWeights], partition_ids: list[int]
    ) -> list[Partition]:
        """Merge the models from the other miners.

        Args:
            information_packets (list[SubmittedWeights]): The paths to the other miners' partitions
            partition_ids (list[int]): The partition indices to merge

        Returns:
            list[Partition]: The merged partitions
        """

        partitions: list[Partition] = []

        # Loop over the partition indices one at a time and download the chunks from all miners.
        # Then perform merge, upload and move on to the next partition index
        for chunk_id in partition_ids:
            logger.debug(f"Miner {self.hotkey} merging chunk {chunk_id} from {len(information_packets)} miners")
            weight_average = None
            optimizer_state_average = None
            weight_counter = 1
            optimizer_state_counter = 1
            weight_start_idx, weight_end_idx = None, None
            optimizer_state_start_idx, optimizer_state_end_idx = None, None

            # Loop over the paths and download the chunks from all miners.
            for information_packet in information_packets:
                weights_path = information_packet.weights_path
                weight_metadata_path = information_packet.weight_metadata_path
                optimizer_state_path = information_packet.optimizer_state_path
                optimizer_state_metadata_path = information_packet.optimizer_state_metadata_path
                weighting_factor = information_packet.weighting_factor

                try:
                    # This is the 1d sequence of bytes that we are going to merge with the other miners
                    weights, weight_metadata = await self._download_chunk(
                        data_path=weights_path,
                        metadata_path=weight_metadata_path,
                        chunk_id=chunk_id,
                        data_type="weights",
                    )
                    optimizer_state, optimizer_state_metadata = await self._download_chunk(
                        data_path=optimizer_state_path,
                        metadata_path=optimizer_state_metadata_path,
                        chunk_id=chunk_id,
                        data_type="optimizer_state",
                    )
                    assert (
                        weight_start_idx is None
                        or weight_start_idx == weight_metadata["sections"][str(chunk_id)]["start_idx"]
                    ), f"Weight start idx is not the same for all miners: {weight_start_idx} != {weight_metadata['sections'][str(chunk_id)]['start_idx']}"
                    assert (
                        weight_end_idx is None
                        or weight_end_idx == weight_metadata["sections"][str(chunk_id)]["end_idx"]
                    ), f"Weight end idx is not the same for all miners: {weight_end_idx} != {weight_metadata['sections'][str(chunk_id)]['end_idx']}"
                    assert (
                        optimizer_state_start_idx is None
                        or optimizer_state_start_idx == optimizer_state_metadata["sections"][str(chunk_id)]["start_idx"]
                    ), f"Optimizer state start idx is not the same for all miners: {optimizer_state_start_idx} != {optimizer_state_metadata['sections'][str(chunk_id)]['start_idx']}"
                    assert (
                        optimizer_state_end_idx is None
                        or optimizer_state_end_idx == optimizer_state_metadata["sections"][str(chunk_id)]["end_idx"]
                    ), f"Optimizer state end idx is not the same for all miners: {optimizer_state_end_idx} != {optimizer_state_metadata['sections'][str(chunk_id)]['end_idx']}"
                    weight_start_idx = weight_metadata["sections"][str(chunk_id)]["start_idx"]
                    weight_end_idx = weight_metadata["sections"][str(chunk_id)]["end_idx"]
                    optimizer_state_start_idx = optimizer_state_metadata["sections"][str(chunk_id)]["start_idx"]
                    optimizer_state_end_idx = optimizer_state_metadata["sections"][str(chunk_id)]["end_idx"]
                    assert (
                        weight_start_idx is not None and weight_end_idx is not None
                    ), "Weight missing start or end idx"
                    assert (
                        optimizer_state_start_idx is not None and optimizer_state_end_idx is not None
                    ), "Optimizer missing start or end idx"

                    if weight_average is None:
                        weight_average = weights.to(torch.float32) * weighting_factor
                        optimizer_state_average = optimizer_state.to(torch.float32) * weighting_factor

                    else:
                        # create a running sum of weights weighted by the weighting factor
                        weight_average += weights.to(torch.float32) * weighting_factor
                        optimizer_state_average += optimizer_state.to(torch.float32) * weighting_factor
                    weight_counter += weighting_factor
                    optimizer_state_counter += weighting_factor

                    # Clean up temporary weights
                    del weights
                    del optimizer_state

                except Exception as e:
                    logger.exception(
                        f"Error downloading chunk {chunk_id} from {weights_path} and {weight_metadata_path}: {e}"
                    )
                    time.sleep(60)
                    continue

            # Average the weights
            weight_average /= weight_counter
            weight_average = weight_average.to(torch.bfloat16)
            optimizer_state_average /= optimizer_state_counter
            optimizer_state_average = optimizer_state_average.to(torch.bfloat16)

            (
                weight_path,
                weight_metadata_path,
                optimizer_state_path,
                optimizer_state_metadata_path,
            ) = await self.upload_weights_with_metadata_and_optimizer_state(
                flattened_weights=weight_average,
                flattened_optimizer_state=optimizer_state_average,
                layer=self.layer,
                miner_hotkey=self.hotkey,
                num_sections=1,
                epoch=self.epoch,
            )

            weight_data = ChunkData(
                chunk_number=chunk_id,
                chunk_start_idx=weight_start_idx,
                chunk_end_idx=weight_end_idx,
                chunk_length=weight_end_idx - weight_start_idx,
            )
            logger.debug(f"Optimizer state average dtype: {optimizer_state_average.dtype}")
            optimizer_state_data = ChunkData(
                chunk_number=chunk_id,
                chunk_start_idx=optimizer_state_start_idx,
                chunk_end_idx=optimizer_state_end_idx,
                chunk_length=optimizer_state_end_idx - optimizer_state_start_idx,
                chunk_dtype=str(optimizer_state_average.dtype).split(".")[-1],
            )
            logger.debug(f"MINER UPLOADING MERGED PARTITIONS: {weight_data}")
            logger.debug(f"MINER UPLOADING MERGED PARTITIONS, REAL SIZE: {weight_average.shape}")
            logger.debug(f"MINER UPLOADING MERGED PARTITIONS: {optimizer_state_data}")
            logger.debug(f"MINER UPLOADING MERGED PARTITIONS, REAL SIZE: {optimizer_state_average.shape}")

            partitions.append(
                Partition(
                    layer=self.layer,
                    weight_data=weight_data,
                    optimizer_state_data=optimizer_state_data,
                    weight_path=weight_path,
                    weight_metadata_path=weight_metadata_path,
                    optimizer_state_path=optimizer_state_path,
                    optimizer_state_metadata_path=optimizer_state_metadata_path,
                    miner_hotkey=self.hotkey,
                    chunk_number=chunk_id,
                )
            )

            # Clean up averaged weights
            del weight_average
            del optimizer_state_average

        return partitions

    async def upload_activations(
        self,
        activation_uid: str,
        activations: torch.Tensor | list[torch.Tensor],
        direction: Literal["forward", "backward", "initial"],
    ):
        logger.info(
            f"📤 Uploading {direction} activation {activation_uid} | Layer: {self.layer} | Miner: {self.hotkey[:8]}"
        )

        # Ensure activations are detached and cloned before upload
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().clone()
        elif isinstance(activations, list):
            activations = [act.detach().clone() if isinstance(act, torch.Tensor) else act for act in activations]

        storage_path = await self.upload_activation(
            uid=activation_uid, layer=self.layer, direction=direction, data=activations
        )

        # await self.api_client.upload_activation_to_orchestrator(
        #     activation_uid=activation_uid,
        #     layer=self.layer,
        #     direction=direction,
        #     activation_path=storage_path,
        # )

        logger.debug(f"📤 Uploaded activation to path: {storage_path}")
        logger.debug(f"📤 Updating status with direction {direction}")
        await self.api_client.update_status(
            status=direction, activation_uid=activation_uid, activation_path=storage_path
        )

        # Clean up the activations tensor after upload
        del activations

        logger.info(f"✅ Successfully uploaded {direction} activation {activation_uid} | Path: {storage_path}")
        return storage_path

    async def forward(self, activation: ActivationResponse | None = None):
        """
        Performs the forward pass.

        If the layer is 0, it will load the data and upload the initial activation to the API.
        If the layer is not 0, it will download a random forward activation from the API and perform the forward pass.

        The forward pass contains:
        - Downloading the forward activation from the API
        - Performing the forward pass
        - Reporting the loss to the API
        - Performing the backward pass
        """
        if await self.out_of_cache:
            logger.warning(f"⚠️ Miner {self.hotkey[:8]} is out of cache, skipping forward pass")
            await asyncio.sleep(1)
            return

        if self.layer != 0:
            assert (
                activation.activation_uid is not None and activation.activation_path is not None
            ), f"Activation is required for layer {self.layer}, activation: {activation}"
        if self.layer == settings.N_LAYERS - 1:
            assert (
                activation.initial_activation is not None and activation.initial_activation_path is not None
            ), f"Initial activation is required for layer {self.layer}, activation: {activation}"

        if self.layer == 0:
            input_activations = await self._load_data()
            activation_uid = str(uuid.uuid4())

            logger.info(
                f"🚀 Starting FORWARD pass for layer {self.layer} | Generating initial activation {activation_uid} | Miner: {self.hotkey[:8]}"
            )
            await self.upload_activations(
                activation_uid=activation_uid,
                activations=input_activations.detach().clone(),
                direction="initial",
            )

        else:
            activation_uid = activation.activation_uid
            input_activation_path = activation.activation_path

            if not activation_uid or not input_activation_path:
                logger.warning(
                    f"⏸️ No forward activations found for layer {self.layer}, miner {self.hotkey[:8]} is idle"
                )
                return

            logger.info(
                f"🚀 Starting FORWARD pass for layer {self.layer} | Processing activation {activation_uid} | Miner: {self.hotkey[:8]}"
            )

            try:
                input_activations = download_activation(path=input_activation_path)
                logger.debug(f"📥 Downloaded activation from {input_activation_path}")
            except Exception as e:
                logger.error(f"❌ Error downloading activation: {e}")
                return

        output_activations, state = await self._forward(input_activations)

        self.processed_forward_activations.append(activation_uid)
        self.saved_forward_activations[activation_uid] = (input_activations, output_activations, state, time.time())
        self.forwards_since_reduce += 1

        if self.layer == settings.N_LAYERS - 1:
            initial_activations_path = activation.initial_activation_path
            if not initial_activations_path:
                logger.error(
                    f"❌ No input activation path found for layer {self.layer}, miner {self.hotkey[:8]} is idle. For activation {activation_uid} and layer path {initial_activations_path} was returned"
                )
                return
            try:
                initial_activations = download_activation(path=initial_activations_path)
                logger.debug(f"📥 Downloaded initial activation from {initial_activations_path}")
            except Exception as e:
                logger.error(f"❌ Error downloading initial activation: {e}")
                return

            output_activations = model_utils.compute_loss(
                logits=output_activations,
                targets=initial_activations,
                vocab_size=self.vocab_size,
                pad_token_id=self.eos_token_id,
                pack=settings.PACK_SAMPLES,
            )

            logger.info(
                f"📊 Computed loss {output_activations:.6f} for activation {activation_uid} | Layer: {self.layer} | Miner: {self.hotkey[:8]}"
            )
            try:
                await self.api_client.report_loss(activation_uid=activation_uid, loss=float(output_activations))
                logger.debug("📤 Reported loss to orchestrator")
            except Exception as e:
                logger.error(f"❌ Error reporting loss: {e}")

            # Update saved activations with loss (keep on CPU)
            self.saved_forward_activations[activation_uid] = (
                input_activations,
                output_activations,
                state,
                time.time(),
            )

            await self.api_client.update_status(status="forward", activation_uid=activation_uid, activation_path=None)
            logger.debug(f"📤 Updated status to 'forward' for activation {activation_uid}")

            try:
                await self.backward(activation=activation)

                # Log the loss and other training state information to wandb.
                # This is used for monitoring the loss during testing
                if settings.USE_WANDB:
                    metrics = {"loss": float(output_activations.item())}
                    await self._log_wandb(metrics)

            except Exception as e:
                logger.exception(f"❌ Error during backward step on last layer: {e}")

        else:
            await self.upload_activations(
                activation_uid=activation_uid,
                activations=output_activations.detach().clone(),
                direction="forward",
            )

        logger.info(
            f"✅ Successfully completed FORWARD pass for activation {activation_uid} on layer {self.layer} | Miner: {self.hotkey[:8]}"
        )

    async def backward(
        self,
        activation: ActivationResponse,
    ):
        logger.info(
            f"🔄 Starting BACKWARD pass for activation {activation.activation_uid} | Layer: {self.layer} | Miner: {self.hotkey[:8]}"
        )

        activation_grads = None
        if self.layer != settings.N_LAYERS - 1 and settings.N_LAYERS > 1:
            # For backward pass, we need to get activations that we have cached forward activations for
            # So we still need to list first, then filter, then randomly select
            activation_grads_path = activation.activation_path
            activation_grads = download_activation(path=activation_grads_path)

            activation_grads = activation_grads.to(settings.DEVICE)

            logger.debug(f"📥 Downloaded activation gradients from {activation_grads_path}")

        # Check if activation is in cache
        if activation.activation_uid not in self.saved_forward_activations:
            logger.warning(f"⚠️ Activation {activation.activation_uid} not found in cache, skipping backward pass")
            return

        # Get activations from cache and move back to GPU
        cached_activations = self.saved_forward_activations[activation.activation_uid]

        # Move to GPU and enable gradients only for floating point tensors
        input_activations = cached_activations[0].to(settings.DEVICE)
        output_activations = cached_activations[1].to(settings.DEVICE)

        state = cached_activations[2]

        self.backwards_since_reduce += 1
        self.backwards_since_sync += 1
        await self._backward(
            output_activations=output_activations,
            activation_grads=activation_grads,
            state=state,
        )

        # Remove from cache
        del self.saved_forward_activations[activation.activation_uid]
        self.backwards_since_reduce += 1
        self.backwards_since_sync += 1

        # Handle different cases for input activation gradients
        if settings.MOCK:
            input_activation_grads = input_activations.detach().to(torch.bfloat16).cpu()

        elif self.layer == 0:
            # Get the embedding layer weight grads instead of the input activations grads
            # This is because input activation grads of the first layer do not exist.
            emb_weight = self.model.tok_emb.weight
            input_activation_grads = emb_weight.grad[: settings.SEQUENCE_LENGTH]
            # Detach and convert to bfloat16 to ensure we only save the values
            input_activation_grads = input_activation_grads.detach().to(torch.bfloat16).cpu()

        else:
            input_activation_grads = input_activations.grad

        await self.upload_activations(
            activation_uid=activation.activation_uid,
            activations=input_activation_grads,
            direction="backward",
        )

        logger.info(
            f"✅ Successfully completed BACKWARD pass for activation {activation.activation_uid} | Layer: {self.layer} | Miner: {self.hotkey[:8]}"
        )

    async def upload_activation(self, uid: str, layer: int, direction: str, data: torch.Tensor) -> str:
        """Upload an activation to S3 storage using orchestrator-coordinated multipart upload."""
        assert isinstance(data, torch.Tensor), f"Activation is not a torch.Tensor: {type(data)}"

        # Generate the S3 path
        path = f"activations/{uid}/{layer}/{direction}/{uuid.uuid4()}.pt"

        # Save tensor to bytes
        buffer = io.BytesIO()
        torch.save(data, buffer)
        data_bytes = buffer.getvalue()

        # Use orchestrator-coordinated upload that automatically handles multipart for large files
        return await smart_upload_via_orchestrator_async(self.api_client, data_bytes, path)

    async def upload_weights(self, data: torch.Tensor, miner_hotkey: str, num_sections: int, epoch: int) -> str:
        """Upload weights to S3 storage using orchestrator-coordinated multipart upload."""
        # Generate the S3 path
        path = f"weights/{miner_hotkey}/{num_sections}/{epoch}/{uuid.uuid4()}.pt"

        # Convert tensor to bytes
        data = data.view(torch.uint8)
        data_bytes = data.cpu().detach().numpy().tobytes()

        # Use orchestrator-coordinated upload that automatically handles multipart for large files
        return await smart_upload_via_orchestrator_async(self.api_client, data_bytes, path)

    async def upload_metadata(
        self,
        metadata: dict[str, Any],
        miner_hotkey: str,
        num_sections: int,
        epoch: int,
        type: Literal["weights", "optimizer_state"],
    ) -> str:
        """Upload metadata to S3 storage."""
        presigned_data = await self.api_client.get_presigned_url(
            path=f"metadata/{type}/{miner_hotkey}/{num_sections}/{epoch}/{uuid.uuid4()}.json"
        )
        return upload_to_bucket(presigned_data, {"file": ("metadata.json", json.dumps(metadata))})

    async def upload_tensor(
        self,
        data: torch.Tensor,
        miner_hotkey: str,
        num_sections: int,
        epoch: int,
        type: Literal["weights", "optimizer_state"],
    ) -> str:
        """Upload weights to S3 storage."""

        path = f"{type}/{miner_hotkey}/{num_sections}/{epoch}/{uuid.uuid4()}.pt"

        # Convert to uint8
        # Convert tensor to bytes
        data = data.view(torch.uint8)
        data_bytes = data.cpu().detach().numpy().tobytes()

        # Use orchestrator-coordinated upload that automatically handles multipart for large files
        return await smart_upload_via_orchestrator_async(self.api_client, data_bytes, path)

    async def create_metadata(self, weights_numpy: torch.Tensor, num_sections: int) -> dict[str, Any]:
        # Create metadata about the tensor
        tensor_metadata = {
            "dtype": str(weights_numpy.dtype),
            "size": weights_numpy.size(),
            "num_elements": weights_numpy.numel(),
            "element_size": weights_numpy.itemsize,
            "total_bytes": weights_numpy.nbytes,  # this is just weights_numpy.numel() * weights_numpy.itemsize
        }

        # Number of sections to split into (in elements, or indices)
        section_size = weights_numpy.numel() // num_sections
        # Create section metadata
        sections_metadata = {}

        for i in range(num_sections):
            start_idx = i * section_size
            end_idx = start_idx + section_size if i < num_sections - 1 else weights_numpy.numel()

            # Calculate corresponding tensor indices
            start_byte = start_idx * tensor_metadata["element_size"]
            end_byte = end_idx * tensor_metadata["element_size"]

            assert start_byte is not None and end_byte is not None, "Start byte and end byte are missing"
            assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"
            sections_metadata[i] = {
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_idx": start_idx,  # e.g for a (100,100) matrix divided into 10 sections, the indices are: 0-999. 1000 - 1999. 2000 - 2999. 3000 - 3999. 4000 - 4999. 5000 - 5999. 6000 - 6999. 7000 - 7999. 8000 - 8999. 9000 - 9999.
                "end_idx": end_idx,
            }

        # Save full tensor metadata
        full_metadata = {"tensor": tensor_metadata, "sections": sections_metadata}

        return full_metadata

    async def upload_tensor_with_metadata(
        self,
        tensor: torch.Tensor,
        miner_hotkey: str,
        num_sections: int,
        epoch: int,
        type: Literal["weights", "optimizer_state"],
    ) -> tuple[str, str]:
        # tensor = tensor.to(torch.float16)
        # tensor_numpy = tensor.detach().cpu().numpy(force=True)
        metadata = await self.create_metadata(tensor, num_sections)
        weight_path = await self.upload_tensor(tensor, miner_hotkey, num_sections, epoch, type)
        metadata_path = await self.upload_metadata(metadata, miner_hotkey, num_sections, epoch, type)
        return weight_path, metadata_path

    async def upload_weights_with_metadata_and_optimizer_state(
        self,
        flattened_weights: torch.Tensor,
        flattened_optimizer_state: torch.Tensor,
        layer: int,
        miner_hotkey: str,
        num_sections: int,
        epoch: int,
    ) -> tuple[str, str, str]:
        # Convert to NumPy array and save in raw binary format
        weight_path, weight_metadata_path = await self.upload_tensor_with_metadata(
            flattened_weights, miner_hotkey, num_sections, epoch, "weights"
        )
        optimizer_state_path, optimizer_state_metadata_path = await self.upload_tensor_with_metadata(
            flattened_optimizer_state,
            miner_hotkey,
            num_sections,
            epoch,
            "optimizer_state",
        )
        return (
            weight_path,
            weight_metadata_path,
            optimizer_state_path,
            optimizer_state_metadata_path,
        )

    async def start(self) -> asyncio.Task:
        return asyncio.create_task(self.run())

    async def _log_wandb(self, metrics: dict):
        """
        Logs the metrics along with other training state information to wandb
        """
        # Log loss to wandb
        if not self.wandb_initialized:
            if settings.WANDB_TOKEN:
                wandb.login(key=settings.WANDB_TOKEN)

            wandb.init(
                project=settings.WANDB_PROJECT,
                entity=settings.WANDB_ENTITY,
                name=f"{settings.RUN_NAME}",
                config={
                    # Model configuration
                    "model_name": settings.MODEL_CFG["model_name"],
                    "n_layers": settings.N_LAYERS,
                    "miners_per_layer": settings.MINERS_PER_LAYER,
                    "model_splits": settings.MODEL_SPLITS,
                    # Training hyperparameters
                    "batch_size": settings.BATCH_SIZE,
                    "sequence_length": settings.SEQUENCE_LENGTH,
                    "total_train_steps": settings.TOTAL_TRAIN_STEPS,
                    "weight_decay": settings.WEIGHT_DECAY,
                    "pack_samples": settings.PACK_SAMPLES,
                    "learning_rate": settings.LEARNING_RATE,
                    "lr_warmup_start_factor": settings.LR_WARMUP_START_FACTOR,
                    "lr_warmup_steps": settings.LR_WARMUP_STEPS,
                    "lr_const_steps": settings.LR_CONST_STEPS,
                    "lr_tail_steps_frac": settings.LR_TAIL_STEPS_FRAC,
                    "lr_final_factor": settings.LR_FINAL_FACTOR,
                    "lr_saw_cycle_length": settings.LR_SAW_CYCLE_LENGTH,
                    # Dataset
                    "dataset_name": settings.DATASET_NAME,
                    # Model merging configuration
                    "local_optimizer_steps": settings.LOCAL_OPTIMIZER_STEPS,
                    "global_optimizer_steps": settings.GLOBAL_OPTIMIZER_STEPS,
                    "miners_required_for_merging": settings.MINERS_REQUIRED_FOR_MERGING,
                    # Runtime configuration
                    "device": str(settings.DEVICE),
                    "mock_mode": settings.MOCK,
                    "completed_optim_steps": self.completed_optim_steps,
                },
            )
            self.wandb_initialized = True

        # load globa gradient norm
        total_grad_norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in self.model.parameters() if p.grad is not None)
        )

        # Extract loss from the metrics dictionary
        loss = metrics["loss"]

        wandb.log(
            {
                "loss": float(loss),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "grads/global_norm": float(total_grad_norm.item()),
            }
        )
