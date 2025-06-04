import asyncio
import copy
import csv
import io
import json
import random
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import bittensor as bt
import numpy as np
from bittensor_wallet.mock import get_mock_wallet
from loguru import logger
from pydantic import BaseModel, Field, model_validator

import settings
from orchestrator import ORCHESTRATOR_NON_SERIALIZABLE_FIELDS
from orchestrator.dashboard_metrics import DashboardMetricsReporter
from orchestrator.metrics_collectors import (
    ActivationMetricsCollector,
    TimeSeriesMetricsCollector,
    WeightMergingMetricsCollector,
)
from orchestrator.miner_registry import MinerData, MinerRegistry
from orchestrator.mongo_state import MongoStateManager
from orchestrator.serializers import LossReport, SubmittedWeights
from orchestrator.validator_client_pool import ValidatorClientPool
from storage.activation_storage import ActivationStore
from storage.serializers import ActivationResponse
from storage.weight_storage import WeightStore
from utils.bt_utils import subtensor
from utils.metagraph_syncer import MetagraphSyncer
from utils.partitions import Partition, PartitionManager
from utils.s3_interactions import generate_presigned_url, upload_to_bucket
from utils.shared_states import MergingPhase, MergingPhaseManager

CHAIN_SCORES_LOCATIONS = "scores/miner_scores.json"


class Orchestrator(BaseModel):
    """Main orchestrator class that manages miners, validators, and model training coordination.

    This class handles:
    - Miner registration and tracking
    - Gradient validation
    - Model weight management
    - Activation storage
    - Metagraph synchronization
    - Training coordination
    """

    activation_store: ActivationStore
    weight_store: WeightStore
    validator_pool: ValidatorClientPool
    partition_manager: PartitionManager = Field(default_factory=PartitionManager)
    dashboard_reporter: Optional[DashboardMetricsReporter] = Field(default=None)
    activation_metrics_collector: ActivationMetricsCollector = Field(default_factory=ActivationMetricsCollector)
    weight_merging_metrics_collector: WeightMergingMetricsCollector = Field(
        default_factory=WeightMergingMetricsCollector
    )
    time_series_collector: TimeSeriesMetricsCollector = Field(default_factory=TimeSeriesMetricsCollector)
    miner_scores: dict[int, list[float]] = Field(default_factory=dict)
    losses: dict[int, list[LossReport]] = Field(default_factory=lambda: defaultdict(list))
    TIMEOUT: int = settings.TIMEOUT
    total_forwards: int = 0
    total_backwards: int = 0
    total_completed: int = 0
    tracked_activations: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: defaultdict(list))
    N_LAYERS: int = 3
    miners_with_submitted_scores: Dict[int, Dict[str, tuple[str, str, str, str]]] = Field(
        default_factory=lambda: defaultdict(dict)
    )
    merging_phases: list[MergingPhaseManager] = Field(default_factory=list)
    validator_init_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    global_miner_scores: Dict[int, List[Tuple[int, float]]] = Field(default_factory=lambda: defaultdict(list))
    miner_registry: MinerRegistry = Field(default_factory=lambda: MinerRegistry(miner_hotkeys=[]))

    # Metagraph syncing attributes
    metagraph_syncer: Optional[Any] = None  # MetagraphSyncer instance
    metagraph: Optional[bt.metagraph] = None
    netuid: Optional[int] = None
    config: Optional[object] = None
    subtensor: Optional[bt.subtensor] = None
    lock: threading.RLock = Field(default_factory=threading.RLock)
    wallet_name: Optional[str] = None
    wallet_hotkey: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _post_orchestrator_setup(self):
        """Post call to initialize the orchestrator."""

        for _ in range(settings.N_LAYERS):
            self.merging_phases.append(MergingPhaseManager())

        if settings.BITTENSOR:
            if self.wallet_name and self.wallet_hotkey:
                wallet = bt.wallet(name=self.wallet_name, hotkey=self.wallet_hotkey)
            else:
                logger.error("No wallet name or hotkey provided")
                raise ValueError("No wallet name or hotkey provided")
        else:
            wallet = get_mock_wallet()
        self.validator_pool.wallet = wallet
        return self

    def _initialize_miner_registry(self):
        """Initialize the miner registry using metagraph data."""
        # Create empty registry - actual state will be loaded by state manager
        self.miner_registry = MinerRegistry(miner_hotkeys=[])
        logger.info("Initialized empty miner registry")

    def setup_metagraph_sync(
        self,
        metagraph_syncer: MetagraphSyncer,
        netuid: int,
        config=None,
        subtensor: bt.subtensor = None,
    ):
        """Setup metagraph syncing for the orchestrator.

        Args:
            metagraph_syncer: Instance of MetagraphSyncer for syncing the metagraph
            netuid: The network UID to connect to
            config: Configuration object
            subtensor: Bittensor subtensor instance
        """
        # Network setup
        self.netuid = netuid
        self.config = config
        self.subtensor = subtensor

        # Metagraph syncing
        self.metagraph_syncer = metagraph_syncer
        if self.metagraph_syncer and self.netuid is not None:
            # Get initial metagraph
            self.metagraph = self.metagraph_syncer.get_metagraph(self.netuid)
            # Register listener for metagraph updates
            self.metagraph_syncer.register_listener(self._on_metagraph_updated, netuids=[self.netuid])
            logger.info(f"Orchestrator initialized with metagraph syncing for netuid {self.netuid}")

    def _on_metagraph_updated(self, metagraph: bt.metagraph, netuid: int):
        """Handle metagraph updates.

        Args:
            metagraph: The updated metagraph
            netuid: The network UID of the updated metagraph

        This method:
        1. Updates the local metagraph copy
        2. Updates the miner registry to reflect changes in the metagraph
        3. Handles deregistration of miners no longer in the metagraph
        4. Registers new miners found in the metagraph
        5. Updates the dashboard reporter callback
        """
        if settings.BITTENSOR:
            with self.lock:
                if netuid == self.netuid:
                    self.metagraph = copy.deepcopy(metagraph)
                    logger.debug(f"Orchestrator metagraph updated for netuid {netuid}")

                    # Determine the hotkeys that are no longer in the metagraph
                    old_hotkeys = set(self.miner_registry.get_all_miner_data().keys())
                    new_hotkeys = set([self.metagraph.hotkeys[uid] for uid in self.metagraph.uids])

                    deregistered_hotkeys = old_hotkeys.difference(new_hotkeys)

                    # Remove miners that are no longer in the metagraph
                    for hk in deregistered_hotkeys:
                        logger.warning(f"Tracked miner {hk[:8]}... no longer exists in metagraph")
                        self.miner_registry.remove_miner_from_registry(miner_hotkey=hk)
                        self.global_miner_scores.pop(self.miner_registry.get_miner_data(miner_hotkey=hk).uid, None)

                    self._update_dashboard_callback()

    async def register(self, hotkey: str) -> int:
        """Register a miner with the orchestrator.

        Args:
            hotkey: The hotkey of the miner to register

        Returns:
            int: The layer assigned to the miner

        This method:
        1. Validates the miner exists in the metagraph (if BITTENSOR mode)
        2. Checks if the miner is already registered
        3. Assigns a layer to the miner
        4. Adds the miner to the registry

        Return the layer assigned to the miner.
        """
        logger.debug(f"Attempting to register miner {hotkey[:8]}...")

        # Handle BITTENSOR mode
        if settings.BITTENSOR:
            if not self.metagraph:
                logger.error("Cannot register miner - metagraph not available")
                return

            if hotkey not in self.metagraph.hotkeys:
                logger.warning(f"Miner {hotkey[:8]} not found in metagraph")
                return

            # Get the uid of the miner
            uid = self.metagraph.uids[self.metagraph.hotkeys.index(hotkey)]
        else:
            uid = np.random.randint(1, 255)

        # Check if miner is already registered
        if hotkey in self.miner_registry.get_all_miner_data().keys():
            logger.info(f"Miner {hotkey[:8]} already registered, returning already assigned layer")
            return self.miner_registry.get_miner_data(miner_hotkey=hotkey).layer

        try:
            # Assign a layer to the miner during registration
            layer = await self.request_layer()
            self.miner_registry.add_miner_to_registry(miner_hotkey=hotkey, layer=layer, uid=uid)

            logger.info(f"Successfully registered miner {hotkey[:8]} (Layer: {layer})")

        except Exception as e:
            logger.exception(f"Failed to register miner {hotkey[:8]}: {str(e)}")
            return None

        return layer

    async def register_validator(self, hotkey: str, host: str, port: int, scheme: str = "http") -> bool:
        """Register a validator with the orchestrator.

        Args:
            hotkey: The validator's hotkey
            host: The validator's host address
            port: The validator's port number
            scheme: The connection scheme (http/https)

        Returns:
            bool: True if registration was successful, False otherwise
        """

        # Add validator to the pool
        success = await self.validator_pool.add_validator(host=host, port=port, scheme=scheme, hotkey=hotkey)

        if not success:
            logger.error(f"Failed to add validator {hotkey[:8]} to pool")
            return False

        return True

    async def initialize_gradient_validator(self, weight_paths: list[str] | None = None):
        """Initialize gradient validators for tracking miners.

        Args:
            weight_paths: Optional list of weight paths for each layer

        This method:
        1. Initializes the validator pool if needed
        2. Selects random miners to track
        3. Assigns validators to track selected miners
        """
        if not settings.VALIDATE:
            logger.warning("Gradient validation is disabled")
            return

        num_miners = len(self.miner_registry.get_all_miner_data())
        if num_miners == 0:
            logger.warning("No miners available for gradient validation")
            return

        validators_to_initialize = min(self.validator_pool.validator_count, num_miners)
        logger.warning(f"Initializing {validators_to_initialize} gradient validators")

        miners_to_track = random.sample(self.miner_registry.get_all_miner_data().keys(), validators_to_initialize)
        logger.warning(f"Tracking miners: {miners_to_track}")

        for miner_hotkey in miners_to_track:
            # Pick a random miner to track
            layer = self.miner_registry.get_miner_data(miner_hotkey).layer

            # Assign the miner to a validator
            success, validator_hotkey = await self.validator_pool.assign_miner_to_validator(
                miner_hotkey=miner_hotkey,
                layer=layer,
                weight_path=(weight_paths[layer] if weight_paths is not None else None),
            )

            if success:
                logger.debug(f"GRADIENT VALIDATOR INITIALIZED TO TRACK MINER {miner_hotkey}")
                self.miner_registry.set_miner_attribute(
                    miner_hotkey=miner_hotkey,
                    attribute="responsible_validator",
                    value=validator_hotkey,
                )
            else:
                logger.warning(f"FAILED TO INITIALIZE GRADIENT VALIDATOR FOR MINER {miner_hotkey}")

    async def update_status(
        self,
        hotkey: str,
        status: Literal["forward", "backward", "idle"],
        activation_uid: str,
    ) -> None:
        """Update the status of a miner.

        Args:
            hotkey: The hotkey of the miner to update
            status: The status to update the miner to ("forward", "backward", or "idle")
            activation_uid: Activation uid associated to the miner's activation.

        This method:
        1. Updates the miner's status in the registry
        2. Tracks statistics (forwards, backwards, completed)
        3. Validates activations if needed
        4. Checks for merging phase conditions
        """
        if not self.activation_store.does_activation_exist(activation_uid):
            logger.warning(f"Activation {activation_uid} does not exist in activation store")
            return
        # Try to pick a miner for validation if this is the first run
        if self.validator_pool.get_available_validators():
            async with self.validator_init_lock:
                if self.validator_pool.get_available_validators():  # Double check after acquiring lock
                    await self.initialize_gradient_validator(weight_paths=None)
                    logger.debug("GRADIENT VALIDATOR INITIALIZED")

        self.miner_registry.set_miner_attribute(miner_hotkey=hotkey, attribute="status", value=status)
        logger.debug(f"Miner {hotkey[:8]} is {status}")

        miner_data: MinerData = self.miner_registry.get_miner_data(miner_hotkey=hotkey)

        if status == "forward":
            # Check to see if the cache is full for the miner's request to avoid overflow
            if miner_data.out_of_cache:
                logger.warning(f"Miner {hotkey[:8]} out of cache... Skipping update_status")
                return

            self.total_forwards += 1
            processed_activations = miner_data.processed_activations + 1
            self.miner_registry.set_miner_attribute(
                miner_hotkey=hotkey,
                attribute="processed_activations",
                value=processed_activations,
            )

            try:
                self.miner_registry.add_to_miner_cache(miner_hotkey=hotkey, activation_uid=activation_uid)
                logger.debug(
                    f"Added activation {activation_uid} to miner {hotkey[:8]} in layer {miner_data.layer} cache after forward pass"
                )
            except Exception as e:
                logger.warning(f"Failed to add activation uid {activation_uid} to cache for miner {hotkey[:8]}: {e}")

        # Check to see if the activation is actually being tracked by this miner. Redundancy.
        scores_to_submit = {}
        if status == "backward" and self.miner_registry.is_activation_cached_by_miner(
            miner_hotkey=hotkey, activation_uid=activation_uid
        ):
            self.total_backwards += 1
            backwards_since_reset = miner_data.backwards_since_reset + 1
            self.miner_registry.set_miner_attribute(
                miner_hotkey=hotkey,
                attribute="backwards_since_reset",
                value=backwards_since_reset,
            )

            scores_to_submit[self.miner_registry.get_miner_data(miner_hotkey=hotkey).uid] = 1

            # Track cache removal: when a miner completes backward pass, remove activation from cache
            try:
                self.miner_registry.remove_from_miner_cache(miner_hotkey=hotkey, activation_uid=activation_uid)
                logger.debug(f"Removed activation {activation_uid} from miner {hotkey[:8]} cache after backward pass")
            except Exception as e:
                logger.warning(f"Failed to remove activation from cache for miner {hotkey[:8]}: {e}")

        if scores_to_submit:
            await self.submit_miner_scores(scores=scores_to_submit)

        # Validating activations
        validation_success = True

        if activation_uid is not None:
            # Track this activation first
            if activation_uid not in self.tracked_activations:
                self.tracked_activations[activation_uid] = []
            self.tracked_activations[activation_uid].append({"hotkey": hotkey, "timestamp": time.time()})

            # Only validate if a validator is available and tracking this miner
            # First check if any validator is tracking this miner
            if hotkey in await self.validator_pool.get_tracked_miners():
                # validator_hotkey = self.miner_registry.get_miner_data(miner_hotkey=hotkey).responsible_validator
                # Now validate with the validator tracking this miner
                validation_result = await self.validator_pool.validate_activation(
                    activation_uid=activation_uid,
                    direction=status,
                    miner_hotkey=hotkey,
                )

                validation_success = validation_result.get("is_valid", True)

            if not validation_success:
                logger.error(
                    f"Miner {hotkey[:8]} has invalid activations: {validation_result.get('reason')}, score: {validation_result.get('score')}, direction: {status}, activation uid: {activation_uid}"
                )
                # Record failed status update
                self.activation_metrics_collector.record_status_updated(
                    activation_uid,
                    success=False,
                    error_message=f"Validation failed: {validation_result.get('reason')}",
                )
            else:
                # Record successful status update (only timing we track)
                self.activation_metrics_collector.record_status_updated(activation_uid, success=validation_success)

        for layer in range(settings.N_LAYERS):
            counts = "\n".join(
                [
                    f"Miner {m.hotkey}: {m.backwards_since_reset} Backwards"
                    for m in self.miner_registry.get_miners_in_layer(layer)
                ]
            )
            logger.debug(f"\n\nLAYER {layer} MERGE COUNTS:\n{counts}\n")

        # Check if all miners are in the IS_TRAINING stage
        if not all([m.stage == MergingPhase.IS_TRAINING for m in self.merging_phases]):
            for layer, m in enumerate(self.merging_phases):
                logger.debug(f"Phase {m.stage} for layer {layer}")
            logger.debug(
                "--------------------- Not all miners are in the IS_TRAINING stage. Skipping weight upload. ---------------------"
            )
            return

        if await self._can_upload_weights():
            await self._start_weight_upload()

        # Update queue and system health metrics periodically
        if self.total_backwards % 10 == 0:  # Update every 10 status updates
            await self._update_system_metrics()

        # Cleanup stale weight merging sessions periodically
        if self.total_backwards % 100 == 0:  # Cleanup every 100 status updates
            cleaned_sessions = self.weight_merging_metrics_collector.cleanup_stale_sessions()
            if cleaned_sessions > 0:
                logger.warning(f"Cleaned up {cleaned_sessions} stale weight merge sessions")

        if settings.ENABLE_DASHBOARD_REPORTING:
            # Update dashboard
            await self.update_dashboard(miner_hotkey=hotkey)

        # Create the MongoDB state manager and save the current state after updating.
        if self.total_backwards % settings.UPLOAD_EVERY_N_UPDATES == 0:
            await self.save_orchestrator_state_to_db()

    async def _can_upload_weights(self) -> bool:
        """Check if we can merge weights for all layers.

        This method checks if the number of miners that have completed the backward pass is greater than or
        equal to the number of miners required for merging. If so, merging is needed.
        """
        miners_to_merge = [
            m
            for m in self.miner_registry.get_all_miner_data().values()
            if m.backwards_since_reset >= settings.GLOBAL_OPTIMIZER_STEPS
        ]
        logger.debug(
            f"{len(miners_to_merge)}/{len(self.miner_registry.get_all_miner_data())} miners to merge: {[m.hotkey[:8] for m in miners_to_merge]}"
        )

        # Check if we need to enter merging phase
        if len(miners_to_merge) >= settings.MINERS_REQUIRED_FOR_WEIGHT_UPLOADING * len(
            self.miner_registry.get_all_miner_data()
        ):
            logger.debug("MINERS READY FOR MERGING. SETTING STATUS TO WEIGHTS_UPLOADING")

            # Progress each section of the model's phase to the next phase (e.g. from NOT_MERGING to WEIGHTS_UPLOADING).
            for phase in self.merging_phases:
                await phase.advance_phase(timeout=settings.PHASE_TIMEOUT, expected_phase=MergingPhase.IS_TRAINING)
            return True
        return False

    async def _start_weight_upload(self):
        # Start merge sessions for each layer
        for layer in range(settings.N_LAYERS):
            layer_miners = [m.hotkey for m in self.miner_registry.get_miners_in_layer(layer)]
            if layer_miners:  # Only start session if there are miners in this layer
                session_id = self.weight_merging_metrics_collector.start_merge_session(
                    layer=layer, target_miners=layer_miners
                )
                logger.debug(f"Started weight merge session {session_id} for layer {layer} with miners: {layer_miners}")

                # Mark all miners in this layer as starting weight upload
                for miner_hotkey in layer_miners:
                    self.miner_registry.start_miner_weight_upload(miner_hotkey=miner_hotkey, session_id=session_id)

        logger.success("--------------------- All layers have started weight upload. ---------------------")

    async def save_orchestrator_state_to_db(self, on_weights_merged: bool = False):
        if settings.MONGO:
            logger.debug("MongoDB FROM save_orchestrator_state_to_db")
            state_manager: MongoStateManager = await MongoStateManager.create()
            await state_manager.save_state(orchestrator=self, on_weights_merged=on_weights_merged)

            # Cleanup stale activation metrics periodically
            await self.cleanup_stale_activation_metrics()
        else:
            logger.warning("MongoDB is disabled, skipping state save")

    async def _ensure_dashboard_initialized(self):
        """Ensure the dashboard reporter is initialized before any reporting."""
        if settings.ENABLE_DASHBOARD_REPORTING and not self.dashboard_reporter:
            self.dashboard_reporter = DashboardMetricsReporter()
            await self.dashboard_reporter.initialize()

    async def update_dashboard(self, miner_hotkey: str):
        """Update the dashboard with the miner's status.

        Args:
            miner_hotkey (str): The hotkey of the miner to update
        """

        await self._ensure_dashboard_initialized()

        # Catch possible race condition
        if not self.dashboard_reporter:
            if settings.DASHBOARD_LOGS:
                logger.warning("Dashboard reporter not initialized yet")
            return

        miner: MinerData = self.miner_registry.get_miner_data(miner_hotkey=miner_hotkey)

        # Report miner status to dashboard
        throughput = 0
        if miner.layer is not None and self.dashboard_reporter:
            current_time = time.time()
            time_since_last_report = current_time - miner.last_throughput_report
            throughput = (
                miner.processed_activations / (time_since_last_report / 60) if time_since_last_report > 0 else 0
            )  # activations per minute

        # Gather metagraph info if available
        coldkey = incentive = None
        if self.metagraph and settings.BITTENSOR:
            uid = self.metagraph.hotkeys.index(miner_hotkey)
            coldkey = self.metagraph.coldkeys[uid]
            incentive = float(self.metagraph.incentive[uid])
        else:
            uid = abs(hash(miner_hotkey)) % (10**8)
            coldkey = f"dummy-coldkey-{miner_hotkey}"
            incentive = 0.0

        await self.dashboard_reporter.report_miner_status(
            miner_uid=uid,
            layer=miner.layer,
            processed_activations=miner.processed_activations,
            throughput=throughput,
            registration_time=miner.registration_time,
            coldkey=coldkey,
            hotkey=miner_hotkey,
            incentive=incentive,
        )

        if settings.DASHBOARD_LOGS:
            logger.debug(
                f"Sending miner status to dashboard: miner_uid={uid}, layer={miner.layer}, processed_activations={miner.processed_activations}, throughput={throughput:.2f} activations/min"
            )

        # Reset counters after reporting
        self.miner_registry.set_miner_attribute(miner_hotkey=miner_hotkey, attribute="processed_activations", value=0)
        self.miner_registry.set_miner_attribute(
            miner_hotkey=miner_hotkey,
            attribute="last_throughput_report",
            value=current_time,
        )

    async def get_partition_count(self, layer: int):
        """Get the number of partitions for a given layer."""
        n_miners = len(self.miner_registry.get_miners_in_layer(layer))
        n_partitions = max(1, n_miners * (n_miners - 1) / 2)
        return n_partitions

    async def is_merging(self, layer: int):
        """Check if the orchestrator is in merging phase.

        Returns:
            bool: True if in merging phase, False otherwise
        """
        return self.merging_phases[layer].stage, await self.get_partition_count(layer)

    async def record_and_report_loss(self, hotkey: str, activation_uid: str, loss: float):
        await self._ensure_dashboard_initialized()
        """
        Record and report a loss value from a miner.
        This method handles:
        1. Internal tracking of losses
        2. CSV file logging for historical data
        3. Dashboard metrics reporting (buffered)

        Args:
            miner_uid: The ID of the miner reporting the loss
            activation_uid: The activation ID associated with this loss
            loss: The loss value
        """
        # Create loss report for internal tracking
        loss_report = LossReport(
            hotkey=hotkey,
            activation_uid=activation_uid,
            loss_value=loss,
            timestamp=time.time(),
        )
        self.losses[hotkey].append(loss_report)
        if settings.DASHBOARD_LOGS:
            logger.info(f"Miner {hotkey} reported loss {loss} for activation {activation_uid}")

        # Log to CSV file for historical data
        current_time = time.strftime("%Y%m%d_%H%M%S")
        loss_dir = Path(settings.LOSSES_DIR)
        loss_dir.mkdir(exist_ok=True)

        recent_file = None
        for f in loss_dir.glob("losses_*.csv"):
            if time.time() - f.stat().st_mtime < settings.LOSS_REPORT_INTERVAL:
                recent_file = f
                break

        if recent_file:
            filename = recent_file
        else:
            filename = loss_dir / f"losses_{current_time}.csv"

        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if filename.stat().st_size == 0:
                writer.writerow(["miner_uid", "activation_uid", "loss_value", "timestamp"])
            writer.writerow(
                [
                    loss_report.hotkey,
                    loss_report.activation_uid,
                    loss_report.loss_value,
                    loss_report.timestamp,
                ]
            )

        # Buffer the loss for dashboard reporting (non-blocking)
        if self.dashboard_reporter:
            if settings.DASHBOARD_LOGS:
                logger.info(f"Sending loss to dashboard: loss={loss}")
            asyncio.create_task(
                self.dashboard_reporter.buffer_loss(
                    loss=loss,
                    number_of_sampled_activations=1,  # Each report is for one activation
                )
            )

    async def request_layer(self, layer: int | None = None):
        """Request a layer assignment for a miner.

        Args:
            layer: Optional specific layer to request

        Returns:
            int: The assigned layer number

        This method:
        1. Determines the layer with the least miners
        2. Assigns the miner to that layer
        3. Updates the miner's layer in the registry
        """

        # Get count of miners per layer
        layer_counts = {l: 0 for l in range(settings.N_LAYERS)}
        incentive_per_layer = {l: 0 for l in range(settings.N_LAYERS)}

        for miner in self.miner_registry.get_all_miner_data().values():
            if miner.layer is not None:
                if settings.BITTENSOR and self.metagraph:
                    miner_uid = self.metagraph.hotkeys.index(miner.hotkey)
                    logger.warning(f"Miner layer {miner.layer}, increasing layer count")
                    layer_counts[miner.layer] += 1
                    incentive_per_layer[miner.layer] += self.metagraph.I[miner_uid]
                else:
                    layer_counts[miner.layer] += 1

        mean_incentive_per_layer = {
            l: incentive_per_layer[l] / layer_counts[l] if layer_counts[l] > 0 else 0 for l in range(settings.N_LAYERS)
        }

        logger.info(f"Layer counts: {layer_counts}")
        logger.info(f"Mean incentive per layer: {mean_incentive_per_layer}")

        # If a specific layer is requested, return that layer
        if layer is not None:
            chosen_layer = layer

        # Get the layer with the least miners
        elif any(layer_counts[l] != layer_counts[0] for l in range(settings.N_LAYERS)):
            min_layer = min(layer_counts, key=layer_counts.get)
            chosen_layer = min_layer

        # If the number of miners in each layer is the same, add to the layer with the least mean incentive
        else:
            min_incentive_layer = min(mean_incentive_per_layer, key=mean_incentive_per_layer.get)
            chosen_layer = min_incentive_layer

        logger.info(f"Chosen layer: {chosen_layer}")
        return chosen_layer

    async def store_stats(
        self,
        forward_miners: list[MinerData],
        backward_miners: list[MinerData],
        idle_miners: list[MinerData],
        stored_forward_activations: list[int],
        stored_backward_activations: list[int],
    ):
        """Store statistics about miners and activations.

        Args:
            forward_miners: List of miners in forward state
            backward_miners: List of miners in backward state
            idle_miners: List of miners in idle state
            stored_forward_activations: List of stored forward activations per layer
            stored_backward_activations: List of stored backward activations per layer
        """
        # TODO: Make this less ugly
        stats = await self.activation_store.get_activations_stats()

        # Get current time for filename
        current_time = time.strftime("%Y%m%d_%H%M%S")

        # Check for recent files
        stats_dir = Path("activation_stats")
        stats_dir.mkdir(exist_ok=True)

        recent_file = None
        for f in stats_dir.glob("activation_stats_*.csv"):
            if time.time() - f.stat().st_mtime < 10:
                recent_file = f
                break

        if recent_file:
            filename = recent_file
        else:
            filename = stats_dir / f"activation_stats_{current_time}.csv"

        # Write activation stats to CSV
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if filename.stat().st_size == 0:
                writer.writerow(
                    ["timestamp", "forward_miners", "backward_miners", "idle_miners"]
                    + [f"forward_activations_layer_{i}" for i in range(settings.N_LAYERS)]
                    + [f"backward_activations_layer_{i}" for i in range(settings.N_LAYERS)]
                )
            writer.writerow(
                [
                    time.time(),
                    len(forward_miners),
                    len(backward_miners),
                    len(idle_miners),
                ]
                + stored_forward_activations
                + stored_backward_activations
            )

    async def save_tracked_activations(self, path: str | None = None):
        """Save tracked activations to a file.

        Args:
            path: Optional path to save the activations to. Defaults to "tracked_activations.json"
        """
        if path is None:
            path = "tracked_activations.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tracked_activations, f)

    async def notify_weights_uploaded(
        self,
        hotkey: str,
        weights_path: str,
        metadata_path: str,
        optimizer_state_path: str,
        optimizer_state_metadata_path: str,
    ):
        """Notify that a miner has uploaded weights.

        Args:
            hotkey: The hotkey of the miner that uploaded weights
            weights_path: The path of the uploaded weights
            metadata_path: The path of the uploaded metadata

        This method:
        1. Records the weight location
        2. Triggers model merging if enqough miners have submitted weights
        3. Updates the miner's last S3 upload time
        """
        logger.info(f"Notifying weights uploaded for miner {hotkey}")
        try:
            # Update last S3 upload time
            self.miner_registry.update_s3_upload_time(miner_hotkey=hotkey)
            miner = self.miner_registry.get_miner_data(hotkey)

            layer = miner.layer
            if layer is None:
                logger.error(f"Miner {hotkey} has no layer, not accepting weights")
                return "Miner missing layer"

            if miner.backwards_since_reset < settings.GLOBAL_OPTIMIZER_STEPS:
                logger.error(
                    f"Miner {hotkey} has not completed enough backwards steps: completed: {miner.backwards_since_reset} required: {settings.GLOBAL_OPTIMIZER_STEPS}, not accepting weights."
                )
                backwards_since_reset = {
                    m.hotkey: m.backwards_since_reset for m in self.miner_registry.get_all_miner_data().values()
                }
                logger.error(f"Other miners have completed: {backwards_since_reset}.")
                return "Miner has not done enough backwards steps"

            if self.merging_phases[layer].stage == MergingPhase.MINERS_MERGING_PARTITIONS:
                logger.error(f"Already in merging phase, not accepting weights from miner {hotkey}")
                return "Already in merging phase, miner can't submit weights anymore"

            # Record weight upload in metrics and update miner status
            self.weight_merging_metrics_collector.record_weight_upload(layer, hotkey)

            await self.weight_store.upload_weights_and_optimizer(
                miner_hotkey=hotkey,
                weights_path=weights_path,
                weight_metadata_path=metadata_path,
                optimizer_state_path=optimizer_state_path,
                optimizer_state_metadata_path=optimizer_state_metadata_path,
            )
            self.miner_registry.complete_miner_weight_upload(hotkey)
            self.miners_with_submitted_scores[layer][hotkey] = (
                weights_path,
                metadata_path,
                optimizer_state_path,
                optimizer_state_metadata_path,
            )
            if hotkey in await self.validator_pool.get_tracked_miners():
                await self.validator_pool.validate_weights(
                    weights_path=weights_path,
                    metadata_path=metadata_path,
                    optimizer_state_path=optimizer_state_path,
                    miner_hotkey=hotkey,
                )
                await self.validator_pool.reset_validators()
            else:
                logger.debug(f"Miner {hotkey} is not in the validator pool, skipping weight validation")
            num_miners_with_enough_backwards = len(
                [
                    m
                    for m in self.miner_registry.get_miners_in_layer(layer)
                    if m.backwards_since_reset >= settings.GLOBAL_OPTIMIZER_STEPS
                ]
            )
            logger.debug(
                f"{len(self.miners_with_submitted_scores[layer])} out of {num_miners_with_enough_backwards} of miners in layer {layer} have submitted weights. Miners: {self.miners_with_submitted_scores[layer]}"
            )

            # If not enough miners have submitted weights, we don't need to do anything
            if (
                len(self.miners_with_submitted_scores[layer])
                < settings.MINERS_REQUIRED_FOR_WEIGHT_UPLOADING * num_miners_with_enough_backwards
            ):
                return "success"

            logger.success(
                f"\n~\n~\n~\n~\n~\n~\n~\n~LAYER {layer} HAS ENOUGH MINERS WITH SUBMITTED WEIGHTS TO START MERGING\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~\n~"
            )

            # Update merge session status for transition to partition merging
            self.weight_merging_metrics_collector.update_session_status(
                layer,
                "weights_collection_completed",
                {
                    "total_weights_received": len(self.miners_with_submitted_scores[layer]),
                    "required_miners": settings.MINERS_REQUIRED_FOR_WEIGHT_UPLOADING * num_miners_with_enough_backwards,
                },
            )

            # Reset previous partition manager and assign the correct partitions to each miner
            self.partition_manager.reset_partition_manager(layer=layer)
            self.partition_manager.create_partition_mappings(
                submitted_weights=self.miners_with_submitted_scores[layer],
                layer=layer,
                registry=self.miner_registry,
            )

            # Once enough weights are there, we start the next merging phase
            logger.info(
                f"LAYER {layer} MOVING TO MERGING PHASE WITH {len(self.partition_manager.get_layer_partitions(layer=layer))} PARTITIONS AND {len(self.miners_with_submitted_scores[layer])}/{self.miner_registry.get_miners_in_layer(layer=layer)} MINERS WITH SUBMITTED WEIGHTS"
            )

            await self.merging_phases[layer].advance_phase(
                timeout=settings.PHASE_TIMEOUT, expected_phase=MergingPhase.WEIGHTS_UPLOADING
            )

            # Update session status for partition merging phase
            self.weight_merging_metrics_collector.update_session_status(
                layer,
                "partitions_merging",
                {
                    "total_partitions": len(self.partition_manager.get_layer_partitions(layer=layer)),
                    "participating_miners": len(self.miners_with_submitted_scores[layer]),
                },
            )

            # Mark all participating miners as starting partition merge
            for miner_hotkey in self.miners_with_submitted_scores[layer].keys():
                self.miner_registry.start_miner_partition_merge(miner_hotkey)

            # Once we have merged, save the orchestrator state to the db
            await self.save_orchestrator_state_to_db(on_weights_merged=True)

            return "success"

        except Exception as e:
            logger.exception(f"Error merging weights: {e}")
            # Mark session as failed
            layer = self.miner_registry.get_miner_data(hotkey).layer
            self.weight_merging_metrics_collector.update_session_status(layer, "failed", {"error": str(e)})
            return f"Exception in notify_weights_uploaded: {e}"

    async def notify_merged_partitions_uploaded(self, hotkey: str, partitions: list[Partition]):
        logger.info(f"Notifying merged partitions uploaded for miner {hotkey}")
        layer = self.miner_registry.get_miner_data(hotkey).layer

        # Record partition completion in metrics and update miner status
        self.weight_merging_metrics_collector.record_partition_completion(layer, hotkey)

        # Mark that this miner has completed partition merging
        self.miner_registry.complete_miner_partition_merge(hotkey)

        assert len(partitions) == len(
            self.partition_manager.get_chunks_for_miner(hotkey, layer)[1]
        ), f"Partition paths and chunks for miner must be the same length, but are {len(partitions)} and {len(self.partition_manager.get_chunks_for_miner(hotkey, layer)[1])}"

        for miner_partition in partitions:
            assert (
                miner_partition.miner_hotkey == hotkey
            ), f"Miner hotkey must match, got: {miner_partition.miner_hotkey} and {hotkey}"
            logger.debug(
                f"Miner {hotkey} received merged partition from miner {miner_partition.miner_hotkey}; Partition: {miner_partition}"
            )
            partition = self.partition_manager.get_partition(miner_partition)
            partition.weight_path = miner_partition.weight_path
            partition.weight_metadata_path = miner_partition.weight_metadata_path
            partition.optimizer_state_path = miner_partition.optimizer_state_path
            partition.optimizer_state_metadata_path = miner_partition.optimizer_state_metadata_path
            partition.weight_data = miner_partition.weight_data
            partition.optimizer_state_data = miner_partition.optimizer_state_data

        # If not enough miners have uploaded their merged partitions, we don't need to do anything
        if (
            len(self.partition_manager.get_layer_partitions(layer, completed_only=True))
            <= len(self.partition_manager.get_layer_partitions(layer)) * settings.MINER_MERGE_PARTITIONS
            or self.merging_phases[layer].stage != MergingPhase.MINERS_MERGING_PARTITIONS
        ):
            return

        logger.info(
            f"LAYER {layer} COMPLETED MERGING PHASE WITH {len(self.partition_manager.get_layer_partitions(layer=layer, completed_only=True))} COMPLETED PARTITIONS OUT OF {len(self.partition_manager.get_layer_partitions(layer=layer))} AND {len(self.miners_with_submitted_scores[layer])}/{self.miner_registry.get_miners_in_layer(layer=layer)} MINERS WITH SUBMITTED WEIGHTS"
        )

        # Record merge session completion
        self.weight_merging_metrics_collector.update_session_status(
            layer,
            "completed",
            {
                "completed_partitions": len(
                    self.partition_manager.get_layer_partitions(layer=layer, completed_only=True)
                ),
                "total_partitions": len(self.partition_manager.get_layer_partitions(layer=layer)),
                "participating_miners": len(self.miners_with_submitted_scores[layer]),
            },
        )

        # Assign new layer weights
        # TODO: Implement layer weights to be sharded
        merged_partitions = self.partition_manager.get_layer_partitions(layer)
        valid_partitions = []
        for partition in merged_partitions:
            if partition.weight_path is None or partition.optimizer_state_path is None:
                logger.warning(f"Partition {partition} has no path")
            else:
                valid_partitions.append(partition)
        logger.debug(f"Merged partitions: {valid_partitions}")
        await self.weight_store.set_layer_partitions(layer=layer, partitions=valid_partitions)
        await self.activation_store.reset_layer(layer)

        self.miners_with_submitted_scores[layer] = {}

        await self.merging_phases[layer].advance_phase(
            timeout=settings.PHASE_TIMEOUT,
            expected_phase=MergingPhase.MINERS_MERGING_PARTITIONS,
        )

        for m in self.miner_registry.get_miners_in_layer(layer):
            self.miner_registry.set_miner_attribute(miner_hotkey=m.hotkey, attribute="backwards_since_reset", value=0)
            # Clear miner cache during merging phase (miners reset their cache during weight merging)
            try:
                logger.debug(
                    f"Clearing cache for miner {m.hotkey[:8]} during merging phase, previous cache: {m.cached_activations}"
                )
                self.miner_registry.clear_miner_cache(m.hotkey)
                logger.debug(f"Cleared cache for miner {m.hotkey[:8]} during merging phase")
            except Exception as e:
                logger.warning(f"Failed to clear cache for miner {m.hotkey[:8]} during merging: {e}")

        # Reset merge status for all miners in this layer
        for miner_data in self.miner_registry.get_miners_in_layer(layer):
            self.miner_registry.update_miner_merge_status(miner_data.hotkey, "idle")

        # Send aggregated loss data to dashboard after weight merge completion
        if settings.ENABLE_DASHBOARD_REPORTING and self.dashboard_reporter:
            await self.dashboard_reporter.send_aggregated_loss()
            if settings.DASHBOARD_LOGS:
                logger.info(f"Sent aggregated loss data to dashboard after layer {layer} weight merge completion")

    async def get_chunks_for_miner(self, hotkey: str) -> tuple[list[SubmittedWeights], list[int]]:
        """Get the chunk locations, chunk numbers, and chunk weight factor for the miner to enable
        butterfly all-reduce.

        A miner will request the information needed for butterfly all-reduce from the orchestrator.
        The orchestrator will then return the information needed for the miner to download the correct chunks from the database.

        Args:
            hotkey: The hotkey of the miner
        """

        layer = self.miner_registry.get_miner_data(hotkey).layer
        information_packets, chunk_numbers = self.partition_manager.get_chunks_for_miner(hotkey=hotkey, layer=layer)

        return information_packets, chunk_numbers

    async def close(self):
        """Clean up resources."""
        if self.dashboard_reporter:
            await self.dashboard_reporter.close()

    def _load_saved_scores(self):
        # Try to load previous scores from S3
        try:
            from utils.s3_interactions import s3_client

            if s3_client:
                response = s3_client.get_object(Bucket=settings.S3_BUCKET, Key=CHAIN_SCORES_LOCATIONS)
                scores = json.loads(response["Body"].read())
                logger.info(f"Loaded previous scores from S3: {CHAIN_SCORES_LOCATIONS}")

                # Add scores with current timestamp
                current_time = time.time()
                for uid_str, score in scores.items():
                    self.global_miner_scores[int(uid_str)].append((current_time, score))

        except Exception as e:
            logger.warning(f"Could not load previous scores from S3: {e}")

        logger.success(f"Global miner scores: {self.global_miner_scores}")

    async def initialize(self):
        """Initialize the orchestrator and its components."""
        # Initialize base components first
        self._initialize_miner_registry()
        self._load_saved_scores()

        # Initialize dashboard reporter if enabled
        if settings.ENABLE_DASHBOARD_REPORTING and settings.DASHBOARD_BASE_URL:
            self.dashboard_reporter = DashboardMetricsReporter()
            await self.dashboard_reporter.initialize()
            # Set up the callback to get miner data
            self._update_dashboard_callback()
        else:
            logger.info(
                "Dashboard reporting is disabled or DASHBOARD_BASE_URL is None. No dashboard metrics will be sent."
            )

        if settings.LOAD_MOST_RECENT_ORCHESTRATOR_STATE_ON_INITIALIZATION and settings.MONGO:
            await self._load_orchestrator_state()

    async def _load_orchestrator_state(self):
        """Load and restore orchestrator state from MongoDB.

        There are attributes of the class that are not serializable, and we store a list of these in ORCHESTRATOR_NON_SERIALIZABLE_FIELDS.
        """
        state_manager: MongoStateManager = await MongoStateManager.create()
        state: Dict[str, Any] = await state_manager.load_system_state()

        logger.debug(f"Loaded state with keys: {list(state.keys())}")

        if state is not None:
            if settings.BITTENSOR:
                mr = copy.deepcopy(state["miner_registry"])
                for loaded_miner_hotkey in mr.get_all_miner_data().keys():
                    if loaded_miner_hotkey not in self.metagraph.hotkeys:
                        logger.warning(
                            f"Miner {loaded_miner_hotkey[:8]} not found in metagraph, removing from registry"
                        )
                        state["miner_registry"].remove_miner_from_registry(miner_hotkey=loaded_miner_hotkey)
                        self.global_miner_scores.pop(
                            self.miner_registry.get_miner_data(miner_hotkey=loaded_miner_hotkey).uid, None
                        )

            try:
                # Get all field names from the Pydantic model
                all_fields = self.__class__.model_fields.keys()

                # Restore all fields that were saved
                restored_fields = []
                for field in all_fields:
                    if field not in ORCHESTRATOR_NON_SERIALIZABLE_FIELDS and field in state:
                        setattr(self, field, state[field])
                        restored_fields.append(field)

                logger.info(f"Successfully restored state from MongoDB with fields and values:\n{restored_fields}")

            except Exception as e:
                logger.error(f"Error restoring system state: {e}")
                logger.error(f"Available keys in state: {list(state.keys())}")
                logger.error(f"Model fields: {list(self.__class__.model_fields.keys())}")

    def _update_dashboard_callback(self):
        """Update the dashboard reporter callback with current orchestrator state."""
        if self.dashboard_reporter:
            self.dashboard_reporter.set_miner_data_callback(
                lambda: {
                    "miners": self.miner_registry.get_all_miner_data(),
                    "metagraph": self.metagraph,
                }
            )
            logger.debug("Updated dashboard reporter callback with current orchestrator state")

    def model_dump(self, **kwargs):
        """Custom model dump that excludes specific fields.

        This method excludes fields that shouldn't be serialized to MongoDB,
        such as locks, event loops, and other runtime-specific objects.
        """
        # Get the default dump
        dump = super().model_dump(**kwargs)

        # Fields to exclude from serialization
        exclude_fields = {
            "lock",  # threading.RLock
            "validator_init_lock",  # asyncio.Lock
            "metagraph_syncer",  # MetagraphSyncer instance
            "metagraph",  # bt.metagraph
            "config",  # config object
            "subtensor",  # bt.subtensor
            "dashboard_reporter",  # DashboardMetricsReporter
            "validator_pool",  # ValidatorClientPool
            "wallet",  # Wallet instance
        }

        # Remove excluded fields
        for field in exclude_fields:
            if field in dump:
                del dump[field]

        return dump

    async def get_miner_activation(self, hotkey: str) -> ActivationResponse:
        """Get a random activation for a miner.

        Args:
            hotkey: The hotkey of the miner
        """
        # This is a super hacky way of doing this, but we just want to make sure miners don't idle due to being considered out of cache.
        for miner in self.miner_registry.get_all_miner_data().values():
            if miner.out_of_cache:
                for activation_uid, upload_time in list(miner.cached_activations.items()):
                    if upload_time < time.time() - settings.ACTIVATION_CACHE_TIMEOUT:
                        self.miner_registry.remove_from_miner_cache(miner.hotkey, activation_uid)
                        logger.warning(
                            f"Removed activation {activation_uid} from miner {miner.hotkey[:8]} cache due to timeout, THIS SHOULDN'T REALLY BE HAPPENING"
                        )
        for activation in list(self.miner_registry.get_miner_data(miner_hotkey=hotkey).cached_activations.keys()):
            if not await self.activation_store.is_activation_active(
                layer=self.miner_registry.get_miner_data(miner_hotkey=hotkey).layer,
                activation_uid=activation,
            ):
                self.miner_registry.get_miner_data(miner_hotkey=hotkey).cached_activations.pop(activation)
                logger.debug(f"Removed inactive activation {activation} from cache for miner {hotkey}")

        miner_data = self.miner_registry.get_miner_data(hotkey)

        logger.debug(f"Getting activation for miner {hotkey} with cached activations: {miner_data.cached_activations}")
        # Find correct activation for miner
        activation_response = await self.activation_store.get_miner_activation(
            layer=miner_data.layer,
            cached_activations=miner_data.cached_activations,
            hotkey=hotkey,
        )

        # Record activation requested in metrics (only track request time)
        if activation_response.activation_uid:
            self.activation_metrics_collector.record_activation_requested(
                activation_uid=activation_response.activation_uid,
                miner_hotkey=hotkey,
                layer=miner_data.layer,
                direction=activation_response.direction,
            )

        # If the miner is out of cache, it should not do anything
        if activation_response.direction == "forward" and miner_data.out_of_cache:
            return ActivationResponse(
                activation_uid=None,
                direction=None,
                activation_path=None,
                reason="out_of_cache",
            )
        # Download activation from activation store if it exists
        if activation_response.activation_uid is not None:
            logger.debug(
                f"Downloading activation {activation_response.activation_uid} for layer {miner_data.layer} for miner {hotkey}, response: {activation_response}"
            )
            assert (
                activation_response is not None
            ), f"Activation path is required for layer {miner_data.layer}, activation: {activation_response}"

            await self.activation_store.download_activation_from_activation_store(
                activation_uid=activation_response.activation_uid,
                direction=activation_response.direction,
                delete=False,
                layer=miner_data.layer,
                miner_hotkey=hotkey,
            )

        return activation_response

    def get_miner_performance_metrics(
        self, miner_hotkey: str, time_window_seconds: Optional[float] = 3600
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific miner."""
        timing_metrics = self.activation_metrics_collector.get_miner_average_times(miner_hotkey, time_window_seconds)
        performance_metrics = self.activation_metrics_collector.miner_performance.get(miner_hotkey, {})

        return {
            "timing_metrics": timing_metrics,
            "performance_metrics": (
                performance_metrics.model_dump() if hasattr(performance_metrics, "model_dump") else performance_metrics
            ),
            "current_state": {
                "active_activations": len(
                    [
                        m
                        for m in self.activation_metrics_collector.active_metrics.values()
                        if m.miner_hotkey == miner_hotkey
                    ]
                ),
                "layer": (
                    self.miner_registry.get_miner_data(miner_hotkey).layer
                    if miner_hotkey in self.miner_registry.get_all_miner_data()
                    else None
                ),
            },
        }

    def get_layer_performance_metrics(self, layer: int, time_window_seconds: Optional[float] = 3600) -> Dict[str, Any]:
        """Get performance metrics for a specific layer."""
        layer_stats = self.activation_metrics_collector.get_layer_statistics(layer, time_window_seconds)
        queue_metrics = self.activation_metrics_collector.queue_metrics.get(layer, {})

        return {
            "layer_statistics": layer_stats,
            "queue_metrics": queue_metrics.model_dump() if hasattr(queue_metrics, "model_dump") else queue_metrics,
            "miners_in_layer": len(self.miner_registry.get_miners_in_layer(layer)),
            "system_bottleneck": layer in self.activation_metrics_collector.system_health.bottleneck_layers,
        }

    def get_real_time_performance_dashboard(self) -> Dict[str, Any]:
        """Get real-time performance data for monitoring dashboard."""
        # Update system health before returning
        self.activation_metrics_collector.update_system_health()

        # Get recent performance data
        recent_completions = [
            m for m in self.activation_metrics_collector.completed_metrics[-50:] if time.time() - m.created_at <= 300
        ]  # Last 5 minutes

        # Calculate success rates
        total_recent = len(recent_completions)
        successful_recent = sum(1 for m in recent_completions if not m.failed)
        success_rate = successful_recent / total_recent if total_recent > 0 else 0

        # Get layer performance overview
        layer_overview = {}
        for layer in range(settings.N_LAYERS):
            layer_metrics = self.get_layer_performance_metrics(layer, 300)  # 5 minute window
            layer_overview[layer] = {
                "active_count": layer_metrics["layer_statistics"].get("active_count", 0),
                "throughput": layer_metrics["layer_statistics"].get("throughput_per_minute", 0),
                "avg_processing_time": layer_metrics["layer_statistics"].get("avg_processing_time", 0),
                "success_rate": layer_metrics["layer_statistics"].get("success_rate", 0),
                "is_bottleneck": layer_metrics.get("system_bottleneck", False),
            }

        # Get top performing and problematic miners
        miner_performance = []
        for hotkey in self.miner_registry.get_all_miner_data().keys():
            perf = self.get_miner_performance_metrics(hotkey, 300)
            if perf["timing_metrics"].get("sample_count", 0) > 0:
                # Extract total activations processed from performance metrics
                total_processed = 0
                if "performance_metrics" in perf and perf["performance_metrics"]:
                    total_processed = perf["performance_metrics"].get("total_activations_processed", 0)

                miner_performance.append(
                    {
                        "hotkey": hotkey,  # Full hotkey for API calls
                        "display_hotkey": hotkey[:8] + "...",  # Truncated for display
                        "layer": perf["current_state"]["layer"],
                        "success_rate": perf["timing_metrics"].get("success_rate", 0),
                        "avg_processing_time": perf["timing_metrics"].get("avg_processing_time", 0),
                        "active_activations": perf["current_state"]["active_activations"],
                        "total_activations_processed": total_processed,
                    }
                )

        return {
            "timestamp": time.time(),
            "system_overview": {
                "total_active_activations": self.activation_metrics_collector.system_health.total_active_activations,
                "system_throughput": self.activation_metrics_collector.system_health.system_throughput,
                "average_latency": self.activation_metrics_collector.system_health.average_system_latency,
                "success_rate": success_rate,
                "bottleneck_layers": self.activation_metrics_collector.system_health.bottleneck_layers,
            },
            "layer_overview": layer_overview,
            "recent_activity": {
                "completions_last_5min": total_recent,
                "successful_completions": successful_recent,
                "failed_completions": total_recent - successful_recent,
            },
            "miner_performance": sorted(miner_performance, key=lambda x: x["success_rate"], reverse=True)[:10],
        }

    async def cleanup_stale_activation_metrics(self):
        """Clean up stale activation metrics (should be called periodically)."""
        cleaned_count = self.activation_metrics_collector.cleanup_stale_metrics()
        if cleaned_count > 0:
            logger.warning(f"Cleaned up {cleaned_count} stale activation metrics")
        return cleaned_count

    async def _update_system_metrics(self):
        """Update system-wide health metrics."""
        # Update queue metrics for each layer
        for layer in range(settings.N_LAYERS):
            try:
                # Get queue data from activation store
                queue_data = await self._get_layer_queue_data(layer)
                self.activation_metrics_collector.update_queue_metrics(layer, queue_data)
            except Exception as e:
                logger.warning(f"Failed to update queue metrics for layer {layer}: {e}")

        # Update overall system health
        self.activation_metrics_collector.update_system_health()

        # Collect time series data
        self.time_series_collector.collect_metrics_snapshot(self)

    async def _get_layer_queue_data(self, layer: int) -> Dict[str, Any]:
        """Get queue data for a specific layer from the activation store."""
        try:
            # Count activations by type and layer
            forward_count = len(
                [
                    a
                    for a in self.activation_store.activations.values()
                    if a.layer == layer and a.direction == "forward" and a.state.value == "available"
                ]
            )
            backward_count = len(
                [
                    a
                    for a in self.activation_store.activations.values()
                    if a.layer == layer and a.direction == "backward" and a.state.value == "available"
                ]
            )
            processing_count = len(
                [
                    a
                    for a in self.activation_store.activations.values()
                    if a.layer == layer and a.state.value == "processing"
                ]
            )

            # Calculate ages of activations
            current_time = time.time()
            activation_ages = []
            for activation in self.activation_store.activations.values():
                if activation.layer == layer and hasattr(activation, "created_at"):
                    age = current_time - activation.created_at
                    activation_ages.append(age)

            oldest_age = max(activation_ages) if activation_ages else 0
            avg_wait_time = sum(activation_ages) / len(activation_ages) if activation_ages else 0

            return {
                "forward_count": forward_count,
                "backward_count": backward_count,
                "processing_count": processing_count,
                "queue_depths": {"forward": forward_count, "backward": backward_count, "processing": processing_count},
                "avg_wait_time": avg_wait_time,
                "oldest_age": oldest_age,
            }
        except Exception as e:
            logger.warning(f"Error getting queue data for layer {layer}: {e}")
            return {
                "forward_count": 0,
                "backward_count": 0,
                "processing_count": 0,
                "queue_depths": {},
                "avg_wait_time": 0.0,
                "oldest_age": 0.0,
            }

    def get_weight_merging_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of weight merging metrics."""
        return {
            "active_sessions": len(self.weight_merging_metrics_collector.active_sessions),
            "completed_sessions": len(self.weight_merging_metrics_collector.completed_sessions),
            "total_tracked": len(self.weight_merging_metrics_collector.active_sessions)
            + len(self.weight_merging_metrics_collector.completed_sessions),
            "active_sessions_by_layer": {
                layer: session.session_id
                for session in self.weight_merging_metrics_collector.active_sessions.values()
                for layer in [session.layer]
            },
        }

    def get_active_merge_sessions(self) -> Dict[str, Any]:
        """Get information about currently active merge sessions."""
        sessions_info = {}
        for session_id, session in self.weight_merging_metrics_collector.active_sessions.items():
            sessions_info[session_id] = {
                "layer": session.layer,
                "status": session.status,
                "started_at": session.started_at,
                "duration_so_far": time.time() - session.started_at,
                "target_miners_count": len(session.target_miners),
                "weights_received_count": len(session.weights_received),
                "partitions_completed_count": len(session.partitions_completed),
                "participation_rate": session.get_participation_rate(),
            }
        return sessions_info

    def get_historical_processing_times(
        self, time_window_hours: int = 24, include_layer_breakdown: bool = True, granularity: str = "hourly"
    ) -> Dict[str, Any]:
        """Get historical processing time data for visualization."""
        # Determine granularity in minutes
        if granularity == "hourly" or time_window_hours <= 24:
            granularity_minutes = 10  # 10-minute buckets for detailed view
        else:
            granularity_minutes = 60  # 1-hour buckets for longer time windows

        # Collect layer data
        layer_data = {}
        for layer in range(self.N_LAYERS):
            # Get processing time data for this layer
            time_series_data = self.time_series_collector.get_time_series_data(
                metric_type="avg_processing_time",
                time_window_hours=time_window_hours,
                granularity_minutes=granularity_minutes,
                layer=layer,
            )

            if not time_series_data["timestamps"]:
                continue

            # Convert to the expected format for the dashboard
            time_series = []
            for timestamp, processing_time in zip(time_series_data["timestamps"], time_series_data["values"]):
                # Calculate variance from recent metrics for this layer
                current_time = time.time()
                recent_metrics = [
                    m
                    for m in self.activation_metrics_collector.completed_metrics
                    if (
                        m.layer == layer
                        and abs(m.created_at - timestamp) <= granularity_minutes * 60  # Within time bucket
                        and m.durations.total_processing_time is not None
                    )
                ]

                # Calculate variance and sample count
                processing_times = [m.durations.total_processing_time for m in recent_metrics]
                variance = 0.0
                sample_count = len(processing_times)

                if len(processing_times) > 1:
                    import statistics

                    variance = statistics.variance(processing_times)
                elif len(processing_times) == 1:
                    variance = 0.01  # Small default variance for single samples

                time_series.append(
                    {
                        "timestamp": timestamp,
                        "avg_processing_time": processing_time,
                        "variance": variance,
                        "sample_count": sample_count,
                    }
                )

            layer_data[str(layer)] = {"time_series": time_series}

        # Calculate summary statistics
        summary_statistics = {}
        for layer in range(self.N_LAYERS):
            layer_key = str(layer)
            if layer_key not in layer_data or not layer_data[layer_key]["time_series"]:
                continue

            # Get recent layer statistics for summary
            layer_stats = self.activation_metrics_collector.get_layer_statistics(layer, time_window_hours * 3600)

            # Extract processing times from time series for additional stats
            processing_times = [
                point["avg_processing_time"]
                for point in layer_data[layer_key]["time_series"]
                if point["avg_processing_time"] > 0
            ]

            if processing_times:
                import statistics

                summary_statistics[layer_key] = {
                    "avg_processing_time": layer_stats.get("avg_processing_time", statistics.mean(processing_times)),
                    "min_processing_time": min(processing_times),
                    "max_processing_time": max(processing_times),
                    "std_deviation": statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0,
                }

        return {
            "layer_data": layer_data,
            "summary_statistics": summary_statistics,
            "time_window_hours": time_window_hours,
            "granularity": granularity,
            "timestamp": time.time(),
        }

    def get_merge_session_timeline(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get merge session timeline data for visualization with success/failure bands."""
        current_time = time.time()
        start_time = current_time - (time_window_hours * 3600)

        # Get all sessions (completed and active) within the time window
        timeline_sessions = []

        # Add completed sessions
        for session in self.weight_merging_metrics_collector.completed_sessions:
            if session.started_at >= start_time:
                session_duration = session.get_session_duration()
                participation_rate = session.get_participation_rate()

                # Determine status color mapping
                status_color = "green"  # default for completed
                if session.status == "failed":
                    status_color = "red"
                elif session.status == "timeout":
                    status_color = "orange"
                elif session.status == "completed":
                    status_color = "green"

                # Get failure reason if available
                failure_reason = None
                if session.status in ["failed", "timeout"]:
                    # Look for failure details in status metadata
                    if hasattr(session, "status_metadata") and session.status_metadata:
                        failure_reason = session.status_metadata.get(
                            "error", session.status_metadata.get("reason", "Unknown")
                        )
                    else:
                        failure_reason = f"Session {session.status}"

                timeline_sessions.append(
                    {
                        "session_id": session.session_id,
                        "layer": session.layer,
                        "start_time": session.started_at,
                        "end_time": (
                            session.completed_at if session.completed_at else session.started_at + session_duration
                        ),
                        "duration": session_duration,
                        "status": session.status,
                        "status_color": status_color,
                        "participation_rate": participation_rate,
                        "target_miners": len(session.target_miners),
                        "weights_received": len(session.weights_received),
                        "partitions_completed": len(session.partitions_completed),
                        "failure_reason": failure_reason,
                        "weights_collection_duration": session.get_weights_collection_duration(),
                        "partition_merging_duration": session.get_partition_merging_duration(),
                        "is_active": False,
                    }
                )

        # Add currently active sessions
        for session_id, session in self.weight_merging_metrics_collector.active_sessions.items():
            if session.started_at >= start_time:
                session_duration = current_time - session.started_at  # Duration so far
                participation_rate = session.get_participation_rate()

                # Active sessions are typically "in progress" - use blue/yellow
                status_color = "blue"
                if session_duration > 600:  # > 10 minutes, might be stuck
                    status_color = "orange"

                timeline_sessions.append(
                    {
                        "session_id": session.session_id,
                        "layer": session.layer,
                        "start_time": session.started_at,
                        "end_time": current_time,  # Current time for active sessions
                        "duration": session_duration,
                        "status": session.status,
                        "status_color": status_color,
                        "participation_rate": participation_rate,
                        "target_miners": len(session.target_miners),
                        "weights_received": len(session.weights_received),
                        "partitions_completed": len(session.partitions_completed),
                        "failure_reason": None,  # Active sessions don't have failure reasons yet
                        "weights_collection_duration": None,  # May not be complete yet
                        "partition_merging_duration": None,  # May not be complete yet
                        "is_active": True,
                    }
                )

        # Sort sessions by start time
        timeline_sessions.sort(key=lambda x: x["start_time"])

        # Calculate summary statistics for the timeline
        layer_stats = {}
        for layer in range(self.N_LAYERS):
            layer_sessions = [s for s in timeline_sessions if s["layer"] == layer]

            if layer_sessions:
                completed_sessions = [s for s in layer_sessions if not s["is_active"]]
                successful_sessions = [s for s in completed_sessions if s["status"] == "completed"]
                failed_sessions = [s for s in completed_sessions if s["status"] in ["failed", "timeout"]]

                avg_duration = (
                    sum(s["duration"] for s in completed_sessions) / len(completed_sessions)
                    if completed_sessions
                    else 0
                )
                avg_participation = sum(s["participation_rate"] for s in layer_sessions) / len(layer_sessions)

                layer_stats[layer] = {
                    "total_sessions": len(layer_sessions),
                    "completed_sessions": len(completed_sessions),
                    "successful_sessions": len(successful_sessions),
                    "failed_sessions": len(failed_sessions),
                    "active_sessions": len([s for s in layer_sessions if s["is_active"]]),
                    "success_rate": len(successful_sessions) / len(completed_sessions) if completed_sessions else 0,
                    "avg_duration": avg_duration,
                    "avg_participation_rate": avg_participation,
                }
            else:
                layer_stats[layer] = {
                    "total_sessions": 0,
                    "completed_sessions": 0,
                    "successful_sessions": 0,
                    "failed_sessions": 0,
                    "active_sessions": 0,
                    "success_rate": 0,
                    "avg_duration": 0,
                    "avg_participation_rate": 0,
                }

        # Create time buckets for system-wide activity overview
        bucket_size = 3600  # 1-hour buckets
        time_buckets = {}

        for session in timeline_sessions:
            bucket_key = int(session["start_time"] // bucket_size) * bucket_size
            if bucket_key not in time_buckets:
                time_buckets[bucket_key] = {
                    "timestamp": bucket_key,
                    "total_sessions": 0,
                    "successful_sessions": 0,
                    "failed_sessions": 0,
                    "active_sessions": 0,
                    "avg_duration": 0,
                    "avg_participation": 0,
                }

            bucket = time_buckets[bucket_key]
            bucket["total_sessions"] += 1

            if session["is_active"]:
                bucket["active_sessions"] += 1
            elif session["status"] == "completed":
                bucket["successful_sessions"] += 1
            elif session["status"] in ["failed", "timeout"]:
                bucket["failed_sessions"] += 1

        # Calculate averages for time buckets
        for bucket in time_buckets.values():
            bucket_sessions = [
                s for s in timeline_sessions if int(s["start_time"] // bucket_size) * bucket_size == bucket["timestamp"]
            ]
            if bucket_sessions:
                bucket["avg_duration"] = sum(s["duration"] for s in bucket_sessions) / len(bucket_sessions)
                bucket["avg_participation"] = sum(s["participation_rate"] for s in bucket_sessions) / len(
                    bucket_sessions
                )

        return {
            "timeline_sessions": timeline_sessions,
            "layer_statistics": layer_stats,
            "time_buckets": list(time_buckets.values()),
            "time_window_hours": time_window_hours,
            "total_sessions": len(timeline_sessions),
            "timestamp": current_time,
            "start_time": start_time,
            "end_time": current_time,
        }

    async def submit_miner_scores(self, scores: dict[int, float]):
        """Submit scores for miners to be tracked globally.

        Args:
            scores: Dictionary mapping miner UIDs to their score values
        """
        timestamp = time.time()
        # update global scores history
        for uid, score in scores.items():
            score_in_history = self.global_miner_scores.get(uid, [])
            score_in_history.append(
                (timestamp, score)
            )  # This will also add it to self.global_miner_scores due to python memory reference

            historical_scores = []

            # remove scores older than SCORE_VALIDITY_PERIOD before updating the global scores history
            for historical_timestamp, historical_score in score_in_history:
                if historical_timestamp > time.time() - settings.SCORE_VALIDITY_PERIOD:
                    historical_scores.append((historical_timestamp, historical_score))

            self.global_miner_scores[uid] = historical_scores

    async def get_global_miner_scores(self):
        """Calculate sum of scores for each miner in the last SCORE_VALIDITY_PERIOD seconds.

        Returns:
            dict: Dictionary mapping miner UIDs to their sum of scores in the last SCORE_VALIDITY_PERIOD seconds
        """
        current_scores = {}

        # give a base score if you're registered with the orchestrator.
        for miner_data in self.miner_registry.get_all_miner_data().values():
            current_scores[miner_data.uid] = 1

        for uid, score_history in self.global_miner_scores.items():
            if not score_history:
                continue

            # get sum of scores since it's already been filtered by time in the submit_miner_scores method
            sum_scores = sum(score for _, score in score_history)

            # add the sum of scores to the current scores
            if uid in current_scores:
                current_scores[uid] += sum_scores
            else:
                current_scores[uid] = sum_scores
                logger.warning(
                    f"Miner {uid} not found in the miner_registry, but it's in the global_miner_scores... This shouldn't happen."
                )

        # Upload the current scores to the S3 bucket
        self.upload_data_to_s3(data=current_scores, path=CHAIN_SCORES_LOCATIONS)
        return current_scores

    def upload_data_to_s3(self, data, path: str):
        try:
            scores_json = json.dumps(data)
            scores_bytes = scores_json.encode("utf-8")

            # Get presigned URL and upload
            presigned_data = generate_presigned_url(path=path)
            buffer = io.BytesIO(scores_bytes)
            upload_to_bucket(presigned_data, {"file": ("data", buffer)})
        except Exception as e:
            logger.error(f"Failed to upload miner scores to S3: {e}")

    def get_miners_grid_status(self) -> Dict[str, Any]:
        """Get comprehensive miner status data for grid visualization."""
        # Get the base grid data from the miner registry
        grid_data = self.miner_registry.get_miners_grid_data()

        # Enhance with real-time merge session information
        for layer, miners in grid_data["layers"].items():
            for miner_info in miners:
                miner_hotkey = miner_info["hotkey"]

                # Get current merge status from the weight merging collector
                merge_status = self.weight_merging_metrics_collector.get_miner_current_status(miner_hotkey)
                merge_progress = self.weight_merging_metrics_collector.get_miner_progress(miner_hotkey)
                session = self.weight_merging_metrics_collector.get_session_for_miner(miner_hotkey)

                # Override status with more accurate merge session data
                if session:
                    miner_info["status"] = merge_status
                    miner_info["progress"] = merge_progress
                    miner_info["merge_session_id"] = session.session_id
                else:
                    # Use the miner's current activity status if not in a merge session
                    miner_data = self.miner_registry.get_miner_data(miner_hotkey)
                    if miner_data:
                        miner_info["status"] = miner_data.status  # forward, backward, idle
                        miner_info["progress"] = 0.0
                        miner_info["merge_session_id"] = None

                # Add performance metrics from the activation metrics collector
                try:
                    perf_metrics = self.get_miner_performance_metrics(miner_hotkey, 3600)  # 1 hour window
                    miner_info["performance_metrics"].update(
                        {
                            "success_rate": perf_metrics["timing_metrics"].get("success_rate", 0),
                            "avg_response_time": perf_metrics["timing_metrics"].get("avg_processing_time", 0),
                            "total_activations_processed": perf_metrics.get("performance_metrics", {}).get(
                                "total_activations_processed", 0
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to get performance metrics for miner {miner_hotkey[:8]}...: {e}")

        return grid_data

    def get_miner_detail(self, miner_hotkey: str) -> Dict[str, Any]:
        """Get detailed information about a specific miner for hover display."""
        try:
            # Get miner data
            miner_data = self.miner_registry.get_miner_data(miner_hotkey)
            if not miner_data:
                return {"error": f"Miner {miner_hotkey} not found"}

            # Get performance metrics
            perf_metrics = self.get_miner_performance_metrics(miner_hotkey, 3600)  # 1 hour window

            # Get merge session information
            merge_status = self.weight_merging_metrics_collector.get_miner_current_status(miner_hotkey)
            session = self.weight_merging_metrics_collector.get_session_for_miner(miner_hotkey)
            merge_performance = self.weight_merging_metrics_collector.get_miner_merge_performance(
                miner_hotkey, 24 * 3600
            )  # 24 hours

            # Calculate uptime
            current_time = time.time()
            time_since_registration = current_time - miner_data.registration_time
            time_since_activity = current_time - miner_data.last_activity_timestamp

            # Build detailed response
            detail = {
                "basic_info": {
                    "hotkey": miner_hotkey,
                    "display_hotkey": f"{miner_hotkey[:10]}...{miner_hotkey[-10:]}",
                    "layer": miner_data.layer,
                    "status": miner_data.status,
                    "merge_status": merge_status,
                    "registered_since": time_since_registration / 3600,  # hours
                    "last_activity": time_since_activity / 60,  # minutes
                },
                "current_activity": {
                    "backwards_since_reset": miner_data.backwards_since_reset,
                    "processed_activations": miner_data.processed_activations,
                    "cached_activations": len(miner_data.cached_activations),
                    "is_out_of_cache": miner_data.out_of_cache,
                    "responsible_validator": miner_data.responsible_validator,
                },
                "performance_metrics": {
                    "success_rate": perf_metrics["timing_metrics"].get("success_rate", 0),
                    "avg_processing_time": perf_metrics["timing_metrics"].get("avg_processing_time", 0),
                    "total_activations_processed": perf_metrics.get("performance_metrics", {}).get(
                        "total_activations_processed", 0
                    ),
                    "active_activations": perf_metrics["current_state"].get("active_activations", 0),
                },
                "merge_performance": {
                    "participation_count": merge_performance.get("participation_count", 0),
                    "weight_participation_rate": merge_performance.get("weight_participation_rate", 0),
                    "partition_participation_rate": merge_performance.get("partition_participation_rate", 0),
                    "avg_weight_upload_time": merge_performance.get("avg_weight_upload_time", 0),
                    "avg_partition_completion_time": merge_performance.get("avg_partition_completion_time", 0),
                    "last_weight_upload_duration": miner_data.get_weight_upload_duration(),
                    "last_partition_merge_duration": miner_data.get_partition_merge_duration(),
                },
                "current_session": None,
            }

            # Add current session details if in an active session
            if session:
                detail["current_session"] = {
                    "session_id": session.session_id,
                    "layer": session.layer,
                    "status": session.status,
                    "started_at": session.started_at,
                    "duration_so_far": current_time - session.started_at,
                    "target_miners_count": len(session.target_miners),
                    "weights_received_count": len(session.weights_received),
                    "partitions_completed_count": len(session.partitions_completed),
                    "participation_rate": session.get_participation_rate(),
                    "has_uploaded_weights": miner_hotkey in session.weights_received,
                    "has_completed_partitions": miner_hotkey in session.partitions_completed,
                }

            return detail

        except Exception as e:
            logger.error(f"Error getting miner detail for {miner_hotkey}: {e}")
            return {"error": f"Failed to get miner details: {str(e)}"}

    def update_miner_merge_status(self, miner_hotkey: str, status: str, session_id: Optional[str] = None):
        """Update miner's merge participation status.

        Args:
            miner_hotkey: The hotkey of the miner
            status: The new merge status
            session_id: Optional session ID for tracking
        """
        try:
            # Update in miner registry
            self.miner_registry.update_miner_merge_status(miner_hotkey, status, session_id)

            # Update in weight merging metrics collector if session_id provided
            if session_id:
                self.weight_merging_metrics_collector.update_miner_merge_status(miner_hotkey, status, session_id)

            logger.debug(f"Updated miner {miner_hotkey[:8]}... merge status to {status}")

        except Exception as e:
            logger.error(f"Failed to update miner {miner_hotkey[:8]}... merge status: {e}")

    def get_miners_by_merge_status(self, layer: Optional[int] = None) -> Dict[str, List[str]]:
        """Get miners grouped by their current merge status.

        Args:
            layer: Optional layer filter

        Returns:
            Dict mapping status to list of miner hotkeys
        """
        return self.weight_merging_metrics_collector.get_miners_by_status(layer)


async def initialize_orchestrator() -> Orchestrator:
    """Initialize the orchestrator with all required components."""
    logger.info("Initializing orchestrator")

    # Initialize base components
    activation_store = ActivationStore(N_LAYERS=settings.N_LAYERS)
    weight_store = WeightStore()
    validator_pool = ValidatorClientPool()

    # Create orchestrator instance
    orchestrator = Orchestrator(
        activation_store=activation_store,
        validator_pool=validator_pool,
        weight_store=weight_store,
        N_LAYERS=settings.N_LAYERS,
        miner_registry=MinerRegistry(miner_hotkeys=[]),
        wallet_name=(settings.ORCHESTRATOR_WALLET_NAME if settings.ORCHESTRATOR_WALLET_NAME else settings.wallet_name),
        wallet_hotkey=(
            settings.ORCHESTRATOR_WALLET_HOTKEY if settings.ORCHESTRATOR_WALLET_HOTKEY else settings.wallet_hotkey
        ),
    )

    if settings.BITTENSOR:
        metagraph_syncer = MetagraphSyncer(subtensor)
        metagraph_syncer.do_initial_sync()
        metagraph_syncer.start()
        orchestrator.setup_metagraph_sync(metagraph_syncer=metagraph_syncer, netuid=settings.netuid)

    # Initialize orchestrator
    await orchestrator.initialize()

    return orchestrator


# Initialize the orchestrator
orchestrator = asyncio.run(initialize_orchestrator())
