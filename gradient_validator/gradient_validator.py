import asyncio
import json
import os
import time
from typing import Literal

import bittensor as bt
import numpy as np
import requests
import torch
from loguru import logger

import model.utils as model_utils
import settings
from base.base_neuron import BaseNeuron
from miner.api_client import APIClient
from settings import (
    ACTIVATION_MAGNITUDE_THRESHOLD,
    BITTENSOR,
    COSINE_SIMILARITY_THRESHOLD,
    DEVICE,
    MOCK,
    N_LAYERS,
    PACK_SAMPLES,
    SEQUENCE_LENGTH,
    VALIDATE,
    WEIGHT_MAGNITUDE_THRESHOLD,
)
from storage.serializers import StorageResponse, ValidationEvent
from utils.s3_interactions import download_activation, download_weights_or_optimizer_state
from utils.vector_utils import flatten_optimizer_state

PENALTY_RATE = 3


class GradientValidator(BaseNeuron):
    available: bool = True
    tracked_miner: int | None = None
    weight_version: str | None = None
    miner_weights: dict[int, float] = {}
    external_ip: str | None = None
    validation_events: list[ValidationEvent] = []

    @classmethod
    async def create(cls):
        """Factory method to create and initialize a GradientValidator instance"""
        validator = cls()
        await validator.initialize()  # Initialize the base class
        validator.metagraph_syncer.register_listener(validator._on_metagraph_updated, netuids=[validator.netuid])
        while True:
            if await validator.api_client.health_check():
                break
            logger.warning("Orchestrator is not healthy, waiting for it to come online")
            await asyncio.sleep(5 if settings.MOCK else 30)
        validator.external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        logger.debug(f"External IP: {validator.external_ip}")
        await validator.api_client.register_validator(
            host=validator.external_ip,
            port=int(settings.VALIDATOR_EXTERNAL_PORT),
            scheme=settings.VALIDATOR_SCHEME,
        )
        return validator

    def _on_metagraph_updated(self, metagraph: bt.metagraph, netuid: int):
        """Processes an update to the metagraph."""
        pass

    def get_miner_info(self, miner_hotkey: int) -> dict:
        """Get information about a miner from the metagraph."""
        if not self.metagraph:
            return {}

        miner_uid = self.metagraph.hotkeys.index(miner_hotkey)
        with self.lock:
            return {
                "uid": miner_uid,
                "hotkey": self.metagraph.hotkeys[miner_uid],
                "coldkey": self.metagraph.coldkeys[miner_uid],
                "stake": float(self.metagraph.S[miner_uid]),
                "trust": float(self.metagraph.T[miner_uid]),
                "consensus": float(self.metagraph.C[miner_uid]),
                "incentive": float(self.metagraph.I[miner_uid]),
                "dividends": float(self.metagraph.D[miner_uid]),
                "emission": float(self.metagraph.E[miner_uid]),
                "active": bool(self.metagraph.active[miner_uid]),
                "validator_permit": bool(self.metagraph.validator_permit[miner_uid]),
                "last_update": int(self.metagraph.last_update[miner_uid]),
            }

    async def load_weights(self, layer: int, miner_hotkey: str, weight_path: str | None = None):
        """Enhanced load_weights that uses metagraph information."""
        self.tracked_miner = miner_hotkey
        self.layer = layer
        self.available = False
        self.miner_weights[self.tracked_miner] = 0

        self._clean_gpu_memory()

        # Load the model
        await self._load_model()
        await self._load_optimizer()
        logger.warning(f"LOADING WEIGHTS FOR MINER {miner_hotkey} with weights path {weight_path}")
        try:
            self.weights, self.optimizer = await self.download_weights()
            torch.nn.utils.vector_to_parameters(self.weights.to(DEVICE), self.model.parameters())
        except Exception as e:
            self.weights = torch.nn.utils.parameters_to_vector(self.model.parameters())
            logger.warning(f"Using default weights for miner {miner_hotkey} because of error: {e}")

        if self.lr_scheduler is None:
            await self._load_lr_scheduler_2()
        if self.layer == 0 or self.layer == N_LAYERS - 1:
            await self._load_vocab_info()
        logger.debug(f"GRADIENT VALIDATOR TRACKING MINER {miner_hotkey}")

    async def forward(self, activation_uid: str, direction: Literal["forward", "backward", "initial"]):
        logger.debug(f"GRADIENT VALIDATOR FORWARD: {activation_uid}, {direction}")
        try:
            if not VALIDATE:
                logger.warning("NOT VALIDATING BECAUSE VALIDATE IS FALSE")
                return True, torch.Tensor([0]), "no-validation-possible"
            logger.debug(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: LAYER {self.layer}, DIRECTION {direction}, ACTIVATION UID {activation_uid}"
            )
            if (
                self.backwards_since_reduce >= settings.LOCAL_OPTIMIZER_STEPS
                and settings.LOCAL_OPTIMIZER_STEPS < settings.GLOBAL_OPTIMIZER_STEPS
            ):
                await self.local_all_reduce()
                self.saved_forward_activations = {}
                self.backwards_since_reduce = 0

            if direction == "backward" and self.layer == N_LAYERS - 1:
                return await self.backward(activation_uid=activation_uid)

            if self.layer == 0 and direction == "forward":
                # Download activations regardless of direction
                activation_response = await self.api_client.download_activation_from_orchestrator(
                    activation_uid=activation_uid,
                    direction="initial",
                    delete=False,
                    layer=N_LAYERS - 1,
                    fetch_historic=True,
                )

            else:
                activation_response: StorageResponse = await self.api_client.download_activation_from_orchestrator(
                    activation_uid=activation_uid,
                    direction=direction,
                    delete=False,
                    layer=self.layer,
                    fetch_historic=True,
                )

            activation_path = activation_response.data["path"]
            if activation_path is None:
                logger.warning(
                    f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: NO ACTIVATION PATH FOR ACTIVATION {activation_uid} DIRECTION {direction}"
                )
                self.miner_weights[self.tracked_miner] -= PENALTY_RATE
                return False, torch.Tensor([0]), "no-activation-path"

            activations = download_activation(activation_path)
            # Handle backward pass early
            if direction == "backward":
                logger.debug(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: BACKWARD PASS, LAYER {self.layer}")
                return await self.backward(activation_uid=activation_uid, backward_activations=activations.to(DEVICE))
            # Forward pass processing
            validator_activations, state = await self._forward(input_activations=activations.to(DEVICE))

            # Special case for last layer
            if self.layer == N_LAYERS - 1:
                response: StorageResponse = await self.api_client.download_activation_from_orchestrator(
                    activation_uid=activation_uid,
                    direction="initial",
                    layer=self.layer,
                    delete=False,
                    fetch_historic=True,
                )
                initial_activations = download_activation(response.data["path"]).to(DEVICE)
                if not MOCK:
                    output_activations = model_utils.compute_loss(
                        logits=validator_activations,
                        targets=initial_activations,
                        vocab_size=self.vocab_size,
                        pad_token_id=self.eos_token_id,
                        pack=PACK_SAMPLES,
                    )
                else:
                    output_activations = torch.Tensor([0.5]).to(DEVICE).requires_grad_(True)
                self.saved_forward_activations[activation_uid] = (
                    activations,
                    output_activations,
                    state,
                )

                return True, torch.Tensor([0]), "no-validation-possible"

            # Middle layers - validate against miner activations
            miner_response: StorageResponse = await self.api_client.download_activation_from_orchestrator(
                activation_uid=activation_uid,
                direction=direction,
                delete=False,
                layer=self.layer + 1,
                fetch_historic=True,
            )

            if miner_response.data["path"] is None:
                logger.error(
                    f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: NO ACTIVATION PATH FOR ACTIVATION {activation_uid} DIRECTION {direction} LAYER {self.layer + 1}"
                )
                self.miner_weights[self.tracked_miner] -= PENALTY_RATE
                return False, torch.Tensor([0]), "no-activation-path"

            miner_activations = download_activation(miner_response.data["path"])

            # Validate and store results
            is_valid, score, reason = await self.validate_activations(
                validator_activations, miner_activations.to(DEVICE), direction="forward", activation_uid=activation_uid
            )
            self.processed_forward_activations.append(activation_uid)
            self.saved_forward_activations[activation_uid] = (
                activations,
                validator_activations,
                state,
            )

            logger.debug(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: FORWARD PASS COMPLETE: VALID: {is_valid}, SCORE: {score}, REASON: {reason}"
            )
            return is_valid, score, reason

        except Exception as e:
            logger.exception(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: Error during forward pass: {e}")
            return False, torch.Tensor([0]), "error-during-forward-pass"

    async def backward(
        self,
        activation_uid: str | None = None,
        backward_activations: torch.Tensor | None = None,
    ) -> tuple[bool, torch.Tensor, str]:
        try:
            # backward pass
            if self.layer != N_LAYERS - 1 and N_LAYERS > 1:
                response: StorageResponse = await self.api_client.download_activation_from_orchestrator(
                    activation_uid=activation_uid,
                    direction="backward",
                    delete=False,
                    layer=self.layer,
                    fetch_historic=True,
                )
                backward_activations = response.data["path"]
                backward_activations = download_activation(backward_activations).to(DEVICE)
            # TODO: If a new miner is being tracked, we may not have the forward activations in the GradientValidator
            # as they were computed in the miner before we started tracking it. We need to handle this case.
            try:
                input_activations, output_activations, state = self.saved_forward_activations[activation_uid]
            except KeyError:
                logger.warning(
                    f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: Miner could not be scores as forward activations are not available"
                )
                return False, torch.Tensor([0]), "no-forward-activations"
            self.backwards_since_reduce += 1
            self.backwards_since_sync += 1
            await self._backward(output_activations, backward_activations, state)
            # We need to find a way to verify that layer 0 does its work correctly, but it has no gradients wrt the input activations
            if MOCK:
                input_activation_grads = input_activations.detach().clone()
            elif self.layer == 0:
                # Get the embedding layer weight grads  instead of the input activations grads
                # This is because input activation grads of the first layer do not exist.
                emb_weight = self.model.tok_emb.weight
                grad_size = (
                    settings.MODEL_CFG["bottleneck_dim"]
                    if settings.MODEL_CFG["bottleneck_dim"] is not None
                    else settings.MODEL_CFG["emb_dim"]
                )
                input_activation_grads = emb_weight.grad[:SEQUENCE_LENGTH, :grad_size]

                # Detach and convert to bfloat16 to ensure we only save the values
                input_activation_grads = input_activation_grads.detach().to(torch.bfloat16).cpu()
            else:
                input_activation_grads = input_activations.grad

            miner_activations: StorageResponse = await self.api_client.download_activation_from_orchestrator(
                activation_uid=activation_uid,
                direction="backward",
                delete=False,
                layer=self.layer - 1,
                fetch_historic=True,
            )

            logger.debug(f"MINER ACTIVATIONS: {miner_activations}")

            miner_activations = download_activation(path=miner_activations.data["path"])
            is_valid, score, reason = await self.validate_activations(
                validator_activations=input_activation_grads,
                miner_activations=miner_activations.to(DEVICE),
                direction="backward",
                activation_uid=activation_uid,
            )

            self._clear_cache_entry(activation_uid=activation_uid)

            self.processed_backward_activations.append(activation_uid)
            logger.debug(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: BACKWARD PASS COMPLETE: VALID: {is_valid}, SCORE: {score}, REASON: {reason}"
            )
            return is_valid, score, reason

        except Exception as e:
            logger.exception(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: Error during backward pass: {e}")
            # self.miner_weights[self.tracked_miner] -= PENALTY_RATE
            return False, torch.Tensor([0]), "error-during-backward-pass"

    async def validate_activations(
        self,
        validator_activations: torch.Tensor,
        miner_activations: torch.Tensor,
        direction: Literal["forward", "backward"],
        activation_uid: str | None = None,
    ) -> tuple[bool, torch.Tensor, str]:
        """
        Validate the activations of the miner against the validator's activations.
        First checks magnitude ratio as a gatekeeper, then cosine similarity if magnitude check passes.
        """
        await self.save_activations(validator_activations, miner_activations, self.layer, direction=direction)

        # Flatten tensors for validation
        validator_flat = validator_activations.flatten().to(DEVICE)
        miner_flat = miner_activations.flatten().to(DEVICE)

        # Calculate norms for logging
        validator_norm = torch.norm(validator_flat).item()
        miner_norm = torch.norm(miner_flat).item()

        if MOCK:
            # In mock mode, use simple cosine similarity
            validator_flat = validator_flat.unsqueeze(0)
            miner_flat = miner_flat.unsqueeze(0)
            similarity = torch.nn.functional.cosine_similarity(validator_flat, miner_flat, dim=1)
            passed = (similarity > COSINE_SIMILARITY_THRESHOLD).item()
            score = similarity.item() if similarity.numel() == 1 else similarity[0].item()
            reason = "passed" if passed else "failed"

            # Log validation event
            self._log_validation_event(
                event_type="activation_validation",
                direction=direction,
                activation_uid=activation_uid,
                success=passed,
                score=score,
                reason=reason,
                validator_norm=validator_norm,
                miner_norm=miner_norm,
                magnitude_ratio=None,
            )

            return passed, similarity, reason
        else:
            # Step 1: Magnitude ratio check (gatekeeper)
            # R(v, m) = min(||v||, ||m||) / (max(||v||, ||m||) + ε)
            eps = 1e-8
            magnitude_ratio = torch.min(torch.tensor(validator_norm), torch.tensor(miner_norm)) / (
                torch.max(torch.tensor(validator_norm), torch.tensor(miner_norm)) + eps
            )
            magnitude_ratio_value = magnitude_ratio.item()

            if magnitude_ratio_value < ACTIVATION_MAGNITUDE_THRESHOLD:
                logger.warning(
                    f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: MAGNITUDE CHECK FAILED - "
                    f"ratio: {magnitude_ratio_value:.4f}, validator_norm: {validator_norm:.4f}, "
                    f"miner_norm: {miner_norm:.4f}, direction: {direction}"
                )
                # Log validation event
                self._log_validation_event(
                    event_type="activation_validation",
                    direction=direction,
                    activation_uid=activation_uid,
                    success=False,
                    score=magnitude_ratio_value,
                    reason="magnitude-ratio-failed",
                    validator_norm=validator_norm,
                    miner_norm=miner_norm,
                    magnitude_ratio=magnitude_ratio_value,
                )
                # self.miner_weights[self.tracked_miner] -= PENALTY_RATE
                return False, magnitude_ratio, "magnitude-ratio-failed"

            # Step 2: Cosine similarity check (only if magnitude check passed)
            validator_flat = validator_flat.unsqueeze(0)
            miner_flat = miner_flat.unsqueeze(0)
            similarity = torch.nn.functional.cosine_similarity(validator_flat, miner_flat, dim=1)
            passed = (similarity > COSINE_SIMILARITY_THRESHOLD).item()
            score = similarity.item() if similarity.numel() == 1 else similarity[0].item()
            reason = "passed" if passed else "failed"

            # Log validation event
            self._log_validation_event(
                event_type="activation_validation",
                direction=direction,
                activation_uid=activation_uid,
                success=passed,
                score=score,
                reason=reason,
                validator_norm=validator_norm,
                miner_norm=miner_norm,
                magnitude_ratio=magnitude_ratio_value,
            )

            return passed, similarity, reason

    def _log_validation_event(
        self,
        event_type: str,
        success: bool,
        score: float,
        reason: str,
        direction: str | None = None,
        activation_uid: str | None = None,
        validator_norm: float | None = None,
        miner_norm: float | None = None,
        magnitude_ratio: float | None = None,
    ):
        """Log a validation event for later analysis"""
        # tracked_miner is now the hotkey directly
        miner_hotkey = self.tracked_miner

        event = ValidationEvent(
            timestamp=time.time(),
            event_type=event_type,
            miner_hotkey=miner_hotkey,
            layer=getattr(self, "layer", None),
            direction=direction,
            activation_uid=activation_uid,
            success=success,
            score=score,
            reason=reason,
            validator_norm=validator_norm,
            miner_norm=miner_norm,
            magnitude_ratio=magnitude_ratio,
        )

        self.validation_events.append(event)
        logger.debug(f"Logged validation event: {event.event_type}, success: {event.success}, score: {event.score:.4f}")

    async def _save_validation_events(self):
        """Save validation events to a JSONL file for analysis"""
        if not self.validation_events:
            logger.debug("No validation events to save")
            return

        # Create directory if it doesn't exist
        os.makedirs("validation_events", exist_ok=True)

        # Generate filename with timestamp and miner info
        timestamp = time.time()
        miner_info = f"_miner_{self.tracked_miner[:8]}" if self.tracked_miner is not None else ""
        filename = f"validation_events_{timestamp}{miner_info}.jsonl"
        filepath = os.path.join("validation_events", filename)

        try:
            with open(filepath, "w") as f:
                for event in self.validation_events:
                    # Convert event to dict using Pydantic's model_dump
                    event_dict = event.model_dump()
                    # Write as JSON line
                    f.write(json.dumps(event_dict) + "\n")

            logger.info(f"Saved {len(self.validation_events)} validation events to {filepath}")

            # Log summary statistics
            total_events = len(self.validation_events)
            successful_events = sum(1 for event in self.validation_events if event.success)
            failed_events = total_events - successful_events

            # Group by event type
            event_types = {}
            for event in self.validation_events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

            logger.info(
                f"Validation summary: {total_events} total events, "
                f"{successful_events} successful, {failed_events} failed. "
                f"Types: {event_types}"
            )

        except Exception as e:
            logger.exception(f"Error saving validation events: {e}")

    async def validate_weights(self, weights_path: str, metadata_path: str, optimizer_state_path: str):
        if not MOCK:
            logger.debug(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: VALIDATING WEIGHTS")

            # Check if optimizer has done enough steps
            if settings.LOCAL_OPTIMIZER_STEPS >= settings.GLOBAL_OPTIMIZER_STEPS:
                await self.local_all_reduce()
            else:
                logger.error("The optimizer has not done enough steps, for some reason it's out of sync with the miner")

            # Download weights and optimizer state
            weights, _ = await self._download_chunk(weights_path, metadata_path, chunk_id="all", data_type="weights")
            miner_optimizer_state = download_weights_or_optimizer_state(
                optimizer_state_path, data_type="optimizer_state"
            )

            # Step 1: Validate optimizer state first
            own_optimizer_tensor, _, _ = flatten_optimizer_state(self.optimizer)
            passed, similarity, reason = await self.validate_optimizer_state(
                own_optimizer_tensor, miner_optimizer_state
            )

            # Log optimizer validation event
            similarity_score = similarity.item() if hasattr(similarity, "item") else float(similarity)
            self._log_validation_event(
                event_type="optimizer_validation", success=passed, score=similarity_score, reason=reason
            )

            logger.debug(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: VALIDATING OPTIMIZER STATE: {passed}, similarity: {similarity}, reason: {reason}"
            )
            if not passed:
                self.miner_weights[self.tracked_miner] -= PENALTY_RATE
                return False, similarity, "failed-optimizer-state"

            # Get validator and miner weights for further validation
            validator_weights = torch.nn.utils.parameters_to_vector(self.model.parameters()).flatten().to(DEVICE)
            miner_weights_flat = weights.flatten().to(DEVICE)

            # Step 2: Magnitude ratio check (gatekeeper)
            # R(v, m) = min(||v||, ||m||) / (max(||v||, ||m||) + ε)
            validator_norm = torch.norm(validator_weights)
            miner_norm = torch.norm(miner_weights_flat)

            eps = 1e-8
            magnitude_ratio = torch.min(validator_norm, miner_norm) / (torch.max(validator_norm, miner_norm) + eps)
            magnitude_ratio_value = magnitude_ratio.item()

            if magnitude_ratio_value < WEIGHT_MAGNITUDE_THRESHOLD:
                logger.warning(
                    f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: WEIGHT MAGNITUDE CHECK FAILED - "
                    f"ratio: {magnitude_ratio_value:.4f}, validator_norm: {validator_norm:.4f}, "
                    f"miner_norm: {miner_norm:.4f}"
                )

                # Log weight validation event for magnitude failure
                self._log_validation_event(
                    event_type="weight_validation",
                    success=False,
                    score=magnitude_ratio_value,
                    reason="weight-magnitude-ratio-failed",
                    validator_norm=validator_norm.item(),
                    miner_norm=miner_norm.item(),
                    magnitude_ratio=magnitude_ratio_value,
                )

                self.miner_weights[self.tracked_miner] -= PENALTY_RATE
                return False, magnitude_ratio, "weight-magnitude-ratio-failed"

            # Step 3: Cosine similarity check (final validation)
            similarity = torch.nn.functional.cosine_similarity(
                validator_weights.unsqueeze(0),
                miner_weights_flat.unsqueeze(0),
                dim=1,
            ).item()
            passed = similarity > COSINE_SIMILARITY_THRESHOLD
            reason = "passed" if passed else "failed"

            # Log weight validation event
            self._log_validation_event(
                event_type="weight_validation",
                success=passed,
                score=similarity,
                reason=reason,
                validator_norm=validator_norm.item(),
                miner_norm=miner_norm.item(),
                magnitude_ratio=magnitude_ratio_value,
            )

            logger.debug(
                f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: WEIGHT VALIDATION - "
                f"magnitude_ratio: {magnitude_ratio}, cosine_similarity: {similarity}, passed: {passed}"
            )

            if passed:
                self.miner_weights[self.tracked_miner] += 1
            else:
                self.miner_weights[self.tracked_miner] -= PENALTY_RATE

            return passed, similarity, reason
        else:
            logger.warning(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: VALIDATING WEIGHTS IN MOCK MODE")

            # Log weight validation event for mock mode
            self._log_validation_event(event_type="weight_validation", success=True, score=1.0, reason="mock-mode")

            self.miner_weights[self.tracked_miner] += 1
            return True, 1.0, "mock-mode"

    async def validate_optimizer_state(
        self, validator_optimizer_state: torch.Tensor, miner_optimizer_state: torch.Tensor
    ):
        similarity = torch.nn.functional.cosine_similarity(
            validator_optimizer_state.flatten().unsqueeze(0).to(DEVICE),
            miner_optimizer_state.flatten().unsqueeze(0).to(DEVICE),
        )
        passed = (similarity > 0.8).item()
        return passed, similarity, "passed" if passed else "failed"

    async def save_activations(
        self,
        validator_activations: torch.Tensor,
        miner_activations: torch.Tensor,
        layer: int,
        direction: str,
    ):
        """
        Saves the activations with metadata to local storage
        """
        # create directory if it doesn't exist
        os.makedirs("validated_activations", exist_ok=True)
        timestamp = time.time()
        # Create dictionary with metadata and activations
        activations_dict = {
            "layer": layer,
            "direction": direction,
            "validator_activations": (
                validator_activations.tolist()
                if isinstance(validator_activations, torch.Tensor)
                else validator_activations
            ),
            "miner_activations": (
                miner_activations.tolist() if isinstance(miner_activations, torch.Tensor) else miner_activations
            ),
            "timestamp": timestamp,
            "uid": self.tracked_miner,
        }

        # Save the activations to local storage
        activations_path = os.path.join("validated_activations", f"activations_{timestamp}_{layer}_{direction}.json")
        with open(activations_path, "w") as f:
            json.dump(activations_dict, f)

    async def reset_validator(self):
        """Enhanced reset that clears metagraph-related state."""
        logger.debug("GRADIENT VALIDATOR: RESET VALIDATOR")
        try:
            # Save validation events to file before resetting
            await self._save_validation_events()

            self.api_client = APIClient(self.wallet)
            await self.api_client.__aenter__()

            # The orchestrator tracks all the miner weights in uid space, but in the validator, we use self.tracked_miner which is a hotkey.
            hotkey_to_uid = {hotkey: str(uid) for hotkey, uid in zip(self.metagraph.hotkeys, self.metagraph.uids)}
            miner_weights: dict[str, float] = {
                hotkey_to_uid[hotkey]: float(weight) for hotkey, weight in self.miner_weights.items()
            }

            await self.api_client.submit_miner_weights(weights=miner_weights)
            self.available = True
            self.tracked_miner = None
            self.layer = None
            self.weights = None
            self.saved_forward_activations = {}
            self.processed_forward_activations = []
            self.processed_backward_activations = []
            self.miner_weights = {}
            self.model = None
            self.optimizer = None
            # Clear validation events after saving
            self.validation_events = []
        except Exception as e:
            logger.exception(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: Error resetting validator: {e}")

    def get_status(self) -> dict:
        """Get current status of the gradient validator including metagraph info."""
        status = {
            "available": self.available,
            "tracked_miner": self.tracked_miner,
            "layer": getattr(self, "layer", None),
            "weight_version": self.weight_version,
            "uid": self.hotkey,
            "netuid": self.netuid,
        }

        if self.tracked_miner is not None and self.metagraph:
            status["miner_info"] = self.get_miner_info(self.tracked_miner)

        return status

    async def weight_loop(self):
        register_validator = True
        while True:
            try:
                logger.debug(f"GRADIENT VALIDATOR [MINER {self.tracked_miner}]: WEIGHT LOOP RUNNING")
                if await self.api_client.health_check():
                    if register_validator:
                        logger.debug(f"Registering validator {self.wallet.hotkey.ss58_address[:8]} with orchestrator")
                        await self.api_client.register_validator(
                            host=self.external_ip,
                            port=int(settings.VALIDATOR_EXTERNAL_PORT),
                            scheme=settings.VALIDATOR_SCHEME,
                        )
                        register_validator = False

                    # Get global weights from API
                    global_weights: dict[int, float] = await self.api_client.get_global_miner_weights()
                    global_weights = {int(uid): weight for uid, weight in global_weights.items()}
                else:
                    register_validator = True
                    logger.warning("Orchestrator is not healthy, skipping weight submission")
                    global_weights = {}

                # Submit global weights to Bittensor
                if len(global_weights) > 0:
                    logger.debug(f"Received global weights: {global_weights}")
                    self.set_weights(weights=global_weights)
                else:
                    logger.warning("No global weights received, temporarily copying weights from the chain")
                    self.set_weights(weights=self.copy_weights_from_chain())

            except Exception as e:
                logger.exception(f"Error in weight loop: {e}")
            finally:
                await asyncio.sleep(settings.WEIGHT_SUBMIT_INTERVAL)

    def set_weights(self, weights: dict[int, float]):
        """
        Sets the validator weights to the metagraph hotkeys based on the global weights.
        """
        logger.info("Attempting to set weights to Bittensor.")
        if not BITTENSOR:
            logger.warning("Bittensor is not enabled, skipping weight submission")
            return

        # TODO REMOVE WHEN WE MERGE IT!
        if not hasattr(self, "wallet") or not self.wallet:
            logger.warning("Wallet not initialized, skipping weight submission")
            return

        if not hasattr(self, "subtensor") or not self.subtensor:
            logger.warning("Subtensor not initialized, skipping weight submission")
            return

        if not hasattr(self, "metagraph") or not self.metagraph:
            logger.warning("Metagraph not initialized, skipping weight submission")
            return

        try:
            # Convert global weights to tensor, Global state of scores is on the orchestrator
            scores = torch.zeros(len(self.metagraph.uids), dtype=torch.float32)
            for uid, weight in weights.items():
                scores[uid] = weight

            # Check if scores contains any NaN values
            if torch.isnan(scores).any():
                logger.warning("Scores contain NaN values. Replacing with 0.")
                scores = torch.nan_to_num(scores, 0)

            # Check if we have any non-zero scores
            if torch.sum(scores) == 0:
                logger.warning("All scores are zero, skipping weight submission")
                return

            # Normalize weights
            raw_weights = torch.nn.functional.normalize(scores, p=1, dim=0)

            if settings.BURN_FACTOR > 1 and settings.netuid == 9 and raw_weights[209] < (1 - 1 / settings.BURN_FACTOR):
                # Divide the raw_weights by settings.burn_factor before further processing
                raw_weights = raw_weights / settings.BURN_FACTOR

                # Add the 1-1/burn factor to the 209th uid
                if len(raw_weights) > 209:  # 209 is the owner hotkey of sn9
                    raw_weights[209] = 1 - 1 / settings.BURN_FACTOR

            # Process the raw weights to final_weights via subtensor limitations
            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=self.metagraph.uids,
                weights=raw_weights.detach().cpu().float().numpy(force=True).astype(np.float32),
                netuid=int(settings.netuid),
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            # Log the weights being set
            weight_dict = dict(zip(processed_weight_uids.tolist(), processed_weights.tolist()))
            logger.info(f"Setting weights for {len(weight_dict)} miners")
            logger.debug(f"Weight details: {weight_dict}")

            # Submit weights to Bittensor chain
            success, response = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=int(settings.netuid),
                uids=processed_weight_uids,
                weights=processed_weights,
                wait_for_finalization=False,
                version_key=settings.__spec_version__,
            )

            if success:
                logger.success("Successfully submitted weights to Bittensor.")
                logger.debug(f"Response: {response}")
            else:
                logger.error("Failed to submit weights to Bittensor")
                logger.error(f"Response: {response}")

        except Exception as e:
            logger.exception(f"Error submitting weights to Bittensor: {e}")

    def copy_weights_from_chain(self) -> dict[int, float]:
        """Copy weights from the chain to the validator.

        Returns:
            dict[int, float]: A dictionary of weights for each miner.
        """
        meta = self.subtensor.metagraph(netuid=int(settings.netuid), lite=False)
        valid_indices = np.where(meta.validator_permit)[0]
        valid_weights = meta.weights[valid_indices]
        valid_stakes = meta.S[valid_indices]
        normalized_stakes = valid_stakes / np.sum(valid_stakes)
        stake_weighted_average = np.dot(normalized_stakes, valid_weights).astype(float).tolist()
        return dict(zip(meta.uids, list(stake_weighted_average)))

    async def start_weight_submission_task(self):
        logger.debug("STARTING WEIGHT SUBMISSION TASK")
        asyncio.create_task(self.weight_loop())