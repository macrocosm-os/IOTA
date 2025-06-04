import asyncio
import json
import math
import time
from typing import Callable, Optional

import httpx
from loguru import logger

import settings


class DashboardMetricsReporter:
    def __init__(self):
        self.base_url = settings.DASHBOARD_BASE_URL
        self.last_loss_report = 0
        self.last_miner_report = 0
        self.activation_count = 0  # Per-merge activation count (for sample size tracking)
        self.total_activation_count = 0  # Cumulative total that never resets
        self.sampled_activations = []
        self.client = httpx.AsyncClient(timeout=10.0)
        self.env = getattr(settings, "DASHBOARD_ENV", "prod")
        self.access_key = getattr(settings, "DASHBOARD_ACCESS_KEY", None)

        # Loss reporting buffers
        self.loss_buffer = []
        self.activation_count_buffer = []

        # Flag to track if periodic reporting is initialized
        self._periodic_report_initialized = False

        # Callback for getting miner data
        self._get_miner_data_callback: Optional[Callable] = None

    def set_miner_data_callback(self, callback: Callable):
        """Set the callback function to get miner data."""
        self._get_miner_data_callback = callback

    async def initialize(self):
        """Initialize the dashboard metrics reporter and start periodic reporting."""
        if not self._periodic_report_initialized:
            asyncio.create_task(self._periodic_miner_report())
            self._periodic_report_initialized = True
            if settings.DASHBOARD_LOGS:
                logger.info("Dashboard metrics reporter initialized with periodic reporting")

    async def _periodic_miner_report(self):
        """Periodically report all miner statuses to the dashboard."""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_miner_report >= settings.MINER_REPORT_INTERVAL:
                    if settings.DASHBOARD_LOGS:
                        logger.info("Sending periodic miner status report to dashboard")
                    # Get all miners using the callback
                    if self._get_miner_data_callback is None:
                        if settings.DASHBOARD_LOGS:
                            logger.warning("No miner data callback set, skipping periodic report")
                        await asyncio.sleep(60)
                        continue
                    data = self._get_miner_data_callback()
                    miners = data.get("miners", {})
                    metagraph = data.get("metagraph")
                    if not miners:
                        if settings.DASHBOARD_LOGS:
                            logger.debug("No miners available for reporting")
                        await asyncio.sleep(60)
                        continue
                    if settings.DASHBOARD_LOGS:
                        logger.debug(f"Reporting for {len(miners)} miners")
                    for miner in miners.values():
                        if miner.layer is not None:
                            time_since_last_report = current_time - miner.last_throughput_report
                            throughput = (
                                miner.processed_activations / (time_since_last_report / 60)
                                if time_since_last_report > 0
                                else 0
                            )
                            # Gather metagraph info if available
                            if settings.BITTENSOR and metagraph:
                                try:
                                    uid = metagraph.hotkeys.index(miner.hotkey)
                                    coldkey = metagraph.coldkeys[uid]
                                    hotkey = metagraph.hotkeys[uid]
                                    incentive = float(metagraph.incentive[uid])
                                except (ValueError, AttributeError) as e:
                                    if settings.DASHBOARD_LOGS:
                                        logger.warning(f"Error getting metagraph info for miner {miner.hotkey}: {e}")
                                    uid = abs(hash(miner.hotkey)) % (10**8)
                                    coldkey = f"dummy-coldkey-{miner.hotkey}"
                                    hotkey = miner.hotkey
                                    incentive = 0.0
                            else:
                                # For mock mode.
                                uid = abs(hash(miner.hotkey)) % (10**8)
                                coldkey = f"dummy-coldkey-{miner.hotkey}"
                                hotkey = miner.hotkey
                                incentive = 0.0

                            # Calculate if miner is active based on last S3 upload
                            is_active = bool(
                                miner.last_s3_upload
                                and current_time - miner.last_s3_upload <= settings.MINER_ACTIVITY_TIMEOUT
                            )

                            await self.report_miner_status(
                                miner_uid=uid,
                                layer=miner.layer,
                                processed_activations=miner.processed_activations,
                                throughput=throughput,
                                registration_time=miner.registration_time,
                                coldkey=coldkey,
                                hotkey=hotkey,
                                incentive=incentive,
                                is_active=is_active,
                            )
                    self.last_miner_report = current_time
            except Exception as e:
                if settings.DASHBOARD_LOGS:
                    logger.error(f"Error in periodic miner report: {e}")
            await asyncio.sleep(60)  # Check every minute

    async def _make_request(self, endpoint: str, payload: dict, max_retries: int = settings.MAX_RETRIES) -> bool:
        """Make a POST request to the dashboard with retry logic."""
        if not self.base_url:
            if settings.DASHBOARD_LOGS:
                logger.warning(f"Dashboard reporting is disabled (no base_url). Skipping request to {endpoint}.")
            return False
        if not self.access_key:
            if settings.DASHBOARD_LOGS:
                logger.warning(f"Dashboard reporting is disabled (no access_key). Skipping request to {endpoint}.")
            return False
        for attempt in range(max_retries):
            try:
                # Ensure all numeric values are properly formatted
                formatted_payload = self._format_payload(payload)
                # Debug log: print payload before sending
                if settings.DASHBOARD_LOGS:
                    logger.debug(
                        f"[DASHBOARD SEND] Attempt {attempt+1}/{max_retries} to endpoint '{endpoint}' with payload: {json.dumps(formatted_payload)}"
                    )
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.access_key}"}
                if not (self.env == "prod" or self.env == "staging"):
                    if settings.DASHBOARD_LOGS:
                        logger.error(f"Invalid environment '{self.env}'. Must be 'prod' or 'staging'.")
                    return False
                response = await self.client.post(
                    f"{self.base_url}/{endpoint}",
                    json=formatted_payload,
                    headers=headers,
                )
                if response.status_code == 401:
                    if settings.DASHBOARD_LOGS:
                        logger.error(f"Authentication failed for {endpoint}. Please check your DASHBOARD_ACCESS_KEY.")
                    return False
                if response.status_code == 400:
                    if settings.DASHBOARD_LOGS:
                        logger.error(f"Invalid payload format for {endpoint}: {formatted_payload}")
                        logger.error(f"Response: {response.text}")
                    return False
                response.raise_for_status()
                if settings.DASHBOARD_LOGS:
                    logger.debug(f"[DASHBOARD SEND] Success: {endpoint} status {response.status_code}")
                return True
            except httpx.HTTPStatusError as e:
                if settings.DASHBOARD_LOGS:
                    logger.error(f"HTTP error reporting to {endpoint} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(settings.RETRY_DELAY)
            except httpx.RequestError as e:
                if settings.DASHBOARD_LOGS:
                    logger.error(
                        f"Request error reporting to {endpoint} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                if attempt < max_retries - 1:
                    await asyncio.sleep(settings.RETRY_DELAY)
            except Exception as e:
                if settings.DASHBOARD_LOGS:
                    logger.error(
                        f"Unexpected error reporting to {endpoint} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                if attempt < max_retries - 1:
                    await asyncio.sleep(settings.RETRY_DELAY)
        if settings.DASHBOARD_LOGS:
            logger.error(
                f"[DASHBOARD SEND] All attempts failed for endpoint '{endpoint}' with payload: {json.dumps(payload)}"
            )
        return False

    def _format_payload(self, payload: dict) -> dict:
        """Format payload values to ensure they match API expectations."""
        formatted = {}
        for key, value in payload.items():
            if isinstance(value, float):
                # Round floats to 6 decimal places to avoid precision issues
                formatted[key] = round(value, 6)
            elif isinstance(value, (int, str, bool)):
                formatted[key] = value
            else:
                # Convert any other types to string
                formatted[key] = str(value)
        return formatted

    async def buffer_loss(self, loss: float, number_of_sampled_activations: int) -> None:
        """Buffer a loss for later reporting to the dashboard.

        Losses are now aggregated between weight merges and sent as a single datapoint
        when send_aggregated_loss() is called.

        Args:
            loss: The loss value to buffer
            number_of_sampled_activations: Number of activations this loss represents
        """
        # Validate and sanitize the loss value before buffering
        if math.isnan(loss):
            logger.warning("Received NaN loss value, skipping")
            return
        elif math.isinf(loss):
            logger.warning(f"Received infinite loss value ({loss}), skipping")
            return

        # Add new data point to buffers
        self.loss_buffer.append(float(loss))
        self.activation_count += int(number_of_sampled_activations)  # Accumulate for this merge period
        self.total_activation_count += int(number_of_sampled_activations)  # Cumulative total

    async def send_aggregated_loss(self) -> None:
        """Send aggregated loss data to dashboard.

        This should be called when weight merges occur to send the average loss
        accumulated since the last weight merge.
        """
        if not self.loss_buffer:
            logger.debug("No loss data to send to dashboard")
            return

        current_time = time.time()  # Get current timestamp

        # Calculate averages for the period between weight merges
        avg_loss = sum(self.loss_buffer) / len(self.loss_buffer)

        # Calculate perplexity safely
        try:
            avg_perplexity = math.exp(avg_loss)
            # Check if perplexity calculation resulted in invalid values
            if math.isnan(avg_perplexity) or math.isinf(avg_perplexity):
                logger.error(f"Perplexity calculation resulted in invalid value for avg_loss {avg_loss}")
                raise ValueError("Invalid perplexity value")
        except OverflowError:
            logger.error(f"Perplexity calculation overflow for avg_loss {avg_loss}")
            raise

        sample_size = len(self.loss_buffer)  # Number of samples in this period

        # Prepare and send the aggregated payload
        payload = {
            "timestamp": int(current_time),
            "loss": float(avg_loss),
            "perplexity": float(avg_perplexity),
            "activation_count": int(self.total_activation_count),  # Use cumulative total
            "sample_size": int(sample_size),
        }

        success = await self._make_request("loss", payload)
        if success:
            self.last_loss_report = current_time
            # Clear buffers after successful report
            self.loss_buffer = []
            # Reset per-merge activation count (but keep total_activation_count)
            self.activation_count = 0
            if settings.DASHBOARD_LOGS:
                logger.info(
                    f"Sent aggregated loss to dashboard: avg_loss={avg_loss:.4f}, avg_perplexity={avg_perplexity:.4f}, sample_size={sample_size}, total_activations={self.total_activation_count}"
                )
        else:
            logger.warning("Failed to send aggregated loss to dashboard")

    async def report_loss(self, loss: float, number_of_sampled_activations: int) -> None:
        """Report loss metrics to the dashboard. This is now just a wrapper around buffer_loss."""
        await self.buffer_loss(loss, number_of_sampled_activations)

    async def report_miner_status(
        self,
        miner_uid: int,
        layer: int,
        processed_activations: int,
        throughput: float,
        registration_time: int,
        coldkey: str = None,
        hotkey: str = None,
        incentive: float = None,
        is_active: bool = False,
    ) -> None:
        """Report miner status to the dashboard."""
        current_time = time.time()
        if settings.DASHBOARD_LOGS:
            logger.debug(f"Reporting miner {miner_uid} (layer {layer}) status at timestamp: {current_time:.2f}")
        # Prepare payload in the format expected by the dashboard
        payload = {
            "timestamp": int(current_time),
            "layer": int(layer) if layer is not None else 0,
            "miner_uid": int(miner_uid),
            "unique_miner_id": str(f"miner-{miner_uid}"),
            "coldkey": str(coldkey) if coldkey is not None else f"coldkey-{miner_uid}",
            "hotkey": str(hotkey) if hotkey is not None else f"hotkey-{miner_uid}",
            "activation_count": int(processed_activations),
            "ip_address": "127.0.0.1",
            "throughput": int(round(throughput * 60)),
            "registration_time": int(registration_time),
            "incentive": float(incentive) if incentive is not None else 0.0,
            "is_active": bool(is_active),
        }
        if settings.DASHBOARD_LOGS:
            logger.debug(f"Sending miner status payload: {json.dumps(payload)}")
        success = await self._make_request("miner", payload)
        if success:
            self.last_miner_report = current_time

    async def report_sync(self) -> None:
        """Report model sync event to the dashboard."""
        payload = {"sync_timestamp": int(time.time())}
        await self._make_request("sync", payload)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
