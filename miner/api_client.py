import aiohttp
from loguru import logger
from typing import Dict, Any, Optional
import settings
import asyncio
import sys
from aiohttp import ClientConnectorError, ContentTypeError
from utils.epistula import generate_header, create_message_body
from storage.serializers import (
    ActivationDownloadRequest,
    ActivationResponse,
    ActivationUploadRequest,
    StorageResponse,
    PresignedUrlRequest,
    MultipartUploadRequest,
    CompleteMultipartUploadRequest,
)
from orchestrator.serializers import (
    LossReportRequest,
    LossReportResponse,
    MinerRegistrationResponse,
    LayerAssignmentResponse,
)
from utils.partitions import Partition


class APIClient:
    def __init__(self, wallet=None, orchestrator_time: str = None):
        self.base_url = f"{settings.ORCHESTRATOR_SCHEME}://{settings.ORCHESTRATOR_HOST}:{settings.ORCHESTRATOR_PORT}"
        self.session = None
        self.max_retries = 3
        self.retry_delay = 3.0  # seconds
        self.wallet = wallet
        self.failed_api_request = False
        self.orchestrator_time = orchestrator_time

    async def __aenter__(self):
        ssl = False if settings.ORCHESTRATOR_SCHEME == "http" else True
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def submit_miner_weights(self, weights: dict[str, float]) -> Dict[str, Any]:
        try:
            if len(weights) == 0:
                logger.warning("No weights to submit")
                return
            return await self._make_request(
                "post",
                f"{self.base_url}/orchestrator/submit_miner_weights",
                json=weights,
            )
        except Exception as e:
            logger.exception(f"Failed to submit miner weights: {e}")
            return None

    async def get_global_miner_weights(self) -> dict[str, Any]:
        return await self._make_request("get", f"{self.base_url}/orchestrator/global_weights_of_miners")

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        if not self.wallet or not self.wallet.hotkey:
            raise ValueError("Wallet and hotkey must be set for API requests")
        # Create message body for signing
        body = kwargs.get("json", {})
        body_bytes = create_message_body(body)
        # Generate Epistula headers
        headers = generate_header(self.wallet.hotkey, body_bytes)

        # Add orchestrator version header if available
        if self.orchestrator_time:
            headers["X-Orchestrator-Version"] = self.orchestrator_time

        # Add headers to request
        if "headers" not in kwargs:
            kwargs["headers"] = {}
        kwargs["headers"].update(headers)

        for attempt in range(self.max_retries):
            try:
                async with getattr(self.session, method)(url, **kwargs) as response:
                    if response.status >= 500:
                        logger.warning(
                            f"Server error {response.status} on attempt {attempt + 1} for url {url} and params {kwargs}"
                        )
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue

                    # Handle version mismatch error
                    if response.status == 409:
                        response_json = await response.json()
                        error_detail = response_json.get("detail", "")
                        if "version mismatch" in error_detail.lower():
                            logger.error(f"❌ Critical error: Orchestrator version mismatch detected: {error_detail}")
                            logger.error("🔄 Shutting down miner to force restart with correct version...")
                            sys.exit(1)  # Exit with error code 1 to indicate abnormal termination
                        logger.warning(
                            f"Miner attempted to make a request ({method} {url}) in the wrong state: {error_detail}"
                        )
                        return {"expected_state": error_detail}

                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_text}")
                        raise aiohttp.ClientError(f"Request failed with status {response.status}: {error_text}")

                    try:
                        return await response.json()
                    except ContentTypeError:
                        error_text = await response.text()
                        logger.error(f"Failed to parse JSON response: {error_text}")
                        raise aiohttp.ClientError(f"Failed to parse JSON response: {error_text}")

            except ClientConnectorError as e:
                if url == f"{self.base_url}/orchestrator/healthcheck":
                    raise
                logger.warning(
                    f"Could not connect to orchestrator for request {method} {url} and params {kwargs}, likely it is down"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                self.failed_api_request = True
                raise
            except aiohttp.ClientResponseError as e:
                if e.status == 429:  # Rate limit status code
                    logger.warning(f"Rate limited on request {method} {url}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise
            except Exception as e:
                logger.warning(
                    f"Request for method {method} url {url} and params {kwargs} failed on attempt {attempt + 1}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                self.failed_api_request = True
                raise

        raise aiohttp.ClientError(f"Failed after {self.max_retries} attempts")

    async def register(self) -> tuple[int, str]:
        logger.info(f"🔗 API: Registering miner {self.wallet.hotkey.ss58_address[:8]}")
        response = await self._make_request("post", f"{self.base_url}/orchestrator/register", json={})

        try:
            response = MinerRegistrationResponse(**response)
            layer = response.layer
            orchestrator_time = response.version
            # Store the orchestrator version for future requests
            self.orchestrator_time = orchestrator_time
            logger.info(f"✅ API: Registration successful | Layer: {layer}")
            return layer, orchestrator_time
        except Exception as e:
            logger.info(response)
            logger.exception(f"❌ API: Failed to register miner: {e}")
            raise

    async def update_status(
        self, status: str, activation_uid: Optional[str] = None, activation_path: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.debug(
            f"📤 API: Updating status to '{status}' | Activation: {activation_uid if activation_uid else 'N/A'}"
        )
        data = {"status": status, "activation_path": activation_path}
        if activation_uid:
            data["activation_uid"] = activation_uid

        response = await self._make_request("post", f"{self.base_url}/orchestrator/miners/status", json=data)
        return response

    async def request_layer(self) -> LayerAssignmentResponse:
        logger.info("🔗 API: Requesting layer assignment")
        response = await self._make_request("post", f"{self.base_url}/orchestrator/miners/request_layer")
        result = LayerAssignmentResponse(**response)
        logger.info(f"✅ API: Layer assignment received | Layer: {result.layer}")
        return result

    async def report_loss(self, activation_uid: str, loss: float) -> LossReportResponse:
        logger.debug(f"📊 API: Reporting loss {loss:.6f} for activation {activation_uid}")
        data = LossReportRequest(activation_uid=activation_uid, loss_value=loss)
        response = await self._make_request(
            "post",
            f"{self.base_url}/orchestrator/miners/report_loss",
            json=data.model_dump(),
        )
        logger.debug("✅ API: Loss reported successfully")
        return LossReportResponse(**response)

    async def notify_weights_uploaded(
        self,
        weights_path: str,
        metadata_path: str,
        optimizer_state_path: str,
        optimizer_state_metadata_path: str,
    ) -> Dict[str, Any]:
        logger.info("📤 API: Notifying weights uploaded")
        response = await self._make_request(
            "post",
            f"{self.base_url}/orchestrator/miners/notify_weights_uploaded?weights_path={weights_path}&metadata_path={metadata_path}&optimizer_state_path={optimizer_state_path}&optimizer_state_metadata_path={optimizer_state_metadata_path}",
        )
        logger.info("✅ API: Weights upload notification sent")
        return response

    async def upload_activation_to_orchestrator(
        self, activation_uid: str, layer: int, direction: str, activation_path: str
    ) -> StorageResponse:
        if layer < 0:
            raise ValueError(f"Layer {layer} is not valid")
        data = ActivationUploadRequest(
            activation_uid=activation_uid,
            layer=layer,
            direction=direction,
            activation_path=activation_path,
        )
        response = await self._make_request(
            "post",
            f"{self.base_url}/storage/activations/upload",
            json=data.model_dump(),
        )
        return StorageResponse(**response)

    async def download_activation_from_orchestrator(
        self,
        activation_uid: str,
        direction: str,
        layer: int | None = None,
        delete: bool = True,
        fetch_historic: bool = False,
    ) -> StorageResponse:
        data = ActivationDownloadRequest(
            activation_uid=activation_uid,
            direction=direction,
            delete=delete,
            layer=layer,
            fetch_historic=fetch_historic,
        )
        response = await self._make_request(
            "post",
            f"{self.base_url}/storage/activations/download",
            json=data.model_dump(),
        )
        return StorageResponse(**response)

    async def get_random_activation(
        self,
    ) -> ActivationResponse:
        """Get a random activation without seeing the full list."""
        logger.debug("🎲 API: Getting random activation")
        response = await self._make_request("post", f"{self.base_url}/storage/activations/random")
        logger.debug(f"Got random activation: {response}")

        # If the request contains an expected state error, return the response
        if response.get("expected_state"):
            return response

        result = ActivationResponse(**response)
        if result.activation_uid:
            logger.debug(
                f"✅ API: Random activation received | UID: {result.activation_uid} | Direction: {result.direction}"
            )
        else:
            logger.debug(f"⏸️ API: No activations available | Reason: {result.reason}")
        return result

    async def get_layer_weights(self, layer: int) -> str:
        logger.info(f"📥 API: Getting weights for layer {layer}")
        response = await self._make_request("get", f"{self.base_url}/storage/weights/layer/{layer}")
        partitions = [Partition(**partition) for partition in response]
        logger.info(f"✅ API: Layer weights received | Partitions: {len(partitions)}")
        return partitions

    async def merge_info(self, layer: int) -> dict[str, Any]:
        """Check if the system is currently in merging phase.
        Returns:
            dict[str, Any]: Dictionary containing the status and num_sections.
        """
        logger.debug(f"🔍 API: Checking merge info for layer {layer}")
        response = await self._make_request("get", f"{self.base_url}/orchestrator/is_merging?layer={layer}")
        logger.debug(f"✅ API: Merge info received | Received status: {response.get('status', 'unknown')}")
        return response

    async def is_activation_active(self, layer: int, activation_uid: int) -> bool:
        """Check if an activation is still needed by checking if it exists as a backward activation in any higher layer."""
        response = await self._make_request(
            "get",
            f"{self.base_url}/storage/activations/is_active?layer={layer}&activation_uid={activation_uid}",
        )
        return response["is_active"]

    async def get_presigned_url(
        self,
        path: str,
        expires_in: int = 3600,
    ) -> dict[str, str]:
        """Get a presigned URL for S3 operations."""
        if not settings.USE_S3:
            return {
                "url": f"activations/{path}",
                "fields": None,
            }

        data = PresignedUrlRequest(
            path=path,
            expires_in=expires_in,
        )
        response = await self._make_request(
            method="get",
            url=f"{self.base_url}/storage/presigned_url",
            json=data.model_dump(),
        )
        return response["data"]["presigned_data"]

    async def register_validator(self, host: str, port: int, scheme: str = "http") -> Dict[str, Any]:
        """Register a validator with the orchestrator."""
        response = await self._make_request(
            "post",
            f"{self.base_url}/orchestrator/register_validator?host={host}&port={port}&scheme={scheme}",
        )
        self.orchestrator_time = str(response.get("version"))
        return response

    async def weight_partition_info(self):
        """Get the weight partition info for a given layer."""
        logger.info("📥 API: Getting weight partition info")
        response = await self._make_request("get", f"{self.base_url}/orchestrator/get_chunks_for_miner")
        logger.info("✅ API: Weight partition info received")
        return response

    async def notify_merged_partitions_uploaded(self, partitions: list[Partition]):
        logger.info(f"📤 API: Notifying merged partitions uploaded | Count: {len(partitions)}")
        response = await self._make_request(
            "post",
            f"{self.base_url}/orchestrator/miners/notify_merged_partitions_uploaded",
            json=[partition.model_dump() for partition in partitions],
        )
        logger.info("✅ API: Merged partitions notification sent")
        return response

    async def initiate_multipart_upload(
        self,
        path: str,
        file_size: int,
        part_size: int = 100 * 1024 * 1024,
        expires_in: int = 3600,
    ) -> dict[str, Any]:
        """Initiate a multipart upload and get presigned URLs for parts."""
        data = MultipartUploadRequest(
            path=path,
            file_size=file_size,
            part_size=part_size,
            expires_in=expires_in,
        )
        response = await self._make_request(
            method="post",
            url=f"{self.base_url}/storage/multipart_upload/initiate",
            json=data.model_dump(),
        )
        return response["data"]

    async def complete_multipart_upload(
        self,
        path: str,
        upload_id: str,
        parts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Complete a multipart upload."""
        data = CompleteMultipartUploadRequest(
            path=path,
            upload_id=upload_id,
            parts=parts,
        )
        response = await self._make_request(
            method="post",
            url=f"{self.base_url}/storage/multipart_upload/complete",
            json=data.model_dump(),
        )
        return response["data"]

    async def health_check(self) -> bool:
        try:
            logger.debug("🏥 API: Health check")
            response = await self._make_request(
                method="get",
                url=f"{self.base_url}/orchestrator/healthcheck",
            )
            logger.debug(f"🏥 API: Health check response: {response}")
            if response.get("status") != "healthy":
                logger.warning("❌ API: Orchestrator failed health check!")
                return False

            logger.debug("✅ API: Health check passed")
            return True

        except Exception as e:
            logger.warning(f"❌ API: Health check failed: {e}")
            return False

    async def is_registered(self) -> bool:
        """Check if the entity is registered with the orchestrator."""
        response = await self._make_request("get", f"{self.base_url}/orchestrator/is_registered")
        return response["is_registered"]
