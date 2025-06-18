import os
import io
import json
import torch
import boto3
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import settings
from pathlib import Path
import requests
from botocore.client import Config
from botocore import UNSIGNED
from typing import Any, Literal
import asyncio
import aiohttp
from utils.partitions import Partition
from utils.vector_utils import check_for_nans


# Load environment variables
load_dotenv()

# Initialize S3 client
s3_client = None


def initialize_s3():
    """Initialize the S3 client with credentials"""
    global s3_client

    if not settings.USE_S3:
        logger.info("S3 storage disabled, using local file system")
        return

    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        logger.warning("AWS credentials not found. Using default credentials.")
        s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    else:
        s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
        )

    logger.info(f"S3 client initialized with bucket: {settings.S3_BUCKET}")


def normalize_s3_path(path: str) -> tuple[str, str]:
    """Fix any s3:/bucket/path errors to s3://bucket/path"""
    bucket = settings.S3_BUCKET
    if path.startswith("s3:/") and not path.startswith("s3://"):
        path = path.replace("s3:/", "s3://", 1)

    if path.startswith("s3://"):
        parts = path[5:].split("/", 1)
        if len(parts) != 2:
            logger.error(f"Invalid S3 path format: {path}")
            raise ValueError(f"Invalid S3 path: {path}")
        bucket_from_path, path = parts

        # Ensure bucket is not None or empty, use the provided bucket or default
        if bucket_from_path:
            bucket = bucket_from_path
        elif not bucket:
            logger.error(f"No bucket specified in path or settings: {path}")
            raise ValueError(f"No bucket specified in path or settings: {path}")

    return bucket, path


def generate_presigned_url(
    path: str,
    expires_in: int = 3600,
) -> dict[str, Any]:
    assert path is not None, "path must be provided to generate_presigned_url"

    if not settings.USE_S3 or not s3_client:
        # Return local path format for non-S3 usage
        return {"url": path, "fields": {}}

    try:
        response = s3_client.generate_presigned_post(
            Bucket=settings.S3_BUCKET,
            Key=path,
            ExpiresIn=expires_in,
        )
        return response
    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        raise


def upload_to_bucket(presigned_data: dict[str, Any], files: dict[str, Any]) -> str:
    response = requests.post(presigned_data["url"], data=presigned_data["fields"], files=files)

    if response.status_code != 204 and response.status_code != 200:
        logger.error(f"Failed to upload file to S3: {response.status_code} {response.text}")
        response.raise_for_status()

    # Get the S3 path from the presigned URL
    key = presigned_data["fields"]["key"]
    logger.debug(f"Uploaded to S3: {key}")
    return f"s3://{settings.S3_BUCKET}/{key}"


def create_multipart_upload(path: str) -> str:
    """Create a multipart upload and return the upload ID."""
    if not settings.USE_S3 or not s3_client:
        raise ValueError("S3 is not enabled or client not initialized")

    try:
        response = s3_client.create_multipart_upload(
            Bucket=settings.S3_BUCKET,
            Key=path,
        )
        upload_id = response["UploadId"]
        logger.debug(f"Created multipart upload {upload_id} for {path}")
        return upload_id
    except ClientError as e:
        logger.error(f"Error creating multipart upload: {e}")
        raise


def generate_presigned_url_for_part(path: str, upload_id: str, part_number: int, expires_in: int = 3600) -> str:
    """Generate a presigned URL for uploading a specific part."""
    if not settings.USE_S3 or not s3_client:
        raise ValueError("S3 is not enabled or client not initialized")

    try:
        response = s3_client.generate_presigned_url(
            ClientMethod="upload_part",
            Params={
                "Bucket": settings.S3_BUCKET,
                "Key": path,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=expires_in,
        )
        logger.debug(f"Generated presigned URL for part {part_number} of upload {upload_id}")
        return response
    except ClientError as e:
        logger.error(f"Error generating presigned URL for part: {e}")
        raise


def upload_part_to_s3(presigned_url: str, data: bytes) -> str:
    """Upload a part using the presigned URL and return the ETag."""
    try:
        response = requests.put(presigned_url, data=data)
        response.raise_for_status()

        # Extract ETag from response headers (remove quotes if present)
        etag = response.headers.get("ETag", "").strip('"')
        logger.debug(f"Uploaded part with ETag: {etag}")
        return etag
    except Exception as e:
        logger.error(f"Error uploading part: {e}")
        raise


def complete_multipart_upload(path: str, upload_id: str, parts: list[dict[str, Any]]) -> str:
    """Complete the multipart upload."""
    if not settings.USE_S3 or not s3_client:
        raise ValueError("S3 is not enabled or client not initialized")

    try:
        response = s3_client.complete_multipart_upload(
            Bucket=settings.S3_BUCKET,
            Key=path,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        logger.debug(f"Completed multipart upload for {path}")
        return f"s3://{settings.S3_BUCKET}/{path}"
    except ClientError as e:
        logger.error(f"Error completing multipart upload: {e}")
        raise


def abort_multipart_upload(path: str, upload_id: str):
    """Abort a multipart upload in case of failure."""
    if not settings.USE_S3 or not s3_client:
        return

    try:
        s3_client.abort_multipart_upload(
            Bucket=settings.S3_BUCKET,
            Key=path,
            UploadId=upload_id,
        )
        logger.debug(f"Aborted multipart upload {upload_id} for {path}")
    except ClientError as e:
        logger.warning(f"Error aborting multipart upload: {e}")


def upload_large_file_multipart(data: bytes, path: str, part_size: int = 100 * 1024 * 1024) -> str:
    """
    Upload a large file using multipart upload with presigned URLs.

    Args:
        data: The file data as bytes
        path: The S3 key path
        part_size: Size of each part in bytes (default 100MB)

    Returns:
        The S3 path of the uploaded file
    """
    if not settings.USE_S3 or not s3_client:
        # For local storage, just write the file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    # If file is small enough, use regular upload
    if len(data) <= part_size:
        logger.debug(f"File size {len(data)} bytes is small enough for single-part upload")
        buffer = io.BytesIO(data)
        presigned_data = generate_presigned_url(path=path)
        return upload_to_bucket(presigned_data, {"file": ("data", buffer)})

    logger.info(f"Starting multipart upload for {path} ({len(data)} bytes, {part_size} bytes per part)")

    upload_id = None
    try:
        # Create multipart upload
        upload_id = create_multipart_upload(path)

        # Calculate number of parts needed
        total_parts = (len(data) + part_size - 1) // part_size
        parts = []

        # Upload each part
        for part_number in range(1, total_parts + 1):
            start_byte = (part_number - 1) * part_size
            end_byte = min(start_byte + part_size, len(data))
            part_data = data[start_byte:end_byte]

            logger.debug(f"Uploading part {part_number}/{total_parts} ({len(part_data)} bytes)")

            # Generate presigned URL for this part
            presigned_url = generate_presigned_url_for_part(path, upload_id, part_number)

            # Upload the part
            etag = upload_part_to_s3(presigned_url, part_data)

            # Store part info for completion
            parts.append(
                {
                    "PartNumber": part_number,
                    "ETag": etag,
                }
            )

        # Complete the multipart upload
        s3_path = complete_multipart_upload(path, upload_id, parts)
        logger.info(f"Successfully completed multipart upload for {path}")
        return s3_path

    except Exception as e:
        # Abort the multipart upload on any error
        if upload_id:
            abort_multipart_upload(path, upload_id)
        logger.error(f"Multipart upload failed for {path}: {e}")
        raise


async def upload_part_to_s3_async(presigned_url: str, data: bytes) -> str:
    """Upload a part using the presigned URL asynchronously and return the ETag."""
    try:
        # Configure timeout for large part uploads
        timeout = aiohttp.ClientTimeout(
            total=30 * 60,  # 30 minutes total timeout
            connect=30,  # 30 seconds to connect
            sock_read=5 * 60,  # 5 minutes to read response
            sock_connect=30,  # 30 seconds for socket connection
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.put(presigned_url, data=data) as response:
                if not response.ok:
                    # Get detailed error information from S3
                    error_body = await response.text()
                    error_headers = dict(response.headers)

                    logger.error("S3 part upload failed")
                    logger.error(f"HTTP Status: {response.status} {response.reason}")
                    logger.error(f"Response Headers: {error_headers}")
                    logger.error(f"Response Body: {error_body}")
                    logger.error(f"Request URL: {presigned_url}")
                    logger.error(f"Part data size: {len(data)} bytes")

                    # Try to parse XML error if it's XML
                    try:
                        import xml.etree.ElementTree as ET

                        root = ET.fromstring(error_body)
                        error_code = root.find(".//Code")
                        error_message = root.find(".//Message")
                        request_id = root.find(".//RequestId")

                        if error_code is not None:
                            logger.error(f"S3 Error Code: {error_code.text}")
                        if error_message is not None:
                            logger.error(f"S3 Error Message: {error_message.text}")
                        if request_id is not None:
                            logger.error(f"S3 Request ID: {request_id.text}")
                    except Exception as xml_parse_error:
                        logger.debug(f"Could not parse S3 error XML: {xml_parse_error}")

                response.raise_for_status()

                # Extract ETag from response headers (remove quotes if present)
                etag = response.headers.get("ETag", "").strip('"')
                logger.debug(f"Uploaded part with ETag: {etag}")
                return etag
    except Exception as e:
        logger.error(f"Error uploading part: {e}")
        raise


async def upload_large_file_multipart_async(
    data: bytes, path: str, part_size: int = 100 * 1024 * 1024, max_concurrent_parts: int = 5
) -> str:
    """
    Upload a large file using multipart upload with presigned URLs asynchronously.

    Args:
        data: The file data as bytes
        path: The S3 key path
        part_size: Size of each part in bytes (default 100MB)
        max_concurrent_parts: Maximum number of parts to upload concurrently

    Returns:
        The S3 path of the uploaded file
    """
    if not settings.USE_S3 or not s3_client:
        # For local storage, just write the file
        await asyncio.get_event_loop().run_in_executor(None, _write_file_sync, path, data)
        return path

    # If file is small enough, use regular upload
    if len(data) <= part_size:
        logger.debug(f"File size {len(data)} bytes is small enough for single-part upload")
        buffer = io.BytesIO(data)
        presigned_data = generate_presigned_url(path=path)
        return upload_to_bucket(presigned_data, {"file": ("data", buffer)})

    logger.info(f"Starting async multipart upload for {path} ({len(data)} bytes, {part_size} bytes per part)")

    upload_id = None
    try:
        # Create multipart upload
        upload_id = create_multipart_upload(path)

        # Calculate number of parts needed
        total_parts = (len(data) + part_size - 1) // part_size

        # Prepare all parts
        part_tasks = []
        for part_number in range(1, total_parts + 1):
            start_byte = (part_number - 1) * part_size
            end_byte = min(start_byte + part_size, len(data))
            part_data = data[start_byte:end_byte]

            # Generate presigned URL for this part
            presigned_url = generate_presigned_url_for_part(path, upload_id, part_number)

            # Create task for uploading this part
            part_tasks.append(
                {
                    "part_number": part_number,
                    "presigned_url": presigned_url,
                    "data": part_data,
                }
            )

        # Upload parts with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent_parts)

        async def upload_single_part(task_info):
            async with semaphore:
                logger.debug(
                    f"Uploading part {task_info['part_number']}/{total_parts} ({len(task_info['data'])} bytes)"
                )
                etag = await upload_part_to_s3_async(task_info["presigned_url"], task_info["data"])
                return {
                    "PartNumber": task_info["part_number"],
                    "ETag": etag,
                }

        # Execute all part uploads concurrently
        parts = await asyncio.gather(*[upload_single_part(task) for task in part_tasks])

        # Sort parts by part number (in case they completed out of order)
        parts.sort(key=lambda x: x["PartNumber"])

        # Complete the multipart upload
        s3_path = complete_multipart_upload(path, upload_id, parts)
        logger.info(f"Successfully completed async multipart upload for {path}")
        return s3_path

    except Exception as e:
        # Abort the multipart upload on any error
        if upload_id:
            abort_multipart_upload(path, upload_id)
        logger.error(f"Async multipart upload failed for {path}: {e}")
        raise


def _write_file_sync(path: str, data: bytes):
    """Helper function for writing files synchronously."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def smart_upload_to_s3(data: bytes, path: str, use_async: bool = True, part_size: int = 100 * 1024 * 1024) -> str:
    """
    Smart upload function that automatically chooses between single-part and multipart upload.

    Args:
        data: The file data as bytes
        path: The S3 key path
        use_async: Whether to use async multipart upload (default True)
        part_size: Size threshold for switching to multipart upload

    Returns:
        The S3 path of the uploaded file
    """
    # S3 requires minimum 5MB per part for multipart (except last part)
    # and has a 5GB limit for single-part uploads
    min_part_size = 5 * 1024 * 1024  # 5MB
    max_single_part_size = 5 * 1024 * 1024 * 1024  # 5GB

    # Adjust part size if it's too small
    actual_part_size = max(part_size, min_part_size)

    # Use multipart if file is larger than max single-part size or larger than part_size
    if len(data) > max_single_part_size or len(data) > actual_part_size:
        if use_async:
            # This should not be called in sync context, but we need to handle it
            raise ValueError("Cannot use async multipart upload in sync context")
        else:
            return upload_large_file_multipart(data, path, actual_part_size)
    else:
        # Use regular single-part upload
        buffer = io.BytesIO(data)
        presigned_data = generate_presigned_url(path=path)
        return upload_to_bucket(presigned_data, {"file": ("data", buffer)})


async def smart_upload_to_s3_async(data: bytes, path: str, part_size: int = 100 * 1024 * 1024) -> str:
    """Async version of smart_upload_to_s3."""
    # S3 requires minimum 5MB per part for multipart (except last part)
    # and has a 5GB limit for single-part uploads
    min_part_size = 5 * 1024 * 1024  # 5MB
    max_single_part_size = 5 * 1024 * 1024 * 1024  # 5GB

    # Adjust part size if it's too small
    actual_part_size = max(part_size, min_part_size)

    # Use multipart if file is larger than max single-part size or larger than part_size
    if len(data) > max_single_part_size or len(data) > actual_part_size:
        return await upload_large_file_multipart_async(data, path, actual_part_size)
    else:
        # Use regular single-part upload
        buffer = io.BytesIO(data)
        presigned_data = generate_presigned_url(path=path)
        return upload_to_bucket(presigned_data, {"file": ("data", buffer)})


async def smart_upload_via_orchestrator_async(
    api_client, data: bytes, path: str, part_size: int = 100 * 1024 * 1024
) -> str:
    """
    Smart upload function that uses the orchestrator's APIs for multipart upload coordination.

    Args:
        api_client: The API client to communicate with orchestrator
        data: The file data as bytes
        path: The S3 key path
        part_size: Size threshold for switching to multipart upload

    Returns:
        The S3 path of the uploaded file
    """

    logger.debug(f"Starting upload request for {path} with {len(data)} bytes")

    # Adaptive part size: use smaller parts for very large files to avoid timeouts
    # AWS S3 minimum part size is 5MB (except last part)
    min_part_size = 5 * 1024 * 1024  # 5MB
    max_part_size = 100 * 1024 * 1024  # 100MB

    adaptive_part_size = min(part_size, max_part_size)

    adaptive_part_size = max(adaptive_part_size, min_part_size)

    # Request upload info from orchestrator
    upload_info = await api_client.initiate_multipart_upload(
        path=path, file_size=len(data), part_size=adaptive_part_size
    )

    if not upload_info.get("use_multipart", False):
        # Use single-part upload
        presigned_data = upload_info["presigned_data"]
        buffer = io.BytesIO(data)
        return upload_to_bucket(presigned_data, {"file": ("data", buffer)})

    # Use multipart upload
    multipart_data = upload_info["multipart_data"]
    upload_id = multipart_data["upload_id"]
    presigned_urls = multipart_data["presigned_urls"]
    part_numbers = multipart_data["part_numbers"]
    actual_part_size = upload_info["part_size"]

    logger.info(f"Starting orchestrator-coordinated multipart upload for {path} ({len(data)} bytes)")

    # Validate multipart upload data before starting
    logger.debug("Multipart upload validation:")
    logger.debug(f"  - Upload ID: {upload_id}")
    logger.debug(f"  - Number of parts: {len(part_numbers)}")
    logger.debug(f"  - Number of URLs: {len(presigned_urls)}")
    logger.debug(f"  - Part numbers: {part_numbers}")
    logger.debug(f"  - Actual part size: {actual_part_size}")
    logger.debug(f"  - Total data size: {len(data)}")

    # Validate that we have matching parts and URLs
    if len(part_numbers) != len(presigned_urls):
        raise ValueError(f"Mismatch: {len(part_numbers)} part numbers but {len(presigned_urls)} URLs")

    # Validate part numbers are sequential and start from 1
    expected_parts = list(range(1, len(part_numbers) + 1))
    if part_numbers != expected_parts:
        logger.warning(f"Part numbers are not sequential: expected {expected_parts}, got {part_numbers}")

    try:
        # Upload parts using presigned URLs with proper timeout configuration
        parts = []

        # Configure timeout for large file uploads (30 minutes total, 5 minutes per operation)
        timeout = aiohttp.ClientTimeout(
            total=30 * 60,  # 30 minutes total timeout
            connect=30,  # 30 seconds to connect
            sock_read=5 * 60,  # 5 minutes to read response
            sock_connect=30,  # 30 seconds for socket connection
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for i, (part_number, url_info) in enumerate(zip(part_numbers, presigned_urls)):
                start_byte = i * actual_part_size
                end_byte = min(start_byte + actual_part_size, len(data))
                part_data = data[start_byte:end_byte]

                logger.debug(f"Uploading part {part_number}/{len(part_numbers)} ({len(part_data)} bytes)")

                # Add detailed debug information before upload
                logger.debug(f"Upload details for part {part_number}:")
                logger.debug(f"  - Part index: {i}")
                logger.debug(f"  - Upload ID: {upload_id}")
                logger.debug(f"  - Part size: {len(part_data)} bytes")
                logger.debug(f"  - Byte range: {start_byte}-{end_byte}")
                logger.debug(f"  - URL: {url_info['url'][:100]}...")  # Truncate URL for readability

                # Upload the part with retry logic for timeout errors
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        start_time = asyncio.get_event_loop().time()
                        async with session.put(url_info["url"], data=part_data) as response:
                            upload_time = asyncio.get_event_loop().time() - start_time

                            if not response.ok:
                                # Get detailed error information from S3
                                error_body = await response.text()
                                error_headers = dict(response.headers)

                                logger.error(
                                    f"S3 multipart upload failed for part {part_number} (attempt {attempt + 1}/{max_retries})"
                                )
                                logger.error(f"HTTP Status: {response.status} {response.reason}")
                                logger.error(f"Response Headers: {error_headers}")
                                logger.error(f"Response Body: {error_body}")
                                logger.error(f"Request URL: {url_info['url']}")
                                logger.error(f"Part data size: {len(part_data)} bytes")
                                logger.error(f"Upload ID: {upload_id}")

                                # Try to parse XML error if it's XML
                                try:
                                    import xml.etree.ElementTree as ET

                                    root = ET.fromstring(error_body)
                                    error_code = root.find(".//Code")
                                    error_message = root.find(".//Message")
                                    request_id = root.find(".//RequestId")

                                    if error_code is not None:
                                        logger.error(f"S3 Error Code: {error_code.text}")
                                        # Check if this is a retryable timeout error
                                        if error_code.text == "RequestTimeout" and attempt < max_retries - 1:
                                            logger.info(
                                                f"RequestTimeout detected, retrying part {part_number} (attempt {attempt + 2}/{max_retries})"
                                            )
                                            await asyncio.sleep(2**attempt)  # Exponential backoff
                                            continue
                                    if error_message is not None:
                                        logger.error(f"S3 Error Message: {error_message.text}")
                                    if request_id is not None:
                                        logger.error(f"S3 Request ID: {request_id.text}")
                                except Exception as xml_parse_error:
                                    logger.debug(f"Could not parse S3 error XML: {xml_parse_error}")

                            response.raise_for_status()

                            # Extract ETag from response headers (remove quotes if present)
                            etag = response.headers.get("ETag", "").strip('"')
                            logger.debug(f"Uploaded part {part_number} with ETag: {etag}")

                            # Log upload performance
                            upload_speed_mbps = (len(part_data) / (1024 * 1024)) / max(upload_time, 0.001)
                            logger.debug(
                                f"Part {part_number} upload completed in {upload_time:.2f}s ({upload_speed_mbps:.2f} MB/s)"
                            )

                            parts.append(
                                {
                                    "PartNumber": part_number,
                                    "ETag": etag,
                                }
                            )
                            break  # Success, exit retry loop

                    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError, aiohttp.ClientError) as timeout_error:
                        logger.warning(
                            f"Timeout uploading part {part_number} (attempt {attempt + 1}/{max_retries}): {timeout_error}"
                        )
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            logger.info(f"Retrying part {part_number} in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Failed to upload part {part_number} after {max_retries} attempts")
                            raise

        # Complete the multipart upload via orchestrator
        completion_result = await api_client.complete_multipart_upload(path=path, upload_id=upload_id, parts=parts)

        s3_path = completion_result["s3_path"]
        logger.info(f"Successfully completed orchestrator-coordinated multipart upload for {path}")
        return s3_path

    except Exception as e:
        # Abort the multipart upload via orchestrator on any error
        try:
            await api_client.abort_multipart_upload(path=path, upload_id=upload_id)
        except Exception:
            logger.warning(f"Failed to abort multipart upload {upload_id} for {path}")

        logger.exception(f"Orchestrator-coordinated multipart upload failed for {path}: {e}")
        raise


def download_activation(path: str) -> torch.Tensor:
    """Download an activation from S3 storage."""
    if not settings.USE_S3 or not s3_client:
        # For local storage, read the file directly
        with open(path, "rb") as f:
            buffer = io.BytesIO(f.read())
        tensor = torch.load(buffer, weights_only=False, map_location=settings.DEVICE)
        assert isinstance(
            tensor, torch.Tensor
        ), f"Downloaded tensor is not a torch.Tensor: {type(tensor)}, path: {path}"
        return tensor

    bucket, path = normalize_s3_path(path)

    # Download from S3
    response = s3_client.get_object(Bucket=bucket, Key=path)
    buffer = io.BytesIO(response["Body"].read())

    tensor = torch.load(buffer, weights_only=False, map_location=settings.DEVICE)
    assert isinstance(tensor, torch.Tensor), f"Downloaded tensor is not a torch.Tensor: {type(tensor)}, path: {path}"
    check_for_nans(tensor, f"activation downloaded from {path}")
    return tensor


def download_weights_or_optimizer_state(
    path: str,
    partition: Partition = Partition(),
    data_type: Literal["weights", "optimizer_state"] = "weights",
) -> torch.Tensor:
    """Download weights from S3 storage."""
    bucket, path = normalize_s3_path(path)

    if data_type == "weights":
        data = partition.weight_data
    elif data_type == "optimizer_state":
        data = partition.optimizer_state_data
    else:
        raise ValueError(f"Invalid type: {data_type}")

    # if partition is not specified, download the full tensor
    if data.chunk_start_byte is None or data.chunk_end_byte is None:
        logger.warning(f"Chunk {partition.chunk_number} has no start or end byte")
        response = s3_client.get_object(Bucket=bucket, Key=path)
    else:
        byte_range = f"bytes={data.chunk_start_byte}-{data.chunk_end_byte-1}"
        response = s3_client.get_object(Bucket=bucket, Key=path, Range=byte_range)

    binary_data = response["Body"].read()
    section_numpy = np.frombuffer(binary_data, dtype=np.uint8)
    section_torch = torch.from_numpy(section_numpy.copy())
    # assumes default dtype if not specified
    section_torch = section_torch.view(getattr(torch, data.chunk_dtype))
    logger.info(f"Downloaded partition {partition} of {path!r}")
    return section_torch


def download_metadata(path: str) -> dict[str, Any]:
    """Download metadata from S3 storage."""
    if not settings.USE_S3 or not s3_client:
        # For local storage, read the file directly
        with open(path, "r") as f:
            return json.loads(f.read())

    bucket, path = normalize_s3_path(path)
    response = s3_client.get_object(Bucket=bucket, Key=path)
    return json.loads(response["Body"].read())


def download_optimizer_state(path: str) -> dict[str, Any]:
    """Download optimizer state from S3 storage."""
    bucket, path = normalize_s3_path(path)
    response = s3_client.get_object(Bucket=bucket, Key=path)
    buffer = io.BytesIO(response["Body"].read())
    return torch.load(buffer, weights_only=False, map_location=settings.DEVICE)


def list_all_files(prefix: str = "") -> list[str]:
    """List all files with a given prefix."""
    if not settings.USE_S3 or not s3_client:
        return os.listdir()

    try:
        response = s3_client.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=prefix)
        if "Contents" in response:
            return [obj["Key"] for obj in response["Contents"]]
        return []
    except Exception as e:
        logger.error(f"Failed to list files with prefix {prefix}: {e}")
        return []


# utils/s3_interactions.py


def delete(path: str):
    """Delete a file from local disk or S3."""

    parsed = urlparse(path)

    if parsed.scheme == "s3":
        if not s3_client:
            logger.warning(f"S3 client not available, cannot delete S3 path: {path}")
            return

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
            logger.debug(f"Deleted from S3: {path}")
        except Exception as e:
            logger.exception(f"Failed to delete from S3: {path}")
            raise
    else:
        file_path = Path(path)
        if not file_path.exists():
            logger.warning(f"Tried to delete local file but not found: {file_path}")
            return
        os.remove(file_path)
        logger.debug(f"Deleted local file: {file_path}")


def file_exists(path: str) -> bool:
    """
    Check if a file exists either on S3 or locally.

    Args:
        path: The path to check. Can be either an S3 path (s3://bucket/key) or a local path.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    parsed = urlparse(path)

    if parsed.scheme == "s3":
        if not settings.USE_S3 or not s3_client:
            logger.warning(f"S3 client not available, cannot check S3 path: {path}")
            return False

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                return False
            else:
                logger.error(f"Error checking if file exists on S3: {e}")
                raise
    else:
        return Path(path).exists()


def verify_file_size(path: str, min_size: int = 0, max_size: int = np.inf) -> bool:
    """
    Verify the size of a file is within a given range.
    """
    if not file_exists(path):
        return False
    size = get_file_size(path)

    if min_size <= size <= max_size:
        return True
    else:
        logger.warning(f"File size is not within min/max ({min_size}/{max_size}) range: {path} ({size} bytes)")
        return False


def get_file_size(path: str) -> int:
    """
    Get the size of a file in bytes, works with both S3 and local files.

    Args:
        path: The path to check. Can be either an S3 path (s3://bucket/key) or a local path.

    Returns:
        int: The file size in bytes.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If S3 is not available for S3 paths.
    """
    parsed = urlparse(path)

    if parsed.scheme == "s3":
        if not settings.USE_S3 or not s3_client:
            raise ValueError(f"S3 client not available, cannot get size of S3 path: {path}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        try:
            response = s3_client.head_object(Bucket=bucket, Key=key)
            size = response.get("ContentLength", 0)
            logger.debug(f"S3 file size for {path}: {size} bytes")
            return size
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                raise FileNotFoundError(f"S3 file not found: {path}")
            else:
                logger.error(f"Error getting file size from S3: {e}")
                raise
    else:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")

        size = file_path.stat().st_size
        logger.debug(f"Local file size for {path}: {size} bytes")
        return size


def compare_files_sampled(path1: str, path2: str, sample_percentage: float, num_samples: int = 10) -> float:
    """
    Compare two files by sampling random byte ranges and calculating bit-level agreement rate.

    Args:
        path1: First file path (can be S3 or local)
        path2: Second file path (can be S3 or local)
        sample_percentage: Percentage of file to sample (0.0 to 100.0)
        num_samples: Number of byte ranges to sample (default 10)

    Returns:
        float: Agreement rate as percentage (0.0 to 100.0)

    Raises:
        ValueError: If files have different sizes or invalid percentage/samples
        FileNotFoundError: If either file doesn't exist
    """
    import random

    if not (0.0 <= sample_percentage <= 100.0):
        raise ValueError(f"Sample percentage must be between 0.0 and 100.0, got {sample_percentage}")

    if num_samples < 1:
        raise ValueError(f"Number of samples must be at least 1, got {num_samples}")

    # Get file sizes
    size1 = get_file_size(path1)
    size2 = get_file_size(path2)

    if size1 != size2:
        raise ValueError(f"Files have different sizes: {path1} ({size1} bytes) vs {path2} ({size2} bytes)")

    if size1 == 0:
        logger.info("Both files are empty, returning 100% agreement")
        return 100.0

    # Calculate total bytes to sample
    total_bytes = size1
    sample_bytes = max(1, int(total_bytes * sample_percentage / 100.0))

    # Calculate bytes per range
    bytes_per_range = max(1, sample_bytes // num_samples)

    logger.info(f"Comparing {path1} and {path2}")
    logger.info(f"File size: {total_bytes} bytes")
    logger.info(f"Sampling {sample_bytes} bytes ({sample_percentage}%) across {num_samples} ranges")
    logger.info(f"Bytes per range: {bytes_per_range}")

    # Generate random starting positions for each range
    max_start_pos = total_bytes - bytes_per_range
    range_starts = sorted(random.sample(range(max_start_pos), min(num_samples, max_start_pos + 1)))

    # Generate ranges to sample
    ranges = [(start, start + bytes_per_range - 1) for start in range_starts]

    # Download sampled bytes from both files
    data1 = _download_byte_ranges(path1, ranges)
    data2 = _download_byte_ranges(path2, ranges)

    # Compare bit by bit
    total_bits = 0
    matching_bits = 0

    for byte1, byte2 in zip(data1, data2):
        for bit_pos in range(8):
            bit1 = (byte1 >> bit_pos) & 1
            bit2 = (byte2 >> bit_pos) & 1
            total_bits += 1
            if bit1 == bit2:
                matching_bits += 1

    agreement_rate = (matching_bits / total_bits) * 100.0 if total_bits > 0 else 100.0

    logger.info(f"Agreement rate: {agreement_rate:.2f}% ({matching_bits}/{total_bits} bits match)")
    return agreement_rate


def _download_byte_ranges(path: str, ranges: list[tuple[int, int]]) -> bytes:
    """
    Download specific byte ranges from a file.

    Args:
        path: File path (S3 or local)
        ranges: List of (start, end) tuples representing byte ranges to download

    Returns:
        bytes: The requested bytes in order
    """
    parsed = urlparse(path)

    if parsed.scheme == "s3":
        if not settings.USE_S3 or not s3_client:
            raise ValueError(f"S3 client not available, cannot download from S3 path: {path}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        result = bytearray()
        for start, end in ranges:
            byte_range = f"bytes={start}-{end}"
            try:
                response = s3_client.get_object(Bucket=bucket, Key=key, Range=byte_range)
                chunk_data = response["Body"].read()
                result.extend(chunk_data)

            except ClientError as e:
                logger.error(f"Error downloading byte range {byte_range} from S3: {e}")
                raise

        return bytes(result)

    else:
        # Local file
        file_path = Path(path)
        result = bytearray()

        with open(file_path, "rb") as f:
            for start, end in ranges:
                f.seek(start)
                result.extend(f.read(end - start + 1))

        return bytes(result)


# Initialize S3 client on import
initialize_s3()
