import time
import settings
from typing import Annotated, Optional

from loguru import logger
from fastapi import APIRouter, HTTPException, Header, FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from utils.bt_utils import verify_entity_type
from utils.epistula import EpistulaHeaders, create_message_body

from orchestrator.orchestrator import orchestrator
from utils.partitions import Partition
from uuid import uuid4

from orchestrator.serializers import (
    AllLossesResponse,
    LayerAssignmentResponse,
    LossReportRequest,
    LossReportResponse,
    MinerRegistrationResponse,
    MinerStatusUpdate,
    SubmittedWeights,
)


def get_signed_by_key(request: Request) -> str:
    """Get the signed_by key for rate limiting. Falls back to IP address if not authenticated."""
    try:
        # Try to get signed_by from request headers
        signed_by = request.headers.get("Epistula-Signed-By")
        if signed_by:
            return signed_by
    except Exception:
        pass
    # Fall back to IP address for unauthenticated endpoints
    return get_remote_address(request)


def get_remote_address(request: Request) -> str:
    """Get the remote address, considering forwarded headers."""
    if "X-Forwarded-For" in request.headers:
        # Get the first address in X-Forwarded-For, which is the client's IP
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    elif request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


app = FastAPI()

# Initialize rate limiter
hotkey_limiter = Limiter(key_func=get_signed_by_key)
ip_limiter = Limiter(key_func=get_remote_address)

# Add rate limit exception handler to app
app.state.hotkey_limiter = hotkey_limiter
app.state.ip_limiter = ip_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

router = APIRouter(prefix="/orchestrator")


# Load in the initialized state of the orchestrator


@router.get("/global_miner_weights", response_model=dict[str, float])
@ip_limiter.limit(settings.IP_LIMIT)
async def get_global_miner_weights(request: Request):  # Required for rate limiting
    return await orchestrator.get_global_miner_scores()


@router.post("/submit_miner_weights", response_model=dict[str, str])
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def submit_miner_weights(
    request: Request,  # Required for rate limiting
    weights: dict[str, float],
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )
    error = headers.verify_signature_v2(create_message_body(weights), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    if settings.BITTENSOR:
        # Verify the entity is a miner and matches the UID
        verify_entity_type(signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="validator")
    await orchestrator.submit_miner_scores(weights)

    return {"message": "Miner weights submitted successfully"}


@router.post("/register", response_model=MinerRegistrationResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def register_miner(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=None,
        layer=None,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        try:
            error = headers.verify_signature_v2(create_message_body({}), time.time())
            if error:
                raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

            miner_entities = orchestrator.miner_registry.get_all_miner_data()

            if settings.BITTENSOR:
                # Verify the entity is a miner
                verify_entity_type(
                    signed_by=signed_by,
                    metagraph=orchestrator.metagraph,
                    required_type="miner",
                )

            logger.info(f"Received registration request from {signed_by[:8]}")

            if signed_by in miner_entities:
                logger.error(f"Miner {signed_by} already registered")
                return MinerRegistrationResponse(hotkey=signed_by, layer=miner_entities[signed_by].layer)

            layer = await orchestrator.register(hotkey=signed_by)
            if layer is None:
                raise HTTPException(status_code=409, detail="Failed to register miner")
            return MinerRegistrationResponse(hotkey=signed_by, layer=layer)
        except Exception as e:
            logger.error(f"Failed to register miner {signed_by[:8]}: {str(e)}")
            raise HTTPException(status_code=409, detail="Failed to register miner") from e


@router.post("/miners/status", response_model=dict)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def update_miner_status(
    request: Request,  # Required for rate limiting
    status_update: MinerStatusUpdate,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body(status_update.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        if settings.BITTENSOR:
            # Verify the entity is a miner and matches the UID
            verify_entity_type(signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="miner")

        try:
            await orchestrator.update_status(
                hotkey=signed_by,
                status=status_update.status,
                activation_uid=status_update.activation_uid,
                activation_path=status_update.activation_path,
            )
            return {"message": "Status updated successfully"}
        except IndexError:
            raise HTTPException(status_code=404, detail="Miner not found")


@router.post("/miners/request_layer", response_model=LayerAssignmentResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def request_layer(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body({}), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        if settings.BITTENSOR:
            # Verify the entity is a miner and matches the UID
            entity_info = verify_entity_type(
                signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="miner"
            )

            logger.info(f"Miner {entity_info['uid']} ({signed_by[:8]}...) requesting layer")

        try:
            layer = await orchestrator.request_layer()
            return LayerAssignmentResponse(layer=layer)
        except IndexError:
            raise HTTPException(status_code=404, detail="Miner not found")


@router.get("/healthcheck")
@ip_limiter.limit(settings.IP_LIMIT)
async def healthcheck(request: Request):  # Required for rate limiting
    return {"status": "healthy"}


@router.post("/miners/notify_weights_uploaded")
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def notify_weights_uploaded(
    request: Request,  # Required for rate limiting
    weights_path: str,
    metadata_path: str,
    optimizer_state_path: str,
    optimizer_state_metadata_path: str,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
        weights_path=weights_path,
        metadata_path=metadata_path,
        optimizer_state_path=optimizer_state_path,
        optimizer_state_metadata_path=optimizer_state_metadata_path,
    ):
        hotkey = signed_by
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body({}), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        if settings.BITTENSOR:
            # Verify the entity is a miner and matches the UID
            verify_entity_type(signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="miner")

        try:
            success = await orchestrator.notify_weights_uploaded(
                hotkey=hotkey,
                weights_path=weights_path,
                metadata_path=metadata_path,
                optimizer_state_path=optimizer_state_path,
                optimizer_state_metadata_path=optimizer_state_metadata_path,
            )
            return {
                "message": "Weights notification processed successfully",
                "success": success,
            }
        except IndexError:
            raise HTTPException(status_code=404, detail="Miner not found")


@router.post("/miners/notify_merged_partitions_uploaded")
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def notify_merged_partitions_uploaded(
    request: Request,  # Required for rate limiting
    partitions: list[Partition],
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
        partitions=partitions,
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )

        body = [p.model_dump() for p in partitions]
        error = headers.verify_signature_v2(create_message_body(body), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        await orchestrator.notify_merged_partitions_uploaded(hotkey=signed_by, partitions=partitions)
        return {"message": "Merged partitions notification processed successfully"}


@router.get("/losses", response_model=AllLossesResponse)
@ip_limiter.limit("15/minute")
async def get_all_losses(request: Request):  # Required for rate limiting
    return AllLossesResponse(losses={str(k): v for k, v in orchestrator.losses.items()})


@router.post("/miners/report_loss", response_model=LossReportResponse)
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def report_loss(
    request: Request,  # Required for rate limiting
    loss_report: LossReportRequest,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
):
    with logger.contextualize(
        activation_uid=loss_report.activation_uid,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
        loss=loss_report.loss_value,
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )

        error = headers.verify_signature_v2(create_message_body(loss_report.model_dump()), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        if settings.BITTENSOR:
            # Verify the entity is a miner and matches the UID
            verify_entity_type(signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="miner")

        try:
            await orchestrator.record_and_report_loss(
                hotkey=signed_by,
                activation_uid=loss_report.activation_uid,
                loss=loss_report.loss_value,
            )
            return LossReportResponse(
                hotkey=signed_by,
                activation_uid=loss_report.activation_uid,
                loss_value=loss_report.loss_value,
                timestamp=time.time(),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/is_merging")
@ip_limiter.limit(settings.IP_LIMIT)
async def is_merging(request: Request, layer: int):  # Required for rate limiting
    """Check if the system is currently in merging phase."""
    # Public endpoint, no authentication required
    merging_phase, num_sections = await orchestrator.is_merging(layer=layer)
    return {"status": merging_phase, "num_sections": num_sections}


@router.get("/get_chunks_for_miner", response_model=tuple[list[SubmittedWeights], list[int]])
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def get_chunks_for_miner(
    request: Request,  # Required for rate limiting
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
) -> tuple[list[SubmittedWeights], list[int]]:
    """Get the chunks for a miner - this is a list of strings, each representing a chunk."""
    with logger.contextualize(
        activation_uid=None,
        layer=orchestrator.miner_registry.get_miner_data(signed_by).layer,
        hotkey=signed_by,
        request_id=str(uuid4()),
    ):
        headers = EpistulaHeaders(
            version=version,
            timestamp=timestamp,
            uuid=uuid,
            signed_by=signed_by,
            request_signature=request_signature,
        )
        error = headers.verify_signature_v2(create_message_body({}), time.time())
        if error:
            raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

        if settings.BITTENSOR:
            # Verify the entity is a miner
            verify_entity_type(signed_by=signed_by, metagraph=orchestrator.metagraph, required_type="miner")

        return await orchestrator.get_chunks_for_miner(hotkey=signed_by)


@router.post("/register_validator")
@hotkey_limiter.limit(settings.HOTKEY_LIMIT)
async def register_validator(
    request: Request,  # Required for rate limiting
    host: str,
    port: int,
    version: Annotated[str, Header(alias="Epistula-Version")],
    timestamp: Annotated[str, Header(alias="Epistula-Timestamp")],
    uuid: Annotated[str, Header(alias="Epistula-Uuid")],
    signed_by: Annotated[str, Header(alias="Epistula-Signed-By")],
    request_signature: Annotated[str, Header(alias="Epistula-Request-Signature")],
    scheme: str = "http",
):
    """Register a validator with the orchestrator."""
    headers = EpistulaHeaders(
        version=version,
        timestamp=timestamp,
        uuid=uuid,
        signed_by=signed_by,
        request_signature=request_signature,
    )

    error = headers.verify_signature_v2(create_message_body({}), time.time())
    if error:
        raise HTTPException(status_code=401, detail=f"Epistula verification failed: {error}")

    if settings.BITTENSOR:
        # Verify the entity is a validator
        verify_entity_type(
            signed_by=signed_by,
            metagraph=orchestrator.metagraph,
            required_type="validator",
        )

    success = await orchestrator.register_validator(hotkey=signed_by, host=host, port=port, scheme=scheme)

    if not success:
        raise HTTPException(status_code=400, detail="Validator registration failed")

    return {"message": "Validator registered successfully"}


@router.get("/metrics/miner_performance/{miner_hotkey}")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_miner_performance(
    request: Request, miner_hotkey: str, time_window_seconds: Optional[int] = 3600  # Required for rate limiting
):
    """Get performance metrics for a specific miner."""
    return orchestrator.get_miner_performance_metrics(miner_hotkey, time_window_seconds)


@router.get("/metrics/layer_performance/{layer}")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_layer_performance(
    request: Request, layer: int, time_window_seconds: Optional[int] = 3600  # Required for rate limiting
):
    """Get performance metrics for a specific layer."""
    return orchestrator.get_layer_performance_metrics(layer, time_window_seconds)


@router.get("/metrics/dashboard")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_real_time_dashboard(request: Request):  # Required for rate limiting
    """Get real-time performance data for monitoring dashboard."""
    return orchestrator.get_real_time_performance_dashboard()


@router.get("/metrics/weight_merging_summary")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_weight_merging_metrics_summary(request: Request):  # Required for rate limiting
    """Get a summary of weight merging metrics."""
    return orchestrator.get_weight_merging_metrics_summary()


@router.get("/metrics/active_merge_sessions")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_active_merge_sessions(request: Request):  # Required for rate limiting
    """Get information about currently active merge sessions."""
    return orchestrator.get_active_merge_sessions()


@router.get("/metrics/merge_performance_history")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_merge_performance_history(
    request: Request,  # Required for rate limiting
    time_window_hours: Optional[int] = 24,
    include_layer_breakdown: Optional[bool] = True,
):
    """Get historical merge performance data."""
    try:
        # Get merge statistics for all layers over the time window
        time_window_seconds = time_window_hours * 3600

        # Overall statistics
        overall_stats = orchestrator.weight_merging_metrics_collector.get_merge_statistics(
            layer=None, time_window_seconds=time_window_seconds
        )

        # Layer-specific statistics if requested
        layer_stats = {}
        if include_layer_breakdown:
            for layer in range(orchestrator.N_LAYERS):
                layer_stats[layer] = orchestrator.weight_merging_metrics_collector.get_merge_statistics(
                    layer=layer, time_window_seconds=time_window_seconds
                )

        # Active sessions info
        active_sessions = orchestrator.get_active_merge_sessions()

        # Recent completed sessions (last 10)
        recent_sessions = []
        for session in orchestrator.weight_merging_metrics_collector.completed_sessions[-10:]:
            recent_sessions.append(
                {
                    "session_id": session.session_id,
                    "layer": session.layer,
                    "status": session.status,
                    "started_at": session.started_at,
                    "completed_at": session.completed_at,
                    "duration": session.get_session_duration(),
                    "participation_rate": session.get_participation_rate(),
                    "target_miners_count": len(session.target_miners),
                    "weights_received_count": len(session.weights_received),
                    "partitions_completed_count": len(session.partitions_completed),
                }
            )

        return {
            "time_window_hours": time_window_hours,
            "overall_statistics": overall_stats,
            "layer_statistics": layer_stats,
            "active_sessions": active_sessions,
            "recent_sessions": recent_sessions,
            "timestamp": time.time(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/historical_processing_times")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_historical_processing_times(
    request: Request,  # Required for rate limiting
    time_window_hours: Optional[int] = 24,
    include_layer_breakdown: Optional[bool] = True,
    granularity: Optional[str] = "hourly",
):
    """Get historical processing time data for trend analysis."""
    try:
        return orchestrator.get_historical_processing_times(
            time_window_hours=time_window_hours,
            include_layer_breakdown=include_layer_breakdown,
            granularity=granularity,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/merge_session_timeline")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_merge_session_timeline(
    request: Request, time_window_hours: Optional[int] = 24  # Required for rate limiting
):
    """Get merge session timeline data for visualization with success/failure bands."""
    try:
        if time_window_hours is None:
            time_window_hours = 24

        return orchestrator.get_merge_session_timeline(time_window_hours=time_window_hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/miners_grid_status")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_miners_grid_status(request: Request):  # Required for rate limiting
    """Get comprehensive miner status data for grid visualization."""
    try:
        return orchestrator.get_miners_grid_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/miner_detail/{miner_hotkey}")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_miner_detail(request: Request, miner_hotkey: str):  # Required for rate limiting
    """Get detailed information about a specific miner for hover display."""
    try:
        return orchestrator.get_miner_detail(miner_hotkey)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/miners_by_status")
@ip_limiter.limit(settings.IP_LIMIT)
async def get_miners_by_status(request: Request, layer: Optional[int] = None):  # Required for rate limiting
    """Get miners grouped by their current merge status."""
    try:
        return orchestrator.get_miners_by_merge_status(layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)
