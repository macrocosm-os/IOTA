import os
import sys
import argparse
import uvicorn
import asyncio
from fastapi import FastAPI
from orchestrator.gradient_validator_api import router as gradient_validator_router
from gradient_validator.gradient_validator import GradientValidator
import settings
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG")  # Add stderr handler for terminal output
if os.path.exists("validators.log"):
    try:
        os.remove("validators.log")
        logger.info("Removed existing validators.log file")
    except OSError as e:
        logger.error(f"Error removing validators.log: {e}")

logger.add(
    "validators.log",
    rotation="10 MB",  # Rotate at 10MB
    level="DEBUG",
    retention=1,  # Keep only latest log file
)


async def create_validator_app():
    """Create and configure a validator FastAPI application."""
    app = FastAPI(title="Gradient Validator Service")

    # Initialize gradient validator with proper layer setup
    validator = await GradientValidator.create()
    # Register validator in app state
    app.state.validator = validator

    # Add gradient validator router
    app.include_router(gradient_validator_router)

    # Add Prometheus instrumentation if enabled
    if os.getenv("PROMETHEUS", "") != "":
        Instrumentator().instrument(app).expose(app)

    asyncio.create_task(validator.is_registered_loop())
    await validator.start_weight_submission_task()
    return app


async def run_validator(host: str, port: int):
    """Run a single validator instance."""
    app = await create_validator_app()
    config = uvicorn.Config(app, host="0.0.0.0", port=int(settings.VALIDATOR_INTERNAL_PORT))
    server = uvicorn.Server(config)

    logger.info(f"Starting Gradient Validator service on {host}:{port}")
    await server.serve()


async def run_multiple_validators(host: str, start_port: int, count: int):
    """Run multiple validator instances."""
    tasks = []
    for i in range(count):
        port = start_port + i
        tasks.append(run_validator(host, port))

    logger.info(f"Starting {count} Gradient Validator instances on ports {start_port}-{start_port + count - 1}")
    await asyncio.gather(*tasks)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch Gradient Validator service(s)")
    parser.add_argument(
        "--host",
        default=os.getenv("VALIDATOR_HOST", settings.VALIDATOR_HOSTS),
        help="Host to run the validator(s) on",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(settings.VALIDATOR_PORTS[0]),
        help="Starting port number for validator(s)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=int(settings.VALIDATOR_COUNT),
        help="Number of validator instances to run",
    )

    args = parser.parse_args()

    try:
        logger.info(f"Launching {args.count} validator(s) on {args.host}:{args.port}")
        if args.count == 1:
            # Run single validator
            asyncio.run(run_validator("0.0.0.0", settings.VALIDATOR_PORTS[0]))
        else:
            # Run multiple validators
            asyncio.run(
                run_multiple_validators(
                    settings.VALIDATOR_HOSTS[0], settings.VALIDATOR_PORTS[0], settings.VALIDATOR_COUNT
                )
            )
    except KeyboardInterrupt:
        logger.info("Shutting down validator service(s)")
    except Exception as e:
        logger.exception(f"Error running validator service(s): {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
