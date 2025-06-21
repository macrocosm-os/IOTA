import os

import torch
from dotenv import load_dotenv
from loguru import logger

from model.configs import LLAMA32_CONFIG_15B, LLAMA32_CONFIG_100M

DOTENV_PATH = os.getenv("DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=DOTENV_PATH):
    # raise ValueError("No .env file found")
    logger.warning("No .env file found")

MOCK = os.getenv("MOCK") == "True"

# Model
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
PRETRAINED = False

MODEL_CFG = LLAMA32_CONFIG_100M if MOCK else LLAMA32_CONFIG_15B

# Bittensor
BITTENSOR = os.getenv("BITTENSOR") == "True"

# Dataset
DATASET_NAME = "HuggingFaceFW/fineweb"
SHUFFLE_DATASET = True

ACTIVATION_CACHE_TIMEOUT = 60 * 20

# Training
BATCH_SIZE = 1
SEQUENCE_LENGTH = 800
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACK_SAMPLES = True
WEIGHT_DECAY = 1e-1
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 2 * 1e-4
TOTAL_TRAIN_STEPS = 100_000
LR_WARMUP_START_FACTOR = 1  # 5e-3
LR_WARMUP_STEPS = 1
LR_CONST_STEPS = 500
LR_TAIL_STEPS_FRAC = 0.02
LR_FINAL_FACTOR = 0.10
LR_SAW_CYCLE_LENGTH = 1000
USE_WANDB = False

# MODEL MERGING
MINER_MERGE_PARTITIONS = 0.3
TARGET_EFFECTIVE_BATCH_SIZE = 30 if MOCK else 500

# swarm
# MODEL_SPLITS = [[-1, 5], [5, 10], [10, -1]]  # For 1B models
# MODEL_SPLITS = [[-1, 3], [3, -1]]  # For 100M models 2 layers
MOCK_MODEL_SPLITS = [[-1, 2], [2, 4], [4, -1]]  # For 100M models 3 layers
# MODEL_SPLITS = [[-1, 9], [9, 19], [19, -1]] # For 3B models
# MODEL_SPLITS = [[-1, 11], [11, 27], [27, -1]]  # For 12B models
REAL_MODEL_SPLITS = [[-1, 8], [8, 19], [19, 30], [30, 41], [41, -1]]  # 15B
# 13B 5 layers
# MODEL_SPLITS = [[-1, 8], [8, 16], [16, 24], [24, 32], [32, -1]]

MODEL_SPLITS = MOCK_MODEL_SPLITS if MOCK else REAL_MODEL_SPLITS

N_LAYERS = len(MODEL_SPLITS)
TIMEOUT = 20000000000000000
PHASE_TIMEOUT = 60 * 60 * 3  # 1 hour

HF_TOKEN = os.getenv("HF_TOKEN")
SYNC_WEIGHTS = os.getenv("SYNC_WEIGHTS") == "True"
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
DEFAULT_AWS_REGION = "eu-north-1"
AWS_REGION = os.getenv("AWS_REGION", DEFAULT_AWS_REGION)

# API
MODEL_MERGING_PORT = int(os.getenv("MODEL_MERGING_PORT", 32803))
MODEL_MERGING_HOST = os.getenv("MODEL_MERGING_HOST", "0.0.0.0")
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 8000))
ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "localhost")
ORCHESTRATOR_SCHEME = os.getenv("ORCHESTRATOR_SCHEME", "http")
ORCHESTRATOR_URL = f"{ORCHESTRATOR_SCHEME}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"

# S3
S3_BUCKET = os.getenv("S3_BUCKET")
USE_S3 = os.getenv("USE_S3", True)  # always use it if not specified


# Epistula
SIGNATURE_TIMEOUT_MS = 10000

# Gradient Validators
VALIDATE = os.getenv("VALIDATE") == "True"
VALIDATOR_COUNT = int(os.getenv("VALIDATOR_COUNT", "1"))
VALIDATOR_HOSTS = (
    os.getenv("VALIDATOR_HOSTS", "localhost").strip().split(",") if os.getenv("VALIDATOR_HOSTS") else ["localhost"]
)
WEIGHT_SUBMIT_INTERVAL: int = 3600  # submit weight every 1 hour
SCORE_VALIDITY_PERIOD = 3000 * 12  # should be equal to immunity period

# Validation Thresholds
COSINE_SIMILARITY_THRESHOLD = 0.9
ACTIVATION_MAGNITUDE_THRESHOLD = 0.8
WEIGHT_MAGNITUDE_THRESHOLD = 0.7

# Handle validator ports more gracefully
validator_ports_env = os.getenv("VALIDATOR_PORTS", "")
if validator_ports_env and validator_ports_env.strip():
    VALIDATOR_PORTS = [int(port.strip()) for port in validator_ports_env.split(",") if port.strip()]
else:
    VALIDATOR_PORTS = [8081]  # Default starting port

VALIDATOR_SCHEME = os.getenv("VALIDATOR_SCHEME", "http")
TCP_PORT_ENV_NAME = f"RUNPOD_TCP_PORT_{str(VALIDATOR_PORTS[0])}"
VALIDATOR_PUBLIC_PORT = os.getenv(TCP_PORT_ENV_NAME, str(VALIDATOR_PORTS[0]))
VALIDATOR_PUBLIC_IP = os.getenv("RUNPOD_PUBLIC_IP", VALIDATOR_HOSTS[0])
VALIDATOR_EXTERNAL_PORT = os.getenv("VALIDATOR_EXTERNAL_PORT", VALIDATOR_PUBLIC_PORT)
VALIDATOR_INTERNAL_PORT = os.getenv("VALIDATOR_INTERNAL_PORT", VALIDATOR_PORTS[0])

# Ensure we have enough host/port entries for each validator
while len(VALIDATOR_HOSTS) < VALIDATOR_COUNT:
    VALIDATOR_HOSTS.append(VALIDATOR_HOSTS[0])
while len(VALIDATOR_PORTS) < VALIDATOR_COUNT:
    VALIDATOR_PORTS.append(VALIDATOR_PORTS[0] + len(VALIDATOR_PORTS))

# CONFIG
LOSSES_DIR = os.getenv("LOSSES_DIR", "losses")
ACTIVATION_DIR = os.getenv("ACTIVATION_DIR", "activation_cache")
DEFAULT_API_BASE_URL = os.getenv("DEFAULT_API_BASE_URL", "http://localhost:8000")
MINER_REGISTRY_PATH = "./miner_registry.pkl"

if MOCK:
    logger.warning("RUNNING IN MOCK MODE!")

network = os.getenv("network", "test")
wallet_name = os.getenv("wallet_name", "test")
wallet_hotkey = os.getenv("wallet_hotkey", "m1")
MINER_HOTKEYS_ENV = os.getenv("MINER_HOTKEYS")
MINER_HOTKEYS = MINER_HOTKEYS_ENV.strip().split(",") if MINER_HOTKEYS_ENV is not None else [wallet_hotkey]
netuid = int(os.getenv("netuid", "9"))
__spec_version__ = 4064

# Weights & Biases
WANDB_PROJECT = "pretrain-test"
WANDB_ENTITY = "macrocosmos"
RUN_NAME = "local_3_bottlenecks_8_lr4schedulers2_5e-4_gclip"
WANDB_TOKEN = os.getenv("WANDB_TOKEN")
# MongoDB Settings
MONGO = os.getenv("MONGO") == "True"
MONGODB_PROTOCOL = os.getenv("MONGODB_PROTOCOL", "mongodb+srv")
MONGO_DB_USERNAME = os.getenv("MONGO_DB_USERNAME")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
MONGODB_DB_NAME = os.getenv("MONGO_DB_DATABASE")
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST")
MONGODB_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
LOAD_MOST_RECENT_ORCHESTRATOR_STATE_ON_INITIALIZATION = False
UPLOAD_EVERY_N_UPDATES = int(os.getenv("UPLOAD_EVERY_N_UPDATES", 10))

# Validator API whitelist
ORCHESTRATOR_KEY = os.getenv("ORCHESTRATOR_KEY", "5EWsttKe7eiV9eJ42HGoSpymN9END7zADs2PmHmjqcWXQ6Ty")

# FOR TESTING ONLY
ORCHESTRATOR_WALLET_NAME = os.getenv("ORCHESTRATOR_WALLET_NAME", None)
ORCHESTRATOR_WALLET_HOTKEY = os.getenv("ORCHESTRATOR_WALLET_HOTKEY", None)


MAX_ACTIVATION_CACHE_SIZE = 2

LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"
MINER_HEALTH_HOST = os.getenv("MINER_HEALTH_HOST", "0.0.0.0")
MINER_HEALTH_PORT = int(os.getenv("MINER_HEALTH_PORT", 9000))
MINER_HEALTH_ENDPOINT = os.getenv("MINER_HEALTH_ENDPOINT", "/health")

LOCAL_OPTIMIZER_STEPS = 2 if MOCK else 10
GLOBAL_OPTIMIZER_STEPS = 2 if MOCK else 10
BURN_FACTOR = 1.25  # 1-1/BurnFactor % is burned, for 5 it's 80%

LIMIT = f"{1*len(MINER_HOTKEYS)}/second" if MOCK else "1/second"
HIGH_LIMIT = f"{5*len(MINER_HOTKEYS)}/second" if MOCK else "10/second"

MINERS_REQUIRED_FOR_WEIGHT_UPLOADING = 0.8
ACTIVATIONS_REQUIRED_FOR_WEIGHT_MERGING_DECREASE_INTERVAL = (
    1 if MOCK else 6
)  # Decreases by 1%, so in this case takes 600 seconds = 10 minutes
PARTITIONS_REQUIRED_FOR_WEIGHT_MERGING_DECREASE_INTERVAL = 1 if MOCK else 6

MIN_ACTIVATION_SIZE = 100
MAX_ACTIVATION_SIZE = 1000000000

MIN_WEIGHT_SIZE = 100  # 100 bytes
MAX_WEIGHT_SIZE = 100000000000  # 100GB
