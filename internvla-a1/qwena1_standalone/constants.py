from pathlib import Path

from huggingface_hub.constants import HF_HOME

ACTION = "action"
OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38

DEFAULT_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_COSMOS_REPO = "nvidia/Cosmos-Tokenizer-CI8x8"
DEFAULT_COSMOS_DIR = Path(HF_HOME) / "hub" / "Cosmos-Tokenizer-CI8x8"
