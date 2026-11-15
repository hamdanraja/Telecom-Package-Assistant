import os
from pathlib import Path

NEAR_TOLERANCE = float(os.getenv("NEAR_TOLERANCE", "0.25"))
DATA_BASIS = os.getenv("DATA_BASIS", "internet")

BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")   # must match `ollama list`
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
THINK_FINAL = os.getenv("THINK_FINAL", "0") == "1"

OPERATOR_FILES = {
    "jazz": "jazz.csv",
    "zong": "zong.csv",
    "ufone": "ufone.csv",
}
