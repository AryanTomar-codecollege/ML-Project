import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_PATH_RAW = os.getenv(
    "MODEL_PATH",
    "models/best_wound_model_v2.pth"
)
MODEL_PATH = Path(MODEL_PATH_RAW)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (ROOT / MODEL_PATH).resolve()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "openai/gpt-oss-120b"
).strip()

OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "").strip()
OMNIROUTE_BASE_URL = os.getenv(
    "OMNIROUTE_BASE_URL", "http://localhost:20128/v1"
).rstrip("/")
OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "auto").strip()

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "55"))
MIN_MARGIN = float(os.getenv("MIN_MARGIN", "12"))
IMAGE_SIZE = 224

# Must match the order stored in best_wound_model_v2.pth.
CLASS_NAMES = [
    "bruise",
    "burn",
    "cut",
    "no_injury",
    "not_known_image",
]

if LLM_PROVIDER not in {"groq", "omniroute"}:
    raise ValueError("LLM_PROVIDER must be 'groq' or 'omniroute'")
