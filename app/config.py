from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
QDRANT_DIR = ROOT_DIR / "data" / "qdrant"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "BAAI/bge-m3"
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
QDRANT_PATH = os.getenv("QDRANT_PATH", str(QDRANT_DIR))
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 60
RETRIEVAL_TOP_K = 8
ANSWER_TOP_K = 5
LOW_CONFIDENCE_THRESHOLD = 0.55

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)
