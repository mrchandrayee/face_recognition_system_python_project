from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
FACE_REG_DIR = BASE_DIR / "registered_faces"
CACHE_FILE = BASE_DIR / "face_cache.pkl"
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
DET_SIZE = (640, 640)
SIMILARITY_THRESHOLD = 0.45
