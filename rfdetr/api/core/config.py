import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

RFDETR_SIZE = os.getenv("RFDETR_SIZE", "M")
RFDETR_CONFIG_PATH = os.getenv("RFDETR_CONFIG_PATH", "configs/config.json")

MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1920"))
MAX_VIDEO_SIDE = int(os.getenv("MAX_VIDEO_SIDE", "1280"))
MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "300"))
STREAM_FRAME_SKIP = int(os.getenv("STREAM_FRAME_SKIP", "2"))

OUTPUT_DIR = ROOT_DIR / os.getenv("OUTPUT_DIR", "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"}
ALLOWED_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv"}
