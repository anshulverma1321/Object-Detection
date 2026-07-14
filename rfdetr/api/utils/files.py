import shutil
import tempfile
from pathlib import Path

from fastapi import UploadFile

from api.core.config import ALLOWED_IMAGE_EXTS, ALLOWED_VIDEO_EXTS


def get_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower().strip() if "." in filename else ""


def is_image_file(filename: str) -> bool:
    return get_ext(filename) in ALLOWED_IMAGE_EXTS


def is_video_file(filename: str) -> bool:
    return get_ext(filename) in ALLOWED_VIDEO_EXTS


def save_upload_temp(upload_file: UploadFile) -> str:
    suffix = "." + get_ext(upload_file.filename or "upload")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    with tmp as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return tmp.name


def safe_filename(path: str) -> str:
    return Path(path).name
