import base64
import tempfile
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps

from fastapi import HTTPException


def normalize_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    image = normalize_image(image)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        data = base64.b64decode(image_base64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(data)
        tmp.close()

        image = Image.open(tmp.name)
        return normalize_image(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


def load_image_from_url(url: str) -> Image.Image:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urlopen(req, timeout=15)
        image = Image.open(response)
        return normalize_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load image URL: {e}")


def resize_pil_keep_aspect(image: Image.Image, max_side: int):
    original_w, original_h = image.size
    largest = max(original_w, original_h)

    if largest <= max_side:
        return image, 1.0, 1.0

    scale = max_side / largest
    new_w = max(1, int(original_w * scale))
    new_h = max(1, int(original_h * scale))

    resized = image.resize((new_w, new_h), Image.LANCZOS)

    scale_x = original_w / new_w
    scale_y = original_h / new_h

    return resized, scale_x, scale_y


def resize_cv2_keep_aspect(frame: np.ndarray, max_side: int):
    h, w = frame.shape[:2]
    largest = max(w, h)

    if largest <= max_side:
        return frame, 1.0, 1.0

    scale = max_side / largest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    scale_x = w / new_w
    scale_y = h / new_h

    return resized, scale_x, scale_y
