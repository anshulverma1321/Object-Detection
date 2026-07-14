import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from api.core.config import OUTPUT_DIR
from api.core.segmentation_model import get_segmentation_model


def segment_image(image):
    """
    Runs Mask2Former segmentation and saves result.
    """

    model = get_segmentation_model()

    segmented = model.segment(image)

    filename = f"segment_{uuid.uuid4().hex}.png"

    output_path = Path(OUTPUT_DIR) / filename

    cv2.imwrite(str(output_path), segmented)

    return {
        "input_type": "image",
        "annotated_file": filename,
        "annotated_url": f"/segment/output/{filename}",
    }