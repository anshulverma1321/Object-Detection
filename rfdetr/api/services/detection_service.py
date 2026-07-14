import os
import uuid
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image

from api.core.config import MAX_IMAGE_SIDE, MAX_VIDEO_SIDE, OUTPUT_DIR
from api.core.model import get_model
from api.utils.image_io import (
    cv2_to_pil,
    pil_to_cv2,
    resize_cv2_keep_aspect,
    resize_pil_keep_aspect,
)


def validate_threshold(threshold: float) -> float:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return threshold


def _call_model_predict(model: Any, image: Image.Image, threshold: float):
    """
    Supports multiple predict signatures:
        model.predict(image, threshold=0.5)
        model.predict(image)
        model(image)
    """
    try:
        return model.predict(image, threshold=threshold)
    except TypeError:
        try:
            return model.predict(image)
        except AttributeError:
            return model(image)


def detections_to_json(detections, scale_x: float = 1.0, scale_y: float = 1.0) -> List[Dict[str, Any]]:
    """
    Converts common detection output structures into JSON.
    Adjust this function only if your RF-DETR output is custom.
    """
    results = []

    # Supervision-like detections: xyxy, confidence, class_id
    if hasattr(detections, "xyxy"):
        boxes = detections.xyxy
        scores = getattr(detections, "confidence", None)
        class_ids = getattr(detections, "class_id", None)
        labels = getattr(detections, "data", {}).get("class_name") if hasattr(detections, "data") else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist() if hasattr(box, "tolist") else list(box)
            item = {
                "box": {
                    "x1": float(x1 * scale_x),
                    "y1": float(y1 * scale_y),
                    "x2": float(x2 * scale_x),
                    "y2": float(y2 * scale_y),
                }
            }
            if scores is not None:
                item["confidence"] = float(scores[i])
            if class_ids is not None:
                item["class_id"] = int(class_ids[i])
            if labels is not None and i < len(labels):
                item["label"] = str(labels[i])
            results.append(item)
        return results

    # Dict output
    if isinstance(detections, dict):
        if "detections" in detections and isinstance(detections["detections"], list):
            return detections_to_json(detections["detections"], scale_x, scale_y)
        return [{"raw_output": str(detections)}]

    # List output
    if isinstance(detections, list):
        for det in detections:
            if isinstance(det, dict):
                item = det.copy()
                box = item.get("box") or item.get("bbox") or item.get("xyxy")

                if box is not None:
                    if isinstance(box, dict):
                        item["box"] = {
                            "x1": float(box.get("x1", box.get("xmin", 0)) * scale_x),
                            "y1": float(box.get("y1", box.get("ymin", 0)) * scale_y),
                            "x2": float(box.get("x2", box.get("xmax", 0)) * scale_x),
                            "y2": float(box.get("y2", box.get("ymax", 0)) * scale_y),
                        }
                    elif isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                        item["box"] = {
                            "x1": float(box[0] * scale_x),
                            "y1": float(box[1] * scale_y),
                            "x2": float(box[2] * scale_x),
                            "y2": float(box[3] * scale_y),
                        }

                results.append(item)
            else:
                results.append({"raw_detection": str(det)})
        return results

    return [{"raw_output": str(detections)}]


def draw_detections(frame: np.ndarray, detections_json: List[Dict[str, Any]]) -> np.ndarray:
    annotated = frame.copy()

    for det in detections_json:
        box = det.get("box")
        if not box:
            continue

        if isinstance(box, dict):
            x1 = int(box.get("x1", 0))
            y1 = int(box.get("y1", 0))
            x2 = int(box.get("x2", 0))
            y2 = int(box.get("y2", 0))
        else:
            x1, y1, x2, y2 = map(int, box[:4])

        label = str(det.get("label", det.get("class_id", "object")))
        conf = det.get("confidence")
        if conf is not None:
            label += f" {float(conf):.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return annotated


def detect_image(image: Image.Image, threshold: float = 0.5, return_annotated: bool = False):
    threshold = validate_threshold(threshold)
    model = get_model()

    original_w, original_h = image.size
    processed, scale_x, scale_y = resize_pil_keep_aspect(image, MAX_IMAGE_SIDE)

    raw = _call_model_predict(model, processed, threshold)
    detections = detections_to_json(raw, scale_x=scale_x, scale_y=scale_y)

    response = {
        "input_type": "image",
        "original_resolution": {"width": original_w, "height": original_h},
        "processed_resolution": {"width": processed.size[0], "height": processed.size[1]},
        "threshold": threshold,
        "total_detections": len(detections),
        "detections": detections,
    }

    if return_annotated:
        frame = pil_to_cv2(image)
        annotated = draw_detections(frame, detections)
        filename = f"annotated_{uuid.uuid4().hex}.jpg"
        output_path = OUTPUT_DIR / filename
        cv2.imwrite(str(output_path), annotated)
        response["annotated_file"] = filename
        response["annotated_url"] = f"/detect/output/{filename}"

    return response


def detect_frame(frame: np.ndarray, threshold: float = 0.5):
    threshold = validate_threshold(threshold)
    model = get_model()

    original_h, original_w = frame.shape[:2]
    processed_frame, scale_x, scale_y = resize_cv2_keep_aspect(frame, MAX_VIDEO_SIDE)
    image = cv2_to_pil(processed_frame)

    raw = _call_model_predict(model, image, threshold)
    detections = detections_to_json(raw, scale_x=scale_x, scale_y=scale_y)

    return {
        "original_resolution": {"width": original_w, "height": original_h},
        "processed_resolution": {"width": processed_frame.shape[1], "height": processed_frame.shape[0]},
        "threshold": threshold,
        "total_detections": len(detections),
        "detections": detections,
    }
