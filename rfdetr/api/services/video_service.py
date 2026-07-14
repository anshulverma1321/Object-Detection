import uuid

import cv2
import numpy as np

from fastapi import HTTPException

from api.core.config import MAX_VIDEO_FRAMES, OUTPUT_DIR, STREAM_FRAME_SKIP
from api.services.detection_service import detect_frame, draw_detections


def process_video_or_stream(
    source,
    threshold: float = 0.5,
    max_frames: int = MAX_VIDEO_FRAMES,
    return_annotated_video: bool = False,
):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    output_filename = None

    if return_annotated_video:
        output_filename = f"annotated_video_{uuid.uuid4().hex}.mp4"
        output_path = OUTPUT_DIR / output_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frames = []
    frame_index = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if processed >= max_frames:
            break

        if frame_index % STREAM_FRAME_SKIP != 0:
            frame_index += 1
            continue

        result = detect_frame(frame, threshold=threshold)
        result["frame_index"] = frame_index
        frames.append(result)

        if writer is not None:
            annotated = draw_detections(frame, result["detections"])
            writer.write(annotated)

        frame_index += 1
        processed += 1

    cap.release()
    if writer is not None:
        writer.release()

    response = {
        "input_type": "video_or_stream",
        "source_resolution": {"width": width, "height": height},
        "fps": fps,
        "processed_frames": processed,
        "frame_skip": STREAM_FRAME_SKIP,
        "threshold": threshold,
        "frames": frames,
    }

    if output_filename:
        response["annotated_file"] = output_filename
        response["annotated_url"] = f"/detect/output/{output_filename}"

    return response


def mjpeg_generator(source, threshold: float = 0.5):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % STREAM_FRAME_SKIP == 0:
            result = detect_frame(frame, threshold=threshold)
            frame = draw_detections(frame, result["detections"])

        ok, encoded = cv2.imencode(".jpg", frame)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )

        frame_index += 1

    cap.release()
