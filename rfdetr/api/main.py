from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
from fastapi.responses import StreamingResponse
from api.core.model import load_model, get_model_status
from api.routes.detect import router as detect_router
from api.routes.segment import router as segment_router

app = FastAPI(
    title="RF-DETR Detection API",
    description="API wrapper for RF-DETR supporting images, videos, live streams, webcams and high-resolution inputs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.core.segmentation_model import load_segmentation_model


@app.on_event("startup")
def startup_event():
    load_model()
    load_segmentation_model()
    
    
@app.get("/")
def root():
    return {
        "message": "RF-DETR API is running",
        "docs": "/docs",
        "health": "/health",
        "supported_inputs": {
            "images": ["jpg", "jpeg", "png", "bmp", "webp", "tiff", "tif"],
            "videos": ["mp4", "avi", "mov", "mkv"],
            "streams": ["rtsp", "rtmp", "webcam index", "ip camera url"],
            "resolutions": ["480p", "720p", "1080p", "2K", "4K", "8K"],
        },
        "endpoints": {
            "upload_image_or_video": "POST /detect",
            "base64_image": "POST /detect/base64",
            "image_url": "POST /detect/url",
            "stream_or_ip_camera": "POST /detect/stream",
            "webcam": "POST /detect/webcam",
            "live_mjpeg": "GET /detect/live?source=0",
            "output_file": "GET /detect/output/{filename}",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": get_model_status(),
    }


app.include_router(detect_router, prefix="/detect", tags=["Detection"])

from api.routes.detect import router as detect_router
from api.routes.segment import router as segment_router


app.include_router(
    detect_router,
    prefix="/detect",
    tags=["Detection"]
)

app.include_router(
    segment_router,
    prefix="/segment",
    tags=["Segmentation"]
)

def mjpeg_generator(source: str, threshold: float = 0.5):
    from api.core.model import get_model

    model_wrapper = get_model()

    # webcam source agar "0", "1" string me aaye to int bana do
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[STREAM ERROR] Could not open source: {source}")
        return

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        try:
            # speed ke liye har 3rd frame detect
            if frame_count % 3 == 0:
                detections = model_wrapper.predict(frame, threshold=threshold)

                # agar supervision detections format hai
                if hasattr(detections, "xyxy"):
                    boxes = detections.xyxy
                    confidences = getattr(detections, "confidence", [])
                    class_ids = getattr(detections, "class_id", [])

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)

                        conf = ""
                        if len(confidences) > i:
                            conf = f"{float(confidences[i]):.2f}"

                        cls = "object"
                        if len(class_ids) > i:
                            cls = str(int(class_ids[i]))

                        label = f"{cls} {conf}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

        except Exception as e:
            print("[STREAM DETECTION ERROR]", e)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

        frame_count += 1

    cap.release()


@app.get("/stream/mjpeg")
def stream_mjpeg(source: str = "0", threshold: float = 0.5):
    return StreamingResponse(
        mjpeg_generator(source, threshold),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )