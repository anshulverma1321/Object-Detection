import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image

from api.core.config import OUTPUT_DIR, MAX_VIDEO_FRAMES
from api.schemas.requests import Base64ImageRequest, StreamRequest, URLImageRequest
from api.services.detection_service import detect_image
from api.services.video_service import mjpeg_generator, process_video_or_stream
from api.utils.files import is_image_file, is_video_file, save_upload_temp
from api.utils.image_io import decode_base64_image, load_image_from_url, normalize_image

router = APIRouter()


@router.post("")
async def detect_upload(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    return_annotated: bool = Form(False),
    max_frames: int = Form(MAX_VIDEO_FRAMES),
):
    filename = file.filename or ""

    temp_path = None
    try:
        if is_image_file(filename):
            temp_path = save_upload_temp(file)
            image = Image.open(temp_path)
            image = normalize_image(image)
            return JSONResponse(detect_image(image, threshold, return_annotated))

        if is_video_file(filename):
            temp_path = save_upload_temp(file)
            return JSONResponse(
                process_video_or_stream(
                    source=temp_path,
                    threshold=threshold,
                    max_frames=max_frames,
                    return_annotated_video=return_annotated,
                )
            )

        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use image: jpg/jpeg/png/bmp/webp/tiff or video: mp4/avi/mov/mkv",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.post("/base64")
def detect_base64(req: Base64ImageRequest):
    image = decode_base64_image(req.image_base64)
    return JSONResponse(detect_image(image, req.threshold, req.return_annotated))


@router.post("/url")
def detect_url(req: URLImageRequest):
    image = load_image_from_url(req.url)
    return JSONResponse(detect_image(image, req.threshold, req.return_annotated))


@router.post("/stream")
def detect_stream(req: StreamRequest):
    return JSONResponse(
        process_video_or_stream(
            source=req.source,
            threshold=req.threshold,
            max_frames=req.max_frames,
            return_annotated_video=req.return_annotated_video,
        )
    )


@router.post("/webcam")
def detect_webcam(
    camera_index: int = 0,
    threshold: float = 0.5,
    max_frames: int = 100,
    return_annotated_video: bool = False,
):
    return JSONResponse(
        process_video_or_stream(
            source=camera_index,
            threshold=threshold,
            max_frames=max_frames,
            return_annotated_video=return_annotated_video,
        )
    )


@router.get("/live")
def live_mjpeg(source: str = "0", threshold: float = 0.5):
    if source.isdigit():
        source = int(source)

    return StreamingResponse(
        mjpeg_generator(source, threshold=threshold),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/output/{filename}")
def get_output(filename: str):
    path = OUTPUT_DIR / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(str(path))
