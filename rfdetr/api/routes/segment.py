import os

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from api.core.config import OUTPUT_DIR
from api.services.segmentation_service import segment_image
from api.utils.files import is_image_file, save_upload_temp
from api.utils.image_io import normalize_image

router = APIRouter()


@router.post("")
async def segment_upload(
    file: UploadFile = File(...)
):
    filename = file.filename or ""

    temp_path = None

    try:

        if not is_image_file(filename):
            raise HTTPException(
                status_code=400,
                detail="Only image files are supported."
            )

        temp_path = save_upload_temp(file)

        image = Image.open(temp_path)

        image = normalize_image(image)

        result = segment_image(image)

        return JSONResponse(result)

    finally:

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.get("/output/{filename}")
def get_output(filename: str):

    path = OUTPUT_DIR / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Output file not found"
        )

    return FileResponse(str(path))