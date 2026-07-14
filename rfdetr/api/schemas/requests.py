from pydantic import BaseModel, Field


class Base64ImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image. Data URL is also accepted.")
    threshold: float = 0.5
    return_annotated: bool = False


class URLImageRequest(BaseModel):
    url: str
    threshold: float = 0.5
    return_annotated: bool = False


class StreamRequest(BaseModel):
    source: str = Field(..., description="RTSP, RTMP, IP camera URL, video URL")
    threshold: float = 0.5
    max_frames: int = 100
    return_annotated_video: bool = False
