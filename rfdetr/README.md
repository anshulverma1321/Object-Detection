# 🚀 RF-DETR FastAPI Wrapper

A production-ready **FastAPI wrapper** for the **RF-DETR Object Detection** model with support for images, videos, webcams, and live RTSP/IP camera streams.

## ✨ Features

- 📷 Image Detection (`jpg`, `jpeg`, `png`, `bmp`, `webp`, `tiff`, `tif`)
- 🎥 Video Detection (`mp4`, `avi`, `mov`, `mkv`)
- 📡 RTSP / RTMP / IP Camera Stream Support
- 🎦 Webcam Inference
- 🖥️ Supports 480p, 720p, 1080p, 2K, 4K & 8K Inputs
- ⚡ Automatic Image Resize & Optimization
- 🎯 Original Resolution Bounding Box Scaling
- 📝 Annotated Image & Video Output
- 📊 REST API with Swagger Documentation
- 🚀 FastAPI + Uvicorn Server
- 🔍 Health Check Endpoint

---

# 📂 Project Structure

```text
RF-DETR-API/
│
├── api/
│   ├── core/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   ├── utils/
│   └── main.py
│
├── outputs/
│
├── utils/
│   └── inference.py
│
├── api.py
├── run_api.py
├── test_requests.py
├── requirements.txt
├── requirements_api.txt
├── start_api_windows.bat
├── README.md
└── test.webp
```

---

# ⚙️ Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- RF-DETR Model
- PyTorch

Install all dependencies:

```bash
pip install -r requirements_api.txt
```

or

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the API

### Option 1 (Recommended)

```bash
python run_api.py
```

### Option 2

```bash
uvicorn api.main:app --reload
```

### Option 3 (Windows)

Double-click:

```text
start_api_windows.bat
```

---

# 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Swagger UI |
| `/health` | GET | Health Check |
| `/detect` | POST | Detect Objects in Images |
| `/detect/video` | POST | Video Detection |
| `/detect/stream` | POST | RTSP/IP Camera Detection |
| `/detect/webcam` | POST | Webcam Detection |
| `/detect/live` | GET | Live MJPEG Stream |

---

# 📷 Image Detection

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
-F "file=@image.jpg" \
-F "threshold=0.5" \
-F "return_annotated=true"
```

### Python

```bash
python test_requests.py --image image.jpg
```

---

# 🎥 Video Detection

```bash
curl -X POST "http://127.0.0.1:8000/detect/video" \
-F "file=@video.mp4" \
-F "threshold=0.5" \
-F "return_annotated=true"
```

---

# 📡 RTSP / IP Camera

```bash
curl -X POST "http://127.0.0.1:8000/detect/stream" \
-H "Content-Type: application/json" \
-d "{\"source\":\"rtsp://username:password@192.168.1.10:554/stream1\",\"threshold\":0.5}"
```

---

# 🎦 Webcam Detection

```bash
curl -X POST "http://127.0.0.1:8000/detect/webcam?camera_index=0"
```

Live Browser Preview

```
http://127.0.0.1:8000/detect/live?source=0
```

---

# 📊 Supported Input Formats

## Images

- JPG
- JPEG
- PNG
- BMP
- WEBP
- TIFF
- TIF

## Videos

- MP4
- AVI
- MOV
- MKV

---

# 🖥️ High Resolution Support

The API safely processes:

- ✅ 480p
- ✅ 720p
- ✅ 1080p
- ✅ 2K
- ✅ 4K
- ✅ 8K

Workflow:

```
Original Image
        │
        ▼
Automatic Resize
        │
        ▼
RF-DETR Prediction
        │
        ▼
Scale Bounding Boxes
        │
        ▼
Annotated Output
```

---

# 📁 Output

Processed files are automatically saved inside:

```text
outputs/
```

Depending on the request, the API returns:

- Detection JSON
- Annotated Images
- Annotated Videos

---

# 🛠 Configuration

Default resize values:

```text
MAX_IMAGE_SIDE = 1920
MAX_VIDEO_SIDE = 1280
```

These values can be modified according to your hardware capabilities.

---

# ⚠️ Common Issue

### ModuleNotFoundError

```
No module named 'utils.inference'
```

Ensure your project contains:

```text
utils/
└── inference.py
```

with a class similar to:

```python
class RFDETRInference:
    ...
```

If your inference file has a different name, update the import inside:

```
api/core/model.py
```

---

# 📖 API Documentation

After starting the server, open:

```
http://127.0.0.1:8000/docs
```

Swagger UI provides interactive API testing for all endpoints.

---

# ❤️ Built With

- FastAPI
- Uvicorn
- PyTorch
- OpenCV
- NumPy
- RF-DETR

---

# 📜 License

This project is intended for research, learning, and production deployment with RF-DETR-based object detection systems.
