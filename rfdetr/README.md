# RF-DETR API Perfect Wrapper

This ZIP contains a modular FastAPI wrapper for your RF-DETR notebook/model.

It supports:

- Images: `jpg`, `jpeg`, `png`, `bmp`, `webp`, `tiff`, `tif`
- Videos: `mp4`, `avi`, `mov`, `mkv`
- Live streams: RTSP, RTMP, webcam, IP camera
- Resolutions: 480p, 720p, 1080p, 2K, 4K, 8K
- Automatic resize/compression before inference
- Original-size bounding box scaling
- Annotated image/video output
- Swagger UI at `/docs`

---

## Correct folder structure

Extract/copy the contents of this ZIP directly inside your RF-DETR project root.

Final structure should look like this:

```text
rfdetr/
├── api/
│   ├── main.py
│   ├── core/
│   ├── routes/
│   ├── services/
│   ├── schemas/
│   └── utils/
├── run_api.py
├── api.py
├── requirements_api.txt
├── start_api_windows.bat
├── test_requests.py
├── outputs/
│
├── utils/
│   └── inference.py      <-- your original RF-DETR inference file
├── configs/
│   └── config.json       <-- your original config
└── rfdetr_pipeline.ipynb
```

Important:

There are two different `utils` folders:

```text
api/utils/         -> API helper files
utils/inference.py -> your original RF-DETR model wrapper
```

The API expects your notebook import to work:

```python
from utils.inference import RFDETRInference
```

If this file is missing, the server will still start, but `/detect` will return a clear model loading error.

---

## Install

```powershell
cd C:\Users\aj985\OneDrive\Desktop\rfdetr
python -m pip install -r requirements_api.txt
```

If you already use a virtual environment, activate it first.

---

## Run

Recommended:

```powershell
python run_api.py
```

Alternative:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or double-click:

```text
start_api_windows.bat
```

Open:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

## Test image

PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect" `
  -F "file=@test.jpg" `
  -F "threshold=0.5" `
  -F "return_annotated=true"
```

Python:

```powershell
python test_requests.py --image test.jpg
```

---

## Test video

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect" `
  -F "file=@video.mp4" `
  -F "threshold=0.5" `
  -F "max_frames=100" `
  -F "return_annotated=true"
```

---

## Test RTSP / IP camera

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect/stream" `
  -H "Content-Type: application/json" `
  -d "{\"source\":\"rtsp://username:password@192.168.1.10:554/stream1\",\"threshold\":0.5,\"max_frames\":100}"
```

---

## Test webcam

```powershell
curl.exe -X POST "http://127.0.0.1:8000/detect/webcam?camera_index=0&threshold=0.5&max_frames=100"
```

Live MJPEG browser preview:

```text
http://127.0.0.1:8000/detect/live?source=0&threshold=0.5
```

---

## Why 4K/8K will not crash directly

The API does not feed full 4K/8K images directly to the model.

Flow:

```text
Original 4K/8K image
↓
Resize to safe model size
↓
Run RF-DETR prediction
↓
Scale detection boxes back to original resolution
↓
Return JSON
```

Config defaults:

```text
MAX_IMAGE_SIDE=1920
MAX_VIDEO_SIDE=1280
```

You can change them in `.env`.

---

## Common issue: No module named utils.inference

This means your original RF-DETR inference file is missing.

Expected file:

```text
rfdetr/utils/inference.py
```

Expected class:

```python
class RFDETRInference:
    ...
```

Expected predict call:

```python
model.predict(image, threshold=0.5)
```

If your actual model file/class has another name, edit:

```text
api/core/model.py
```

and change the loader.

---

## PowerShell profile error fix

If PowerShell shows an error like:

```text
Microsoft.PowerShell_profile.ps1
series1=pd.Series([2,4,6,8,10,12])
```

You pasted Python code into PowerShell profile by mistake.

Fix:

```powershell
notepad $PROFILE
```

Delete the Python line, save, close PowerShell, reopen it.
