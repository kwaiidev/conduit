# ASL Sign Language Recognition

Real-time American Sign Language (A-Z) detection using MediaPipe and deep learning.

## Quick Start

### 1. Setup Environment

**Option A – Conda (Windows):**

```bash
conda env create -f environment/deployment.yml
conda activate asl-deployment
```

**Option B – Linux / macOS (venv + pip):**

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements_linux.txt
```

### 2. Model weights

The app expects a trained weights file at `data/weights/asl_crop_v4_1_mobilenet_weights.pth`. If that file is missing:

- **Train your own:** use the scripts in `train/` (e.g. `train_evaluate.py`) with your ASL dataset, then copy the best `.pth` to `data/weights/asl_crop_v4_1_mobilenet_weights.pth`.
- Or obtain the pre-trained weights from the project maintainers and place them in `data/weights/`.

### 3. Run the Desktop App

**Run from the project root** (the `Sign-Language-Recognition` folder), otherwise Python won’t find the `app` module:

```bash
cd /path/to/Sign-Language-Recognition
source .venv/bin/activate
python -m app.frame
```

On Linux, if you get `No module named 'tkinter'`, install it:

```bash
sudo apt install python3-tk
```

### 4. Run the API Server

From the project root:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Then open:

- Video stream: http://localhost:8000/video
- API docs: http://localhost:8000/docs

## API Endpoints

| Endpoint    | Method | Description                    |
| ----------- | ------ | ------------------------------ |
| `/predict`  | GET    | Get current detected letter    |
| `/sentence` | GET    | Get accumulated sentence       |
| `/video`    | GET    | Live video stream              |
| `/add`      | POST   | Add current letter to sentence |
| `/space`    | POST   | Add space                      |
| `/reset`    | POST   | Clear sentence                 |

## Project Structure

```
├── app/           # Application code
│   ├── frame.py   # Tkinter GUI
│   ├── api.py     # FastAPI server
│   └── frame_utils.py
├── model/         # CNN architectures
├── train/         # Training scripts
├── data/weights/  # Trained models
└── utils/         # Helper functions
```

## How It Works

1. Webcam captures hand gestures
2. MediaPipe detects 21 hand landmarks
3. MobileNetV2 classifies the sign (A-Z)
4. Predictions are smoothed over 10 frames
5. Letters build into sentences

## Requirements

- Python 3.10+
- Webcam
- CUDA GPU (optional, for faster inference)
