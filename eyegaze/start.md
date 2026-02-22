# Eye Gaze setup and startup (Debian Linux)

This guide starts the `eyegaze` service in a local Python virtual environment.

## 1) Go to the package directory

```bash
cd /path/to/conduit/eyegaze
```

## 2) Create and activate a venv inside `eyegaze`

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 4) Run the eye-gaze service

```bash
python eye_tracker_service.py --camera 0 --http-host 127.0.0.1 --http-port 8767 --http-streaming true
```

If camera 0 is not your device, try:

```bash
python eye_tracker_service.py --camera 1 --http-host 127.0.0.1 --http-port 8767 --http-streaming true
```

Optional debug mode:

```bash
python eye_tracker_service.py --debug --camera 0 --http-streaming true
```

## 5) Verify service is running

```bash
curl http://127.0.0.1:8767/status
curl http://127.0.0.1:8767/cv
```

Video stream URL:

```text
http://127.0.0.1:8767/video
```

## 6) Stop

Press `Ctrl + C` in the terminal running the service, then:

```bash
deactivate
```

