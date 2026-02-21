#!/usr/bin/env python3

from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import os
import typing as _t
import sys
import threading

import mediapipe as mp


class MediaPipeFaceMeshBackend:
    """Small wrapper around the two supported mediapipe runtimes."""

    def __init__(self, args: _t.Any, *, debug: bool = False) -> None:
        self.args = args
        self.debug = debug
        self._backend = "unknown"
        self.face_mesh = None
        self._face_landmarker = None
        self._tasks_image = None
        self._tasks_image_format = None
        self._running_mode = None
        self._tasks_timestamp_ms = 0

        self._init_face_mesh_backend()

    @property
    def is_solutions(self) -> bool:
        return self._backend == "solutions"

    def _init_face_mesh_backend(self) -> None:
        def _create_face_landmarker(options: _t.Any) -> _t.Any:
            return face_landmarker.FaceLandmarker.create_from_options(options)

        def _create_with_timeout(options: _t.Any) -> Any:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_create_face_landmarker, options)
                try:
                    return future.result(timeout=5.5)
                except concurrent.futures.TimeoutError as exc:
                    raise RuntimeError(
                        "FaceLandmarker initialization timed out on this runtime. "
                        "Try a clean Python 3.11 + mediapipe install or reinstall `opencv-python`/`mediapipe`."
                    ) from exc
                except Exception as exc:
                    raise RuntimeError(f"FaceLandmarker initialization failed: {exc}") from exc

        if hasattr(mp, "solutions"):
            self._backend = "solutions"
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if hasattr(mp, "tasks"):
            if sys.version_info >= (3, 14) and sys.platform == "darwin":
                raise RuntimeError(
                    "Mediapipe face tasks on macOS are currently unstable on Python 3.14 in "
                    "this environment. Please use Python 3.11 for this build."
                )

            try:
                from mediapipe.tasks.python.core.base_options import BaseOptions
                from mediapipe.tasks.python.vision import face_landmarker
                from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
                    VisionTaskRunningMode,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Mediapipe tasks backend is present but required symbols are missing: "
                    f"{exc}"
                ) from exc

            model_path = self.args.face_landmarker_task
            if not model_path:
                raise RuntimeError(
                    "Mediapipe 'tasks' package is installed, but no task model file is configured. "
                    "Pass --face-landmarker-task /path/to/face_landmarker.task"
                )
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Face landmarker task model not found: {model_path}. "
                    "Download a Mediapipe FaceLandmarker task model and pass its path."
                )

            self._backend = "tasks"
            self._tasks_image = Image
            self._tasks_image_format = ImageFormat
            self._running_mode = VisionTaskRunningMode.VIDEO
            cpu_options = face_landmarker.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
                running_mode=self._running_mode,
                min_tracking_confidence=0.5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                num_faces=1,
            )
            try:
                self._face_landmarker = _create_with_timeout(cpu_options)
            except Exception as cpu_error:
                # Fallback to Mediapipe default delegate for environments where CPU enum is unsupported.
                default_options = face_landmarker.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=self._running_mode,
                    min_tracking_confidence=0.5,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    num_faces=1,
                )
                try:
                    self._face_landmarker = _create_with_timeout(default_options)
                except Exception as default_error:
                    err_msg = str(default_error)
                    if (
                        "NSOpenGLPixelFormat" in err_msg
                        or "kGpuService" in err_msg
                        or "could not create an NSOpenGLPixelFormat" in err_msg
                    ):
                        raise RuntimeError(
                            "Mediapipe task graph cannot start GPU service in this runtime "
                            "(Could not create NSOpenGLPixelFormat). "
                            "On macOS this is usually fixed by running in Python 3.11 with a fresh mediapipe install "
                            "that provides the full runtime. Then run: "
                            "python3.11 -m venv .venv && source .venv/bin/activate "
                            "&& pip install -r eyegaze/requirements.txt"
                        ) from default_error

                    raise RuntimeError(
                        "Failed to initialize FaceLandmarker with CPU delegate. "
                        "This build of mediapipe may require a different Python/runtime combo "
                        "(cp311 wheels are known to work better than cp314)."
                    ) from default_error
            return

        raise AttributeError("Mediapipe SDK has neither 'solutions' nor 'tasks'.")

    def run_face_mesh(self, frame_rgb) -> _t.Any:
        if self._backend == "solutions":
            return self.face_mesh.process(frame_rgb)

        self._tasks_timestamp_ms += 16
        mp_image = self._tasks_image(image_format=self._tasks_image_format.SRGB, data=frame_rgb)
        if self._face_landmarker is None:
            raise RuntimeError("Tasks backend not initialized")
        return self._face_landmarker.detect_for_video(mp_image, int(self._tasks_timestamp_ms))

    def get_face_landmarks(self, results: _t.Any) -> _t.Optional[_t.Any]:
        if self._backend == "solutions":
            landmarks = getattr(results, "multi_face_landmarks", None)
            return landmarks[0].landmark if landmarks else None

        landmarks = getattr(results, "face_landmarks", None)
        if not landmarks:
            return None
        return landmarks[0]

    def close(self) -> None:
        if self._backend == "solutions" and self.face_mesh is not None:
            self.face_mesh.close()
        elif self._face_landmarker is not None and hasattr(self._face_landmarker, "close"):
            self._face_landmarker.close()
