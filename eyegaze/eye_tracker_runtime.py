from __future__ import annotations

import time
import traceback
import math
from typing import Any, Optional

import cv2
import numpy as np

from filters import now_ms
from gaze_geometry import (
    LEFT_EYE_EAR_INDEXES,
    LEFT_EYE_HORIZ,
    RIGHT_EYE_EAR_INDEXES,
    RIGHT_EYE_HORIZ,
    compute_eye_aspect_ratio,
)


class EyeTrackerRuntimeMixin:
    def run(self) -> None:
        self.event_bus.start()
        self._start_http_server()
        print(
            "[Tracker] Eye tracker running. Press Q or - to quit | "
            f"invert_x={'on' if self.args.invert_gaze_x else 'off'}, "
            f"invert_y={'on' if self.args.invert_gaze_y else 'off'} | "
            f"legacy_3d_invert_both={'on' if self.args.legacy_3d_invert_both else 'off'} | "
            f"cursor_mode={self.args.cursor_mode}"
        )
        self._startup_log(
            f"startup_debug_enabled interval_s={self._startup_debug_interval_s:.2f} backend={self._camera_backend}",
            force=True,
        )

        if self.args.debug:
            for window_name in ("Integrated Eye Tracking", "Head/Eye Debug"):
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                except cv2.error:
                    pass

        last_toggle = 0.0
        last_calibration_key = 0.0
        last_runtime_heartbeat = 0.0
        loop_count = 0

        def _handle_runtime_controls(last_toggle_state: float) -> tuple[bool, float, int]:
            key_local = self._read_key()
            if self._is_key_down("f7"):
                if time.time() - last_toggle_state > 0.35:
                    self.mouse_control_enabled = not self.mouse_control_enabled
                    print(f"[Mouse Control] {'Enabled' if self.mouse_control_enabled else 'Disabled'}")
                    last_toggle_state = time.time()
                    time.sleep(0.1)
            should_quit = key_local in (ord("q"), ord("Q"), 27, ord("-"), ord("_")) or self._is_key_down("q")
            return should_quit, last_toggle_state, key_local

        running = True
        while running:
            loop_count += 1
            heartbeat_now = time.time()
            if self._is_startup_debug() and heartbeat_now - last_runtime_heartbeat >= self._startup_debug_interval_s:
                self._startup_log(
                    "loop_alive "
                    f"iter={loop_count} ready={self._camera_ready} status={self._camera_status} "
                    f"fail={self._camera_fail_streak} invalid={self._camera_invalid_streak} stale={self._camera_stale_streak} "
                    f"recoveries={self._camera_recovery_count} stats=[{self._camera_stats_summary()}]",
                    force=True,
                )
                last_runtime_heartbeat = heartbeat_now

            if self.cap is None:
                self._emit_noop("camera_unavailable")
                self._set_camera_issue("camera_unavailable")
                self._recover_camera_capture("capture_none")
                status_frame = self._camera_status_frame(
                    "Camera unavailable",
                    "Trying to recover camera stream",
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._camera_fail_streak += 1
                self._emit_noop("camera_frame_missing")
                self._set_camera_issue("frame_missing")
                if self._camera_fail_streak >= self._camera_recover_after_failures:
                    self._recover_camera_capture("missing_frames")
                    self._camera_fail_streak = 0
                status_frame = self._camera_status_frame(
                    "Camera initializing",
                    "Waiting for readable frame",
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            frame_ok, frame_reason = self._camera_frame_quality(frame)
            if not frame_ok:
                self._camera_invalid_streak += 1
                self._emit_noop(f"camera_frame_invalid:{frame_reason}")
                self._set_camera_issue(frame_reason)
                if frame_reason == "frame_stale" or self._camera_invalid_streak >= self._camera_recover_after_invalid:
                    self._recover_camera_capture(frame_reason)
                    self._camera_invalid_streak = 0
                status_frame = self._camera_status_frame(
                    "Stabilizing camera",
                    frame_reason.replace("_", " "),
                )
                self._publish_http_frame(status_frame)
                if self.args.debug:
                    cv2.imshow("Integrated Eye Tracking", status_frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            self._mark_camera_frame_valid(frame)

            if not self._cv_processing_enabled:
                if self.args.debug:
                    cv2.putText(
                        frame,
                        "CV processing disabled",
                        (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )
                self._publish_http_frame(frame)
                should_quit, last_toggle, _ = _handle_runtime_controls(last_toggle)
                if should_quit:
                    running = False
                continue

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._run_face_mesh(frame_rgb)
            face_landmarks = self._get_face_landmarks(results)

            def _frame_safe_point(ix: int, iy: int) -> tuple[int, int]:
                return (int(max(0, min(w - 1, ix))), int(max(0, min(h - 1, iy))))

            def _frame_safe_circle(cx: int, cy: int, radius: int, color: tuple[int, int, int], thickness: int) -> None:
                cx, cy = _frame_safe_point(cx, cy)
                rr = max(0, int(radius))
                if rr <= 0:
                    cv2.circle(frame, (cx, cy), 0, color, thickness)
                    return
                rr = min(rr, max(0, w - 1), max(0, h - 1))
                if rr <= 0:
                    return
                cv2.circle(frame, (cx, cy), rr, color, thickness)

            avg_combined_direction = None
            left_sphere_world = None
            right_sphere_world = None
            scaled_radius_l = None
            scaled_radius_r = None
            iris_3d_left = None
            iris_3d_right = None

            if face_landmarks is not None:
                try:
                    left_ear = compute_eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_INDEXES)
                    right_ear = compute_eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_INDEXES)
                except Exception:
                    left_ear = 1.0
                    right_ear = 1.0
                eyes_open = not (left_ear < self._blink_threshold and right_ear < self._blink_threshold)
                if not eyes_open:
                    self._emit_noop("blink")

                try:
                    head_center, R_final, nose_points_3d = self._compute_and_draw_coordinate_box(
                        frame,
                        face_landmarks,
                        self.nose_indices,
                        self.R_ref_nose,
                        color=(0, 255, 0),
                        size=80,
                        w=w,
                        h=h,
                    )
                except Exception as exc:
                    self._emit_noop(f"pose_unavailable:{exc}")
                    head_center, R_final = None, None
                    nose_points_3d = None

                if head_center is not None and R_final is not None:
                    self._latest_head_center = head_center.copy()
                    self._latest_rotation = R_final.copy()
                    self._latest_nose_scale = self.compute_scale(nose_points_3d) if nose_points_3d is not None else None
                    left_iris = face_landmarks[468]
                    right_iris = face_landmarks[473]
                    x_iris_l = int(left_iris.x * w)
                    y_iris_l = int(left_iris.y * h)
                    x_iris_r = int(right_iris.x * w)
                    y_iris_r = int(right_iris.y * h)
                    iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w], dtype=float)
                    iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w], dtype=float)

                    if not self.left_sphere_locked:
                        _frame_safe_circle(x_iris_l, y_iris_l, 10, (255, 25, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio = current_nose_scale / self.left_calibration_nose_scale if self.left_calibration_nose_scale else 1.0
                        scaled_offset = self.left_sphere_local_offset * scale_ratio
                        left_sphere_world = head_center + R_final @ scaled_offset
                        x_sphere_l = int(left_sphere_world[0])
                        y_sphere_l = int(left_sphere_world[1])
                        scaled_radius_l = int(self.base_radius * scale_ratio)
                        _frame_safe_circle(x_sphere_l, y_sphere_l, scaled_radius_l, (255, 255, 25), 2)

                    if not self.right_sphere_locked:
                        _frame_safe_circle(x_iris_r, y_iris_r, 10, (25, 255, 25), 2)
                    else:
                        current_nose_scale = self.compute_scale(nose_points_3d)
                        scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale if self.right_calibration_nose_scale else 1.0
                        scaled_offset_r = self.right_sphere_local_offset * scale_ratio_r
                        right_sphere_world = head_center + R_final @ scaled_offset_r
                        x_sphere_r = int(right_sphere_world[0])
                        y_sphere_r = int(right_sphere_world[1])
                        scaled_radius_r = int(self.base_radius * scale_ratio_r)
                        _frame_safe_circle(x_sphere_r, y_sphere_r, scaled_radius_r, (25, 255, 255), 2)

                    combined_parts = []
                    if left_sphere_world is not None:
                        left_dir = iris_3d_left - np.asarray(left_sphere_world, dtype=float)
                        norm_left = np.linalg.norm(left_dir)
                        if norm_left > 1e-9:
                            combined_parts.append(self._normalize(left_dir))
                    if right_sphere_world is not None:
                        right_dir = iris_3d_right - np.asarray(right_sphere_world, dtype=float)
                        norm_right = np.linalg.norm(right_dir)
                        if norm_right > 1e-9:
                            combined_parts.append(self._normalize(right_dir))

                    if not combined_parts:
                        l_left = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[0], fallback=(0.0, 0.0))
                        r_left = self._landmark_xy(face_landmarks, LEFT_EYE_HORIZ[1], fallback=(0.0, 0.0))
                        l_center = np.array([w * (l_left[0] + r_left[0]) * 0.5, h * (l_left[1] + r_left[1]) * 0.5, 0.0], dtype=float)
                        if len(face_landmarks) > 468:
                            l_left_z = float(face_landmarks[LEFT_EYE_HORIZ[0]].z)
                            r_left_z = float(face_landmarks[LEFT_EYE_HORIZ[1]].z)
                            l_center[2] = w * (l_left_z + r_left_z) * 0.5
                        if np.all(np.isfinite(l_center)):
                            left_dir = iris_3d_left - l_center
                            norm_left = np.linalg.norm(left_dir)
                            if norm_left > 1e-9:
                                combined_parts.append(self._normalize(left_dir))

                        l_right = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[0], fallback=(0.0, 0.0))
                        r_right = self._landmark_xy(face_landmarks, RIGHT_EYE_HORIZ[1], fallback=(0.0, 0.0))
                        r_center = np.array([w * (l_right[0] + r_right[0]) * 0.5, h * (l_right[1] + r_right[1]) * 0.5, 0.0], dtype=float)
                        if len(face_landmarks) > 473:
                            l_right_z = float(face_landmarks[RIGHT_EYE_HORIZ[0]].z)
                            r_right_z = float(face_landmarks[RIGHT_EYE_HORIZ[1]].z)
                            r_center[2] = w * (l_right_z + r_right_z) * 0.5
                        if np.all(np.isfinite(r_center)):
                            right_dir = iris_3d_right - r_center
                            norm_right = np.linalg.norm(right_dir)
                            if norm_right > 1e-9:
                                combined_parts.append(self._normalize(right_dir))

                    if combined_parts:
                        combined_vec = np.mean(np.stack(combined_parts, axis=0), axis=0)
                        norm_combined = np.linalg.norm(combined_vec)
                        if norm_combined > 1e-9:
                            avg_combined_direction = self._normalize(combined_vec)
                            self.combined_gaze_directions.append(avg_combined_direction)
                    elif self.combined_gaze_directions:
                        self.combined_gaze_directions.clear()

                    if self.combined_gaze_directions:
                        combined_vec = np.mean(np.stack(self.combined_gaze_directions, axis=0), axis=0)
                        norm_combined = np.linalg.norm(combined_vec)
                        if norm_combined > 1e-9:
                            avg_combined_direction = combined_vec / norm_combined
                        else:
                            avg_combined_direction = None
                    else:
                        avg_combined_direction = None

                target_x = None
                target_y = None
                raw_yaw = None
                raw_pitch = None
                self._update_dynamic_cursor_profile(face_landmarks)
                try:
                    if self.args.cursor_mode == "legacy_3d":
                        target_x, target_y = self._legacy_3d_target(
                            face_landmarks=face_landmarks,
                            avg_combined_direction=avg_combined_direction,
                        )
                        raw_yaw = self._legacy_raw_yaw_deg
                        raw_pitch = self._legacy_raw_pitch_deg
                    elif self.args.cursor_mode == "iris_direct":
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                    else:
                        target_x, target_y = self._feature_mapper_target(face_landmarks)

                    if target_x is None or target_y is None:
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                        raw_yaw = None
                        raw_pitch = None

                except Exception as exc:
                    try:
                        target_x, target_y = self._iris_direct_target(face_landmarks)
                        raw_yaw = None
                        raw_pitch = None
                    except Exception:
                        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
                        msg = "".join(tb[-2:]).replace("\n", " | ").strip()
                        self._log_failure(exc)
                        self._emit_noop(f"gaze_target_unavailable:{msg}")
                        target_x, target_y = None, None

                if target_x is not None and target_y is not None:
                    ts = now_ms() / 1000.0
                    target_x = float(self.x_filter(np.array([target_x], dtype=float), ts)[0])
                    target_y = float(self.y_filter(np.array([target_y], dtype=float), ts)[0])
                    target_x = max(0, min(self.monitor_width - 1, target_x))
                    target_y = max(0, min(self.monitor_height - 1, target_y))
                    self._last_gaze_norm = (
                        target_x / max(1.0, float(self.monitor_width - 1)),
                        target_y / max(1.0, float(self.monitor_height - 1)),
                    )
                    self.mouse_position = [target_x, target_y]
                    self._emit_gaze(target_x, target_y, 0.98)
                    if self.mouse_control_enabled:
                        smoothed_x, smoothed_y = self._clamp_jump(
                            float(target_x),
                            float(target_y),
                            ts,
                        )
                        self._move_cursor(smoothed_x, smoothed_y)
                    else:
                        self._log_gaze_cursor_trace(
                            target_x,
                            target_y,
                            moved=False,
                            backend_name=None,
                        )
                    self._write_screen_position(target_x, target_y)

                    if raw_yaw is not None and raw_pitch is not None:
                        texts = [
                            f"Screen: ({target_x}, {target_y})",
                            f"Yaw: {raw_yaw:.2f}",
                            f"Pitch: {raw_pitch:.2f}",
                        ]
                    else:
                        texts = [f"Screen: ({target_x}, {target_y})"]
                    for i, text in enumerate(texts):
                        color = (0, 255, 0)
                        cv2.putText(
                            frame,
                            text,
                            (20, 28 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            color,
                            2,
                        )
            else:
                self._emit_noop("no_face")
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            if self.args.debug:
                for idx, lm in enumerate(face_landmarks or []):
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    _frame_safe_circle(x, y, 0, (255, 255, 255), -1)

                if face_landmarks is not None:
                    landmarks3d = np.array(
                        [[p.x * w, p.y * h, p.z * w] for p in face_landmarks],
                        dtype=float,
                    )
                    self._update_orbit_from_keys()
                    self._render_debug_view_orbit(
                        h,
                        w,
                        head_center3d=head_center if face_landmarks is not None else None,
                        sphere_world_l=left_sphere_world if self.left_sphere_locked else None,
                        scaled_radius_l=scaled_radius_l if self.left_sphere_locked else None,
                        sphere_world_r=right_sphere_world if self.right_sphere_locked else None,
                        scaled_radius_r=scaled_radius_r if self.right_sphere_locked else None,
                        iris3d_l=iris_3d_left,
                        iris3d_r=iris_3d_right,
                        left_locked=self.left_sphere_locked,
                        right_locked=self.right_sphere_locked,
                        landmarks3d=landmarks3d,
                        combined_dir=avg_combined_direction,
                        gaze_len=5230,
                        monitor_corners=self.monitor_corners,
                        monitor_center=self.monitor_center_w,
                        monitor_normal=self.monitor_normal_w,
                        gaze_markers=self.gaze_markers,
                    )

                cv2.imshow("Integrated Eye Tracking", frame)

            self._publish_http_frame(frame)

            should_quit, last_toggle, key = _handle_runtime_controls(last_toggle)
            if should_quit:
                running = False
                continue

            api_calibration_requested, api_calibration_source = self._peek_center_calibration_request()
            if key == ord("c") or api_calibration_requested:
                now = time.time()
                if now - last_calibration_key >= 0.4:
                    calibrated = self._apply_center_calibration(
                        w=w,
                        h=h,
                        face_landmarks=face_landmarks,
                        head_center=head_center,
                        R_final=R_final,
                        nose_points_3d=nose_points_3d,
                        iris_3d_left=iris_3d_left,
                        iris_3d_right=iris_3d_right,
                        avg_combined_direction=avg_combined_direction,
                    )
                    if calibrated:
                        last_calibration_key = now
                        if api_calibration_requested:
                            self._clear_center_calibration_request()
                    elif api_calibration_requested and self.args.debug:
                        print(
                            "[Screen Calibration] Deferred queued request "
                            f"(source={api_calibration_source or 'api'}) until face/pose data is stable."
                        )

            if key == ord("s") and avg_combined_direction is not None:
                self._screen_calibrate(avg_combined_direction, face_landmarks)

            if key == ord("x"):
                if (
                    face_landmarks is not None
                    and head_center is not None
                    and R_final is not None
                    and nose_points_3d is not None
                ):
                    self._add_gaze_marker(
                        avg_combined_direction,
                        face_landmarks,
                        w,
                        h,
                        head_center,
                        R_final,
                        iris_3d_left,
                        iris_3d_right,
                        nose_points_3d,
                    )

        self._shutdown()

    def _legacy_3d_target(
        self,
        face_landmarks: Any,
        avg_combined_direction: Optional[np.ndarray],
    ) -> tuple[int, int]:
        geometry_target = self._legacy_3d_geometry_target(avg_combined_direction)
        if geometry_target is not None:
            if avg_combined_direction is not None:
                dir_vec = np.asarray(avg_combined_direction, dtype=float).reshape(-1)
                if dir_vec.size >= 3:
                    dir_vec = dir_vec[:3]
                    zf = abs(float(dir_vec[2]))
                    if zf < 1e-6:
                        zf = 1e-6
                    self._legacy_raw_yaw_deg = math.degrees(math.atan2(float(dir_vec[0]), zf))
                    self._legacy_raw_pitch_deg = math.degrees(math.atan2(-float(dir_vec[1]), zf))
            return geometry_target

        if avg_combined_direction is not None and self.left_sphere_locked and self.right_sphere_locked:
            clamped_x = self._stabilized_target[0]
            clamped_y = self._stabilized_target[1]
            clamped_x = max(0.0, min(float(self.monitor_width - 1), clamped_x))
            clamped_y = max(0.0, min(float(self.monitor_height - 1), clamped_y))
            return (int(round(clamped_x)), int(round(clamped_y)))
        return self.gaze_processing._legacy_target(
            face_landmarks=face_landmarks,
            avg_combined_direction=avg_combined_direction,
        )

    def _iris_direct_target(self, face_landmarks: Any) -> tuple[int, int]:
        return self.gaze_processing._iris_direct_target(face_landmarks=face_landmarks)

    def _feature_mapper_target(self, face_landmarks: Any) -> tuple[int, int]:
        return self.gaze_processing._feature_mapper_target(face_landmarks=face_landmarks)

    def _emit_gaze(self, x: int, y: int, confidence: float) -> None:
        now = now_ms()
        if now - self.last_emit_ms < self.args.emit_interval_ms:
            return
        self.last_emit_ms = now

        clamped_x = int(max(0, min(self.monitor_width - 1, x)))
        clamped_y = int(max(0, min(self.monitor_height - 1, y)))
        denom_x = float(max(1.0, float(self.monitor_width - 1)))
        denom_y = float(max(1.0, float(self.monitor_height - 1)))
        event = {
            "source": "gaze",
            "timestamp": now,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "intent": "gaze_target",
            "payload": {
                "target_x": clamped_x,
                "target_y": clamped_y,
                "x_norm": max(0.0, min(1.0, clamped_x / denom_x)),
                "y_norm": max(0.0, min(1.0, clamped_y / denom_y)),
            },
        }
        self._publish_http_event(event)
        self.event_bus.broadcast(event)

    def _move_cursor(self, target_x: int, target_y: int) -> None:
        if not self.mouse_control_enabled:
            return
        moved, last_err, backend_name = self.cursor_backends.move_cursor(
            target_x,
            target_y,
            monitor_width=self.monitor_width,
            monitor_height=self.monitor_height,
        )
        if moved:
            if self.args.debug and backend_name is not None:
                print(
                    f"[Tracker] cursor moved via backend '{backend_name}' to ({target_x}, {target_y})"
                )
                if last_err is not None:
                    print(
                        f"[Tracker] cursor movement warning for backend '{backend_name}': {last_err}"
                    )
            self._log_gaze_cursor_trace(
                target_x,
                target_y,
                moved=True,
                backend_name=backend_name,
            )
            return

        self._log_gaze_cursor_trace(
            target_x,
            target_y,
            moved=False,
            backend_name=backend_name,
        )
        if not self._cursor_move_warned:
            print("[Tracker] cursor-move failed. Check OS accessibility/input permissions.")
            if self.args.debug and last_err is not None:
                print(
                    f"[Tracker] cursor backends tested: [{self.cursor_backends.backend_names}] last_error={last_err}"
                )
            self._cursor_move_warned = True

    def _emit_noop(self, reason: str) -> None:
        now = now_ms()
        if now - self.last_emit_ms < self.args.emit_interval_ms:
            return
        self.last_emit_ms = now
        event = {
            "source": "gaze",
            "timestamp": now,
            "confidence": 0.0,
            "intent": "noop",
            "payload": {"reason": reason},
        }
        self.event_bus.broadcast(event)
        self._no_emit_reason.append(reason)
        self._publish_http_event(event)

    def _shutdown(self) -> None:
        self._stop_http_server()
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.event_bus.stop()
        self.face_mesh_backend.close()
