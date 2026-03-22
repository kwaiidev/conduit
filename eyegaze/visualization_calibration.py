from __future__ import annotations

from typing import Any, Optional

import numpy as np


class EyeTrackerVisualizationCalibrationMixin:
    def _calibrate_spheres(
        self,
        w: int,
        h: int,
        head_center: np.ndarray,
        R_final: np.ndarray,
        face_landmarks: Any,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        nose_points_3d: np.ndarray,
    ) -> None:
        current_nose_scale = self._compute_scale(nose_points_3d)

        camera_dir_world = np.array([0, 0, 1], dtype=float)
        camera_dir_local = R_final.T @ camera_dir_world

        self.owner.left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        self.owner.left_sphere_local_offset += self.owner.base_radius * camera_dir_local
        self.owner.left_calibration_nose_scale = current_nose_scale
        self.owner.left_sphere_locked = True

        self.owner.right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        self.owner.right_sphere_local_offset += self.owner.base_radius * camera_dir_local
        self.owner.right_calibration_nose_scale = current_nose_scale
        self.owner.right_sphere_locked = True

        sphere_world_l_calib = head_center + R_final @ self.owner.left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ self.owner.right_sphere_local_offset
        left_dir = iris_3d_left - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        if np.linalg.norm(left_dir) > 1e-9:
            left_dir /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9:
            right_dir /= np.linalg.norm(right_dir)
        forward_hint = left_dir + right_dir
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint = self._normalize(forward_hint)
        else:
            forward_hint = None

        (
            self.owner.monitor_corners,
            self.owner.monitor_center_w,
            self.owner.monitor_normal_w,
            self.owner.units_per_cm,
        ) = self._create_monitor_plane(
            head_center,
            R_final,
            face_landmarks,
            w,
            h,
            forward_hint=forward_hint,
            gaze_origin=(sphere_world_l_calib + sphere_world_r_calib) / 2,
            gaze_dir=forward_hint,
        )

        self.owner.debug_world_frozen = True
        self.owner.orbit_pivot_frozen = self.owner.monitor_center_w.copy()
        print("[Debug View] World pivot frozen at monitor center.")
        print(
            f"[Monitor] units_per_cm={self.owner.units_per_cm:.3f}, center={self.owner.monitor_center_w}, "
            f"normal={self.owner.monitor_normal_w}"
        )
        print("[Both Spheres Locked] Eye sphere calibration complete.")

    def _screen_calibrate(
        self,
        avg_combined_direction: np.ndarray,
        face_landmarks: Any = None,
    ) -> None:
        _, _, raw_yaw, raw_pitch = self.owner.gaze_processing._convert_3d_gaze_to_screen(
            avg_combined_direction,
            face_landmarks,
        )
        self.owner.calibration_offset_yaw = -raw_yaw
        self.owner.calibration_offset_pitch = -raw_pitch
        print(
            f"[Screen Calibrated] Offset Yaw: {self.owner.calibration_offset_yaw:.2f}, "
            f"Offset Pitch: {self.owner.calibration_offset_pitch:.2f}"
        )

    def _add_gaze_marker(
        self,
        avg_combined_direction: Optional[np.ndarray],
        face_landmarks: Any,
        w: int,
        h: int,
        head_center: np.ndarray,
        R_final: np.ndarray,
        iris_3d_left: np.ndarray,
        iris_3d_right: np.ndarray,
        nose_points_3d: np.ndarray,
    ) -> None:
        del w, h
        if self.owner.monitor_corners is None or self.owner.monitor_center_w is None or self.owner.monitor_normal_w is None:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")
            return
        current_nose_scale = self._compute_scale(nose_points_3d)
        if self.owner.left_calibration_nose_scale and self.owner.right_calibration_nose_scale:
            scale_ratio_l = current_nose_scale / self.owner.left_calibration_nose_scale
            scale_ratio_r = current_nose_scale / self.owner.right_calibration_nose_scale
        else:
            scale_ratio_l = scale_ratio_r = 1.0
        sphere_world_l_now = head_center + R_final @ (self.owner.left_sphere_local_offset * scale_ratio_l)
        sphere_world_r_now = head_center + R_final @ (self.owner.right_sphere_local_offset * scale_ratio_r)

        if avg_combined_direction is not None:
            D = self._normalize(np.asarray(avg_combined_direction))
        else:
            lg = iris_3d_left - sphere_world_l_now
            rg = iris_3d_right - sphere_world_r_now
            if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                print("[Marker] Gaze direction invalid; try again.")
                return
            lg = lg / np.linalg.norm(lg)
            rg = rg / np.linalg.norm(rg)
            D = self._normalize(lg + rg)

        O = (sphere_world_l_now + sphere_world_r_now) * 0.5
        C = np.asarray(self.owner.monitor_center_w, dtype=float)
        N = self._normalize(np.asarray(self.owner.monitor_normal_w, dtype=float))
        denom = float(np.dot(N, D))
        if abs(denom) < 1e-6:
            print("[Marker] Gaze ray parallel to monitor; no marker.")
            return
        t = float(np.dot(N, (C - O)) / denom)
        if t <= 0.0:
            print("[Marker] Intersection behind/at eye; no marker.")
            return
        P = O + t * D
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.owner.monitor_corners]
        u = p1 - p0
        v = p3 - p0
        u_len2 = float(np.dot(u, u))
        v_len2 = float(np.dot(v, v))
        if u_len2 <= 1e-9 or v_len2 <= 1e-9:
            print("[Marker] Monitor dimensions degenerate; no marker.")
            return
        wv = P - p0
        a = float(np.dot(wv, u) / u_len2)
        b = float(np.dot(wv, v) / v_len2)
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            if self.owner.args.invert_gaze_x or self.owner.args.legacy_3d_invert_both:
                a = 1.0 - a
            if self.owner.args.invert_gaze_y or self.owner.args.legacy_3d_invert_both:
                b = 1.0 - b

            self.owner.gaze_markers.append((a, b))
            if self.owner.monitor_width and self.owner.monitor_height:
                marker_x = int(a * max(0, self.owner.monitor_width - 1))
                marker_y = int(b * max(0, self.owner.monitor_height - 1))
                print(
                    f"[Marker] Added at a={a:.3f}, b={b:.3f} "
                    f"-> screen_px=({marker_x}, {marker_y})"
                )
            else:
                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
        else:
            print("[Marker] Gaze not on monitor; no marker.")
