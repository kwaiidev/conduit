from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation as Rscipy
except Exception:
    Rscipy = None


class EyeTrackerVisualizationPrimitives:
    @staticmethod
    def _normalize(v: Any) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    @staticmethod
    def _focal_px(width: int, fov_deg: float) -> float:
        return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

    @staticmethod
    def _rot_x(a: float) -> np.ndarray:
        ca, sa = math.cos(a), math.sin(a)
        return np.array(
            [
                [1, 0, 0],
                [0, ca, -sa],
                [0, sa, ca],
            ],
            dtype=float,
        )

    @staticmethod
    def _rot_y(a: float) -> np.ndarray:
        ca, sa = math.cos(a), math.sin(a)
        return np.array(
            [
                [ca, 0, sa],
                [0, 1, 0],
                [-sa, 0, ca],
            ],
            dtype=float,
        )

    def _draw_gaze(
        self,
        frame: np.ndarray,
        eye_center: np.ndarray,
        iris_center: np.ndarray,
        eye_radius: int,
        color: tuple[int, int, int],
        gaze_length: int,
    ) -> None:
        h, w = frame.shape[:2]

        def _safe_point(p: np.ndarray) -> tuple[int, int]:
            x = int(round(float(p[0])))
            y = int(round(float(p[1])))
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return x, y

        def _safe_line(p0: np.ndarray, p1: np.ndarray, thickness_: int) -> None:
            cv2.line(frame, _safe_point(p0), _safe_point(p1), color, thickness_)

        gaze_direction = iris_center - eye_center
        gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        gaze_endpoint = eye_center + gaze_direction * gaze_length

        _safe_line(eye_center[:2], gaze_endpoint[:2], 2)

        iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)
        _safe_line(eye_center[:2], iris_offset[:2], 1)

        up_dir = np.array([0, -1, 0], dtype=float)
        right_dir = np.cross(gaze_direction, up_dir)
        if np.linalg.norm(right_dir) < 1e-6:
            right_dir = np.array([1, 0, 0], dtype=float)
        up_dir = np.cross(right_dir, gaze_direction)
        up_dir /= np.linalg.norm(up_dir)
        right_dir /= np.linalg.norm(right_dir)
        ellipse_axes = (
            int((eye_radius / 3) * np.linalg.norm(right_dir[:2])),
            int((eye_radius / 3) * np.linalg.norm(up_dir[:2])),
        )
        cv2.ellipse(
            frame,
            _safe_point(eye_center[:2]),
            ellipse_axes,
            math.degrees(math.atan2(gaze_direction[1], gaze_direction[0])),
            0,
            360,
            color,
            1,
        )

        _safe_line(iris_offset[:2], gaze_endpoint[:2], 1)

    def _draw_wireframe_cube(
        self,
        frame: np.ndarray,
        center: np.ndarray,
        R: np.ndarray,
        size: int = 80,
    ) -> None:
        right = R[:, 0]
        up = -R[:, 1]
        forward = -R[:, 2]
        hw, hh, hd = size, size, size
        h, w = frame.shape[:2]

        def corner(x_sign, y_sign, z_sign):
            return (
                center
                + x_sign * hw * right
                + y_sign * hh * up
                + z_sign * hd * forward
            )

        corners = [corner(x, y, z) for x in (-1, 1) for y in (1, -1) for z in (-1, 1)]
        projected = [
            (
                max(0, min(w - 1, int(round(pt[0])))),
                max(0, min(h - 1, int(round(pt[1])))),
            )
            for pt in corners
        ]
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

    def _compute_and_draw_coordinate_box(
        self,
        frame: np.ndarray,
        face_landmarks: Any,
        indices: list[int],
        ref_matrix_container: list[Optional[np.ndarray]],
        color: tuple[int, int, int] = (0, 255, 0),
        size: int = 80,
        w: int | None = None,
        h: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = frame.shape[1] if w is None else w
        height = frame.shape[0] if h is None else h
        points_3d = np.array(
            [[face_landmarks[i].x * width, face_landmarks[i].y * height, face_landmarks[i].z * width]
             for i in indices],
            dtype=float,
        )
        center = np.mean(points_3d, axis=0)
        for i in indices:
            x = max(0, min(width - 1, int(round(float(face_landmarks[i].x * width)))))
            y = max(0, min(height - 1, int(round(float(face_landmarks[i].y * height)))))
            cv2.circle(frame, (x, y), 3, color, -1)

        centered = points_3d - center
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, np.argsort(-eigvals)]
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, 2] *= -1

        if Rscipy is not None:
            r = Rscipy.from_matrix(eigvecs)
            roll, pitch, yaw = r.as_euler("zyx", degrees=False)
            yaw *= 1
            roll *= 1
            R_final = Rscipy.from_euler("zyx", [roll, pitch, yaw]).as_matrix()
        else:
            R_final = eigvecs

        if ref_matrix_container[0] is None:
            ref_matrix_container[0] = R_final.copy()
        else:
            R_ref = ref_matrix_container[0]
            if R_ref is not None:
                for i in range(3):
                    if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                        R_final[:, i] *= -1

        self._draw_wireframe_cube(frame, center, R_final, size)
        axis_length = size * 1.2
        axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
        axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        height, width = frame.shape[:2]

        def _safe2d_point(p: np.ndarray) -> tuple[int, int]:
            x = max(0, min(width - 1, int(round(float(p[0])))))
            y = max(0, min(height - 1, int(round(float(p[1])))))
            return x, y

        for i in range(3):
            end_pt = center + axis_dirs[i] * axis_length
            p0 = _safe2d_point(center)
            p1 = _safe2d_point(end_pt)
            cv2.line(frame, p0, p1, axis_colors[i], 2)
        return center, R_final, points_3d

    def _create_monitor_plane(
        self,
        head_center: np.ndarray,
        R_final: np.ndarray,
        face_landmarks: Any,
        w: int,
        h: int,
        forward_hint: Optional[np.ndarray] = None,
        gaze_origin: Optional[np.ndarray] = None,
        gaze_dir: Optional[np.ndarray] = None,
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
        try:
            lm_chin = face_landmarks[152]
            lm_fore = face_landmarks[10]
            chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
            fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
            face_h_units = np.linalg.norm(fore_w - chin_w)
            upc = face_h_units / 15.0
        except Exception:
            upc = 5.0

        head_forward = -R_final[:, 2]
        if forward_hint is not None:
            nf = np.linalg.norm(forward_hint)
            if nf > 1e-9:
                head_forward = forward_hint / nf

        if gaze_origin is not None and gaze_dir is not None:
            gd = self._normalize(gaze_dir)
            plane_point = head_center + head_forward * (50.0 * upc)
            plane_normal = head_forward
            denom = np.dot(plane_normal, gd)
            if abs(denom) > 1e-6:
                t = np.dot(plane_normal, plane_point - gaze_origin) / denom
                center_w = gaze_origin + t * gd
            else:
                center_w = head_center + head_forward * (50.0 * upc)
        else:
            center_w = head_center + head_forward * (50.0 * upc)

        world_up = np.array([0, -1, 0], dtype=float)
        head_right = np.cross(world_up, head_forward)
        if np.linalg.norm(head_right) < 1e-9:
            head_right = np.array([1, 0, 0], dtype=float)
        head_right /= np.linalg.norm(head_right)
        head_up = np.cross(head_forward, head_right)
        head_up = self._normalize(head_up)

        mon_w_cm, mon_h_cm = 60.0, 40.0
        half_w = (mon_w_cm * 0.5) * upc
        half_h = (mon_h_cm * 0.5) * upc

        p0 = center_w - head_right * half_w - head_up * half_h
        p1 = center_w + head_right * half_w - head_up * half_h
        p2 = center_w + head_right * half_w + head_up * half_h
        p3 = center_w - head_right * half_w + head_up * half_h
        normal_w = self._normalize(head_forward)
        return [p0, p1, p2, p3], center_w, normal_w, upc

    @staticmethod
    def _compute_scale(points_3d: np.ndarray) -> float:
        n = len(points_3d)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += float(np.linalg.norm(points_3d[i] - points_3d[j]))
                count += 1
        return total / count if count > 0 else 1.0
