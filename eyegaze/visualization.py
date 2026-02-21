#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from typing import Any, Optional

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation as Rscipy
except Exception:
    Rscipy = None


class EyeTrackerVisualization:
    """Debug rendering and geometric calibration helpers extracted from EyeTrackerService."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

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

    def _draw_wireframe_cube(self, frame: np.ndarray, center: np.ndarray, R: np.ndarray, size: int = 80) -> None:
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

    def _update_orbit_from_keys(self) -> None:
        yaw_step = math.radians(1.5)
        pitch_step = math.radians(1.5)
        zoom_step = 12.0
        changed = False

        if self.owner._is_key_down("j"):
            self.owner.orbit_yaw -= yaw_step
            changed = True
        if self.owner._is_key_down("l"):
            self.owner.orbit_yaw += yaw_step
            changed = True
        if self.owner._is_key_down("i"):
            self.owner.orbit_pitch += pitch_step
            changed = True
        if self.owner._is_key_down("k"):
            self.owner.orbit_pitch -= pitch_step
            changed = True
        if self.owner._is_key_down("["):
            self.owner.orbit_radius += zoom_step
            changed = True
        if self.owner._is_key_down("]"):
            self.owner.orbit_radius = max(80.0, self.owner.orbit_radius - zoom_step)
            changed = True
        if self.owner._is_key_down("r"):
            self.owner.orbit_yaw = 0.0
            self.owner.orbit_pitch = 0.0
            self.owner.orbit_radius = 600.0
            changed = True

        self.owner.orbit_pitch = max(math.radians(-89), min(math.radians(89), self.owner.orbit_pitch))
        self.owner.orbit_radius = max(80.0, self.owner.orbit_radius)

        if changed:
            now = time.time()
            if now - self.owner._last_orbit_debug >= 0.06:
                print(
                    f"[Orbit Debug] yaw={math.degrees(self.owner.orbit_yaw):.2f}°, "
                    f"pitch={math.degrees(self.owner.orbit_pitch):.2f}°, "
                    f"radius={self.owner.orbit_radius:.2f}, fov={self.owner.orbit_fov_deg:.1f}°"
                )
                self.owner._last_orbit_debug = now

    def _render_debug_view_orbit(
        self,
        h: int,
        w: int,
        head_center3d: Optional[np.ndarray] = None,
        sphere_world_l: Optional[np.ndarray] = None,
        scaled_radius_l: Optional[float] = None,
        sphere_world_r: Optional[np.ndarray] = None,
        scaled_radius_r: Optional[float] = None,
        iris3d_l: Optional[np.ndarray] = None,
        iris3d_r: Optional[np.ndarray] = None,
        left_locked: bool = False,
        right_locked: bool = False,
        landmarks3d: Optional[np.ndarray] = None,
        combined_dir: Optional[np.ndarray] = None,
        gaze_len: int = 430,
        monitor_corners: Optional[list[np.ndarray]] = None,
        monitor_center: Optional[np.ndarray] = None,
        monitor_normal: Optional[np.ndarray] = None,
        gaze_markers: Optional[list[tuple[float, float]]] = None,
    ) -> None:
        if head_center3d is None:
            return

        debug = np.zeros((h, w, 3), dtype=np.uint8)
        head_w = np.asarray(head_center3d, dtype=float)
        if self.owner.debug_world_frozen and self.owner.orbit_pivot_frozen is not None:
            pivot_w = np.asarray(self.owner.orbit_pivot_frozen, dtype=float)
        else:
            if monitor_center is not None:
                pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
            else:
                pivot_w = head_w

        f_px = self._focal_px(w, self.owner.orbit_fov_deg)
        cam_offset = self._rot_y(self.owner.orbit_yaw) @ (self._rot_x(self.owner.orbit_pitch) @ np.array([0.0, 0.0, self.owner.orbit_radius]))
        cam_pos = pivot_w + cam_offset
        up_world = np.array([0.0, -1.0, 0.0])
        fwd = self._normalize(pivot_w - cam_pos)
        right = self._normalize(np.cross(fwd, up_world))
        up = self._normalize(np.cross(right, fwd))
        V = np.stack([right, up, fwd], axis=0)

        def project_point(P):
            Pw = np.asarray(P, dtype=float)
            Pc = V @ (Pw - cam_pos)
            if Pc[2] <= 1e-3:
                return None
            x = f_px * (Pc[0] / Pc[2]) + w * 0.5
            y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            return (int(x), int(y)), Pc[2]

        def draw_poly(points, color=(0, 200, 255), thickness=2):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs):
                return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                _safe_line(a, b, color, thickness)

        def draw_cross(P, size=12, color=(255, 0, 255), thickness=2):
            res = project_point(P)
            if res is None:
                return
            (x, y), _ = res
            p_left = (x - size, y)
            p_right = (x + size, y)
            p_up = (x, y - size)
            p_down = (x, y + size)
            _safe_line((x, y), p_left, color, thickness)
            _safe_line((x, y), p_right, color, thickness)
            _safe_line((x, y), p_up, color, thickness)
            _safe_line((x, y), p_down, color, thickness)

        def draw_arrow(P0, P1, color=(0, 200, 255), thickness=3):
            a = project_point(P0)
            b = project_point(P1)
            if a is None or b is None:
                return
            p0, p1 = a[0], b[0]
            _safe_line(p0, p1, color, thickness)
            v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
            n = np.linalg.norm(v)
            if n > 1e-3:
                v /= n
                l = np.array([-v[1], v[0]])
                ah = 10
                a1 = (
                    int(p1[0] - v[0] * ah + l[0] * ah * 0.6),
                    int(p1[1] - v[1] * ah + l[1] * ah * 0.6),
                )
                a2 = (
                    int(p1[0] - v[0] * ah - l[0] * ah * 0.6),
                    int(p1[1] - v[1] * ah - l[1] * ah * 0.6),
                )
                _safe_line(p1, a1, color, thickness)
                _safe_line(p1, a2, color, thickness)

        def _safe_point(res: Optional[tuple[tuple[int, int], float]]) -> Optional[tuple[int, int]]:
            if res is None:
                return None
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple):
                (x, y), _ = res
            elif isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], (int, np.integer)) and isinstance(res[1], (int, np.integer)):
                x, y = res
            else:
                return None
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        def _safe_line(
            p0_res: Optional[tuple[tuple[int, int], float]],
            p1_res: Optional[tuple[tuple[int, int], float]],
            color: tuple[int, int, int],
            thickness: int,
        ) -> None:
            p0 = _safe_point(p0_res)
            p1 = _safe_point(p1_res)
            if p0 is None or p1 is None:
                return
            clipped = cv2.clipLine((0, 0, w, h), p0, p1)
            if isinstance(clipped, tuple) and len(clipped) == 3 and clipped[0]:
                c0, c1 = clipped[1], clipped[2]
                cv2.line(debug, c0, c1, color, thickness)
            elif clipped is True:
                cv2.line(debug, p0, p1, color, thickness)

        def _safe_circle(
            center_res: Optional[tuple[tuple[int, int], float]],
            radius: int,
            color: tuple[int, int, int],
            thickness: int = 1,
            lineType: int = cv2.LINE_8,
        ) -> None:
            p = _safe_point(center_res)
            if p is None:
                return
            cx, cy = p
            rr = max(0, int(radius))
            if rr <= 0:
                cv2.circle(debug, (cx, cy), 0, color, thickness)
                return
            max_r = min(rr, max(0, w - 1), max(0, h - 1))
            if max_r <= 0:
                return
            cv2.circle(debug, p, max_r, color, thickness, lineType=lineType)

        def _safe_put_text(
            text: str,
            org: tuple[int, int],
            font_scale: float,
            color: tuple[int, int, int],
            thickness: int,
        ) -> None:
            x = int(min(max(0, org[0]), w - 1))
            y = int(min(max(0, org[1]), h - 1))
            if 0 <= x < w and 0 <= y < h:
                cv2.putText(
                    debug,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )

        if landmarks3d is not None:
            for P in landmarks3d:
                res = project_point(P)
                _safe_circle(res, 0, (200, 200, 200), -1)

        draw_cross(head_w, size=12, color=(255, 0, 255), thickness=2)
        hc2d = project_point(head_w)
        if hc2d is not None:
            _safe_put_text(
                "Head Center",
                (hc2d[0][0] + 12, hc2d[0][1] - 12),
                0.5,
                (255, 0, 255),
                1,
            )

        draw_cross(
            self.owner.orbit_pivot_frozen if self.owner.debug_world_frozen and self.owner.orbit_pivot_frozen is not None else pivot_w,
            size=8,
            color=(180, 120, 255),
            thickness=2,
        )
        if monitor_center is not None:
            mc2d = project_point(monitor_center)
            pv2d = project_point(
                self.owner.orbit_pivot_frozen
                if self.owner.debug_world_frozen and self.owner.orbit_pivot_frozen is not None
                else pivot_w
            )
            if mc2d is not None and pv2d is not None and hc2d is not None:
                _safe_line(pv2d, hc2d, (160, 100, 255), 1)
                _safe_line(pv2d, mc2d, (160, 100, 255), 1)

        left_dir = None
        right_dir = None
        if left_locked and sphere_world_l is not None:
            res = project_point(sphere_world_l)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
                _safe_circle((cx, cy), r_px, (255, 255, 25), 1)
                if iris3d_l is not None:
                    left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                    p1 = project_point(np.asarray(sphere_world_l) + self._normalize(left_dir) * gaze_len)
                    _safe_line((cx, cy), p1[0] if p1 is not None else None, (155, 155, 25), 1)
        elif iris3d_l is not None:
            res = project_point(iris3d_l)
            if res is not None:
                _safe_circle(res, 2, (255, 255, 25), 1)

        if right_locked and sphere_world_r is not None:
            res = project_point(sphere_world_r)
            if res is not None:
                (cx, cy), z = res
                r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
                _safe_circle((cx, cy), r_px, (25, 255, 255), 1)
                if iris3d_r is not None:
                    right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                    p1 = project_point(np.asarray(sphere_world_r) + self._normalize(right_dir) * gaze_len)
                    _safe_line((cx, cy), p1[0] if p1 is not None else None, (25, 155, 155), 1)
        elif iris3d_r is not None:
            res = project_point(iris3d_r)
            if res is not None:
                _safe_circle(res, 2, (25, 255, 255), 1)

        if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
            origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
            if combined_dir is None and (left_dir is not None or right_dir is not None):
                parts = []
                if left_dir is not None:
                    parts.append(self._normalize(left_dir))
                if right_dir is not None:
                    parts.append(self._normalize(right_dir))
                if parts:
                    combined_dir = self._normalize(np.mean(parts, axis=0))
            if combined_dir is not None:
                p0 = project_point(origin_mid)
                p1 = project_point(origin_mid + self._normalize(combined_dir) * (gaze_len * 1.2))
                _safe_line(p0[0] if p0 is not None else None, p1[0] if p1 is not None else None, (155, 200, 10), 2)

        if monitor_corners is not None:
            draw_poly(monitor_corners, (0, 200, 255), 2)
            draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
            draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
            if monitor_center is not None:
                draw_cross(monitor_center, size=8, color=(0, 200, 255), thickness=2)
                if monitor_normal is not None:
                    tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (
                        20.0 * (self.owner.units_per_cm or 1.0)
                    )
                    draw_arrow(monitor_center, tip, color=(0, 220, 255), thickness=2)

        if gaze_markers and monitor_corners is not None:
            p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
            u = p1 - p0
            width_world = float(np.linalg.norm(u))
            if width_world > 1e-9:
                u_hat = u / width_world
                r_world = 0.01 * width_world
                for (a, b) in gaze_markers:
                    Pm = p0 + a * u + b * (p3 - p0)
                    projP = project_point(Pm)
                    projR = project_point(Pm + u_hat * r_world)
                    if projP is not None and projR is not None:
                        center_px = projP[0]
                        r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                        _safe_circle((center_px, 1.0), r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        if (
            monitor_corners is not None
            and monitor_center is not None
            and monitor_normal is not None
            and combined_dir is not None
            and sphere_world_l is not None
            and sphere_world_r is not None
        ):
            O = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) * 0.5
            D = self._normalize(np.asarray(combined_dir))
            C = np.asarray(monitor_center)
            N = self._normalize(np.asarray(monitor_normal))
            denom = float(np.dot(N, D))
            if abs(denom) > 1e-6:
                t = float(np.dot(N, (C - O)) / denom)
                if t > 0.0:
                    P = O + t * D
                    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                    u = p1 - p0
                    v = p3 - p0
                    wv = P - p0
                    u_len2 = float(np.dot(u, u))
                    v_len2 = float(np.dot(v, v))
                    if u_len2 > 1e-9 and v_len2 > 1e-9:
                        a = float(np.dot(wv, u) / u_len2)
                        b = float(np.dot(wv, v) / v_len2)
                        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                            projP = project_point(P)
                            if projP is not None:
                                center_px = projP[0]
                                width_world2 = math.sqrt(u_len2)
                                r_world = 0.05 * width_world2
                                u_hat = u / max(width_world2, 1e-9)
                                projR = project_point(P + u_hat * r_world)
                                if projR is not None:
                                    r_px = int(
                                        max(
                                            1,
                                            np.linalg.norm(np.array(projR[0]) - np.array(center_px)),
                                        )
                                    )
                                    _safe_circle(
                                        (center_px, 1.0),
                                        r_px,
                                        (0, 255, 255),
                                        2,
                                        lineType=cv2.LINE_AA,
                                    )

        help_text = [
            "C = calibrate screen center",
            "J = yaw left",
            "L = yaw right",
            "I = pitch up",
            "K = pitch down",
            "[ = zoom out",
            "] = zoom in",
            "R = reset view",
            "X = add marker",
            "q = quit",
            "F7 = toggle mouse control",
        ]

        y0 = h - (len(help_text) * 18) - 10
        x0 = 10
        for i, text in enumerate(help_text):
            y = y0 + i * 18
            _safe_put_text(text, (x0, y), 0.5, (200, 200, 200), 1)

        cv2.imshow("Head/Eye Debug", debug)

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

        self.owner.monitor_corners, self.owner.monitor_center_w, self.owner.monitor_normal_w, self.owner.units_per_cm = self._create_monitor_plane(
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
        del w, h  # currently unused but kept for API compatibility with prior method signatures
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
            if self.owner.args.invert_gaze_x:
                a = 1.0 - a
            if self.owner.args.invert_gaze_y:
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
