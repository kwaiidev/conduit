from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np


class EyeTrackerVisualizationOrbitMixin:
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
        cam_offset = self._rot_y(self.owner.orbit_yaw) @ (
            self._rot_x(self.owner.orbit_pitch) @ np.array([0.0, 0.0, self.owner.orbit_radius])
        )
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

        def _safe_point(res):
            if res is None:
                return None
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], tuple):
                (x, y), _ = res
            elif (
                isinstance(res, tuple)
                and len(res) == 2
                and isinstance(res[0], (int, np.integer))
                and isinstance(res[1], (int, np.integer))
            ):
                x, y = res
            else:
                return None
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        def _safe_line(
            p0_res,
            p1_res,
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
            center_res,
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
