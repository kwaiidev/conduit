#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Any, Optional


class CursorBackendManager:
    """OS cursor backend selection and movement fallback chain."""

    _MOVE_TOLERANCE_PX = 3
    _VERIFY_ATTEMPTS = 3
    _VERIFY_SLEEP_SEC = 0.004

    def __init__(self, args: Any, monitor_width: int, monitor_height: int) -> None:
        self.args = args
        self.monitor_width = int(monitor_width)
        self.monitor_height = int(monitor_height)
        self._cursor_backends: list[dict[str, Any]] = []
        self._cursor_backend_index = 0
        self._cursor_backend = {"type": "none"}
        self._cursor_move_warned = False
        self._cursor_pos = None
        self._backend_names: list[str] = []
        self._init_cursor_backends()

    @property
    def backends(self) -> list[dict[str, Any]]:
        return self._cursor_backends

    @property
    def backend_names(self) -> str:
        return ", ".join(self._backend_names)

    @property
    def active_backend(self) -> dict[str, Any]:
        return self._cursor_backend

    def _init_cursor_backends(self) -> None:
        self._cursor_backends = []
        self._backend_names = []
        self._cursor_backend = {"type": "none"}
        self._cursor_backend_index = 0

        def _add(name: str, entry: dict[str, Any]) -> None:
            self._cursor_backends.append(entry)
            self._backend_names.append(name)

        try:
            import pyautogui

            if hasattr(pyautogui, "FAILSAFE"):
                pyautogui.FAILSAFE = False
            if hasattr(pyautogui, "PAUSE"):
                pyautogui.PAUSE = 0
            _add("pyautogui", {"type": "pyautogui", "api": pyautogui, "read": True})
        except Exception:
            pass

        if self._is_darwin():
            move_fn = self._make_darwin_cursor_move()
            if move_fn is not None:
                # CoreGraphics fallback: keep available when other toolkits are blocked.
                _add("coregraphics", {"type": "coregraphics", "move": move_fn, "read": True})

        if self._is_windows():
            move_fn = self._make_win32_cursor_move()
            if move_fn is not None:
                self._cursor_backends.append({"type": "win32", "move": move_fn, "read": False})
                self._backend_names.append("win32")
        elif self._is_linux():
            move_fn = self._make_linux_cursor_move()
            if move_fn is not None:
                _add("xdotool", {"type": "xdotool", "move": move_fn, "read": False})
        elif self._is_darwin():
            pass
        else:
            try:
                from pynput.mouse import Controller

                controller = Controller()
                _ = controller.position
                _add("pynput", {"type": "pynput", "api": controller, "read": True})
            except Exception:
                pass

        if self._cursor_backends:
            self._cursor_backend = self._cursor_backends[0]

    def ensure_cursor_backends(self) -> bool:
        if self._cursor_backends:
            return True
        self._init_cursor_backends()
        self._cursor_move_warned = False
        return bool(self._cursor_backends)

    def _as_int_pair(self, value: Any) -> Optional[tuple[int, int]]:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            pass
        try:
            x = int(getattr(value, "x"))
            y = int(getattr(value, "y"))
            return x, y
        except Exception:
            return None

    def read_cursor_position(self) -> Optional[tuple[int, int]]:
        if not self._cursor_backends:
            return None

        ordered_backends: list[dict[str, Any]] = []
        if self._cursor_backend and isinstance(self._cursor_backend, dict):
            ordered_backends.append(self._cursor_backend)
        for entry in self._cursor_backends:
            if entry is self._cursor_backend:
                continue
            ordered_backends.append(entry)

        for entry in ordered_backends:
            pos = self._read_cursor_position_for_backend(entry)
            if pos is not None:
                return pos
        return None

    def _read_cursor_position_for_backend(self, backend: dict[str, Any]) -> Optional[tuple[int, int]]:
        kind = backend.get("type", "none")
        if kind == "pyautogui":
            api = backend.get("api")
            if api is not None and hasattr(api, "position"):
                try:
                    return self._as_int_pair(api.position())
                except Exception:
                    pass
            return None

        if kind == "pynput":
            api = backend.get("api")
            if api is None:
                return None
            try:
                return self._as_int_pair(api.position)
            except Exception:
                pass
            return None

        if kind == "coregraphics":
            return self._read_darwin_cursor_position()

        if kind == "win32":
            # Avoid platform-specific read fallback for now.
            return None

        if kind == "xdotool":
            return None

        return None

    def _is_within_target(self, actual: tuple[int, int], target: tuple[int, int]) -> bool:
        return (
            abs(actual[0] - target[0]) <= self._MOVE_TOLERANCE_PX
            and abs(actual[1] - target[1]) <= self._MOVE_TOLERANCE_PX
        )

    def _read_and_verify_position(
        self,
        backend: dict[str, Any],
        target: tuple[int, int],
    ) -> tuple[bool, Optional[tuple[int, int]], Optional[Exception]]:
        last_err: Optional[Exception] = None
        last_pos: Optional[tuple[int, int]] = None
        if not backend.get("read", False):
            return False, None, None

        for _ in range(self._VERIFY_ATTEMPTS):
            try:
                end_pos = self._read_cursor_position_for_backend(backend)
            except Exception as exc:
                last_err = exc
                end_pos = None
            if end_pos is None:
                continue
            last_pos = end_pos
            if self._is_within_target(end_pos, target):
                return True, end_pos, None
            last_err = RuntimeError(
                f"backend {backend.get('type', 'unknown')} readback was ({end_pos[0]}, {end_pos[1]})"
            )
            if self._VERIFY_SLEEP_SEC > 0:
                try:
                    time.sleep(self._VERIFY_SLEEP_SEC)
                except Exception:
                    pass

        return False, last_pos, last_err


    @staticmethod
    def _dispatch_cursor_move(backend: dict[str, Any], new_x: int, new_y: int) -> None:
        kind = backend.get("type", "none")
        move_fn = backend.get("move")
        if move_fn is not None:
            move_fn(new_x, new_y)
            return
        if kind == "pyautogui":
            try:
                backend["api"].moveTo(new_x, new_y, 0, _pause=False)
            except TypeError:
                try:
                    backend["api"].moveTo(new_x, new_y, 0)
                except TypeError:
                    backend["api"].moveTo(new_x, new_y)
            return
        if kind == "pynput":
            backend["api"].position = (new_x, new_y)
            return
        raise RuntimeError(f"Unsupported cursor backend: {kind}")

    def move_cursor(self, target_x: int, target_y: int, *, monitor_width: Optional[int] = None, monitor_height: Optional[int] = None) -> tuple[bool, Optional[Exception], Optional[str]]:
        if not self._cursor_backends:
            if not self.ensure_cursor_backends():
                return False, None, None
            if not self._cursor_backends:
                return False, None, None

        new_x = int(target_x)
        new_y = int(target_y)
        if monitor_width is not None:
            monitor_width = max(1, int(monitor_width))
            new_x = max(0, min(monitor_width - 1, new_x))
        if monitor_height is not None:
            monitor_height = max(1, int(monitor_height))
            new_y = max(0, min(monitor_height - 1, new_y))

        backends_count = len(self._cursor_backends)
        target = (new_x, new_y)

        last_err: Optional[Exception] = None
        last_attempt_index: Optional[int] = None
        last_mismatch: Optional[tuple[int, int]] = None

        had_read_backend = any(entry.get("read", False) for entry in self._cursor_backends)

        def _ordered_attempt_indices() -> list[int]:
            return [
                (self._cursor_backend_index + step) % backends_count for step in range(backends_count)
            ]

        def _dispatch_and_optionally_verify(attempt_idx: int) -> tuple[bool, bool, Optional[Exception], Optional[tuple[int, int]]]:
            """Return (success, readable, err, last_readback)."""
            backend = self._cursor_backends[attempt_idx]
            kind = backend.get("type", "none")
            try:
                if getattr(self.args, "debug", False):
                    print(
                        f"[CursorBackend] dispatch '{kind}' to ({new_x}, {new_y}) "
                        f"idx={attempt_idx}"
                    )
                self._dispatch_cursor_move(backend, new_x, new_y)
                if not backend.get("read", False):
                    self._cursor_backend = backend
                    self._cursor_backend_index = attempt_idx
                    self._cursor_move_warned = False
                    self._cursor_pos = (new_x, new_y)
                    if getattr(self.args, "debug", False):
                        print(
                            f"[CursorBackend] backend '{kind}' dispatched without readback support; "
                            "treating as moved."
                        )

                    return True, False, None, None

                moved, end_pos, verify_err = self._read_and_verify_position(backend, target)
                if moved:
                    self._cursor_backend = backend
                    self._cursor_backend_index = attempt_idx
                    self._cursor_move_warned = False
                    self._cursor_pos = (new_x, new_y)
                    if getattr(self.args, "debug", False):
                        print(
                            f"[CursorBackend] backend '{kind}' readback matched target "
                            f"{end_pos} ~= {target}."
                        )
                    return True, True, None, end_pos

                if getattr(self.args, "debug", False):
                    print(
                        f"[CursorBackend] backend '{kind}' readback mismatch: "
                        f"target={target}, readback={end_pos}"
                    )
                return False, True, verify_err, end_pos
            except Exception as exc:
                if getattr(self.args, "debug", False):
                    print(f"[CursorBackend] backend '{kind}' exception: {exc}")
                return False, bool(backend.get("read", False)), exc, None
            return False, bool(backend.get("read", False)), None, None

        # Try readable backends first. Keep moving on the first readable backend that verifies
        # movement. If none verify, retain the last mismatch and continue.
        for attempt_idx in _ordered_attempt_indices():
            backend = self._cursor_backends[attempt_idx]
            if not backend.get("read", False):
                continue
            last_attempt_index = attempt_idx
            moved, _, verify_err, readback = _dispatch_and_optionally_verify(attempt_idx)
            last_err = verify_err or last_err
            if readback is not None:
                last_mismatch = readback
            if moved:
                return True, None, backend.get("type", "unknown")

        # Fallback to non-readable backends when verification isn't possible.
        for attempt_idx in _ordered_attempt_indices():
            backend = self._cursor_backends[attempt_idx]
            if backend.get("read", False):
                continue
            last_attempt_index = attempt_idx
            moved, _, _err, _ = _dispatch_and_optionally_verify(attempt_idx)
            if moved:
                return True, None, backend.get("type", "unknown")
            if _err is not None:
                last_err = _err

        # No non-readable backends available and no readable backend confirmed movement.
        # If a readable backend was attempted, keep the last attempted backend as active only
        # when the command path executed (read failure is often false-negative on permissions).
        if not had_read_backend:
            if last_err is not None:
                return False, last_err, None
            if getattr(self.args, "debug", False):
                print(
                    f"[CursorBackend] all backends attempted but no dispatch succeeded at {target}"
                )
            return False, None, None

        # Last resort: readable backends may not be verifiable even when they move.
        if last_attempt_index is not None:
            backend = self._cursor_backends[last_attempt_index]
            self._cursor_backend = backend
            self._cursor_backend_index = last_attempt_index
            self._cursor_pos = (new_x, new_y)
            if getattr(self.args, "debug", False):
                print(
                    "[CursorBackend] no verified readback from readable backends; "
                    f"returning best-effort success for {backend.get('type', 'unknown')} "
                    f"last_mismatch={last_mismatch}"
                )
            return True, None, backend.get("type", "unknown")

        return False, last_err, None

    def _make_darwin_cursor_move(self) -> Any | None:
        try:
            import ctypes
            import ctypes.util

            cg_path = ctypes.util.find_library("CoreGraphics")
            if cg_path is None:
                cg_path = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            cg = ctypes.CDLL(cg_path)

            class CGPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

            k_cg_event_mouse_moved = 5
            k_cg_event_tap = 0
            k_cg_hid_session = 0
            k_cg_left_mouse_button = 0

            if hasattr(cg, "CGDisplayMoveCursorToPoint"):
                cg.CGDisplayMoveCursorToPoint.argtypes = [ctypes.c_uint32, CGPoint]
                cg.CGDisplayMoveCursorToPoint.restype = ctypes.c_int32
                cg.CGMainDisplayID.restype = ctypes.c_uint32
                display_id = cg.CGMainDisplayID()
            else:
                display_id = None

            if hasattr(cg, "CGWarpMouseCursorPosition"):
                cg.CGWarpMouseCursorPosition.argtypes = [CGPoint]
                cg.CGWarpMouseCursorPosition.restype = ctypes.c_int32
            if hasattr(cg, "CGAssociateMouseAndMouseCursorPosition"):
                cg.CGAssociateMouseAndMouseCursorPosition.argtypes = [ctypes.c_int32]
                cg.CGAssociateMouseAndMouseCursorPosition.restype = ctypes.c_int32
                try:
                    cg.CGAssociateMouseAndMouseCursorPosition(1)
                except Exception:
                    pass
            if hasattr(cg, "CGEventCreateMouseEvent"):
                cg.CGEventCreateMouseEvent.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    CGPoint,
                    ctypes.c_uint32,
                ]
                cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
            if hasattr(cg, "CGEventCreate"):
                cg.CGEventCreate.argtypes = [ctypes.c_void_p]
                cg.CGEventCreate.restype = ctypes.c_void_p
            if hasattr(cg, "CGEventPost"):
                cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
                cg.CGEventPost.restype = None
            if hasattr(cg, "CGPostMouseEvent"):
                cg.CGPostMouseEvent.argtypes = [CGPoint, ctypes.c_bool, ctypes.c_uint32, ctypes.c_uint32]
                cg.CGPostMouseEvent.restype = None
            if hasattr(cg, "CGEventSourceCreate"):
                cg.CGEventSourceCreate.argtypes = [ctypes.c_uint32]
                cg.CGEventSourceCreate.restype = ctypes.c_void_p
            if hasattr(cg, "CFRelease"):
                cg.CFRelease.argtypes = [ctypes.c_void_p]
                cg.CFRelease.restype = None

            display_h = float(self.monitor_height)

            def _move(x: int, y: int) -> None:
                px = float(x)
                py = float(y)
                if display_h > 0:
                    py = (display_h - 1.0) - py
                point = CGPoint(px, py)

                if display_id is not None and hasattr(cg, "CGDisplayMoveCursorToPoint"):
                    try:
                        result = int(cg.CGDisplayMoveCursorToPoint(display_id, point))
                        if result != 0:
                            raise RuntimeError(
                                f"CGDisplayMoveCursorToPoint failed with result code {result}"
                            )
                        return
                    except Exception:
                        pass

                if hasattr(cg, "CGEventCreateMouseEvent") and hasattr(cg, "CGEventPost"):
                    source = None
                    event = None
                    try:
                        if hasattr(cg, "CGEventSourceCreate"):
                            source = cg.CGEventSourceCreate(k_cg_hid_session)
                        event = cg.CGEventCreateMouseEvent(
                            source,
                            k_cg_event_mouse_moved,
                            point,
                            k_cg_left_mouse_button,
                        )
                        if not event:
                            raise RuntimeError("CGEventCreateMouseEvent returned NULL")
                        cg.CGEventPost(k_cg_event_tap, event)
                        return
                    finally:
                        try:
                            if event:
                                cg.CFRelease(event)
                        except Exception:
                            pass
                        try:
                            if source:
                                cg.CFRelease(source)
                        except Exception:
                            pass

                if hasattr(cg, "CGPostMouseEvent"):
                    try:
                        cg.CGPostMouseEvent(point, True, k_cg_left_mouse_button, 0)
                        return
                    except Exception:
                        pass

                if hasattr(cg, "CGWarpMouseCursorPosition"):
                    result = int(cg.CGWarpMouseCursorPosition(point))
                    if result != 0:
                        raise RuntimeError(f"CoreGraphics warp failed with result code {result}")
                    return

                raise RuntimeError("CoreGraphics backend has no supported backend")

            return _move
        except Exception:
            return None

    def _make_win32_cursor_move(self) -> Any | None:
        try:
            import ctypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            def _move(x: int, y: int) -> None:
                user32.SetCursorPos(x, y)

            _ = _move
            return _move
        except Exception:
            return None

    def _make_linux_cursor_move(self) -> Any | None:
        try:
            import shutil
            import subprocess

            xdotool = shutil.which("xdotool")
            if xdotool is None:
                return None

            def _move(x: int, y: int) -> None:
                subprocess.run([xdotool, "mousemove", str(int(x)), str(int(y))], check=True)

            return _move
        except Exception:
            return None

    def _read_darwin_cursor_position(self) -> Optional[tuple[int, int]]:
        try:
            import ctypes
            import ctypes.util

            cg_path = ctypes.util.find_library("CoreGraphics")
            if cg_path is None:
                cg_path = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
            cg = ctypes.CDLL(cg_path)

            class CGPoint(ctypes.Structure):
                _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

            if not hasattr(cg, "CGEventCreate") or not hasattr(cg, "CGEventGetLocation"):
                return None
            cg.CGEventCreate.argtypes = [ctypes.c_void_p]
            cg.CGEventCreate.restype = ctypes.c_void_p
            cg.CGEventGetLocation.argtypes = [ctypes.c_void_p]
            cg.CGEventGetLocation.restype = CGPoint
            if hasattr(cg, "CFRelease"):
                cg.CFRelease.argtypes = [ctypes.c_void_p]
                cg.CFRelease.restype = None

            display_h = float(self.monitor_height)
            event = cg.CGEventCreate(None)
            if not event:
                return None
            try:
                point = cg.CGEventGetLocation(event)
                x = float(point.x)
                y = float(point.y)
                if display_h > 0:
                    y = (display_h - 1.0) - y
                return int(x), int(y)
            finally:
                try:
                    cg.CFRelease(event)
                except Exception:
                    pass
        except Exception:
            return None
        return None

    def _is_darwin(self) -> bool:
        import sys

        return sys.platform == "darwin"

    def _is_windows(self) -> bool:
        import sys

        return sys.platform.startswith("win")

    def _is_linux(self) -> bool:
        import sys

        return sys.platform.startswith("linux")
