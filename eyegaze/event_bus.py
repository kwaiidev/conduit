#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import threading
from typing import Any, Dict, Optional


class SocketEventBus:
    """Simple local TCP socket broadcaster."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._clients: Dict[int, socket.socket] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        print(f"[Tracker] IPC server listening on tcp://{self.host}:{self.port}")

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                continue
            except Exception:
                continue
            print(f"[Tracker] client connected: {addr}")
            conn.setblocking(True)
            with self._lock:
                self._clients[conn.fileno()] = conn

    def broadcast(self, message: Dict[str, Any]) -> None:
        if not self._running:
            return
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        dead = []
        with self._lock:
            for key, sock in list(self._clients.items()):
                try:
                    sock.sendall(payload)
                except Exception:
                    dead.append(key)
            for key in dead:
                self._disconnect(key)

    def _disconnect(self, key: int) -> None:
        sock = self._clients.pop(key, None)
        if sock:
            try:
                sock.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for key in list(self._clients.keys()):
                self._disconnect(key)
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
