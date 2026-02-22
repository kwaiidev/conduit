import { app, BrowserWindow, ipcMain, screen, globalShortcut } from "electron";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import net from "net";

let win: BrowserWindow | null = null;
let currentMode: 'full' | 'overlay' = 'full';
let isToggling = false;
let cvBackendPid: number | null = null;
const SYSTEM_TARGET_TIMEOUT_MS = 700;
const SNAP_GLOBAL_SHORTCUT = "CommandOrControl+Shift+G";
const SNAP_TARGET_COOLDOWN_MS = 520;

let lastSnappedTargetId: string | null = null;
let lastSnappedTargetAt = 0;

type SystemSnapTarget = {
  id: string;
  kind: string;
  label: string;
  left: number;
  top: number;
  width: number;
  height: number;
};

type SystemSnapTargetsResponse = {
  ok: boolean;
  provider: string;
  reason?: string;
  targets: SystemSnapTarget[];
};

type SnapCommitResponse = {
  ok: boolean;
  message: string;
  target?: {
    id: string;
    x: number;
    y: number;
  };
};

async function moveCursorToScreenPoint(x: number, y: number): Promise<boolean> {
  if (process.platform !== "darwin") {
    return false;
  }
  const clampedX = Number.isFinite(x) ? Math.max(0, x) : 0;
  const clampedY = Number.isFinite(y) ? Math.max(0, y) : 0;
  const script = `
import ctypes
import sys

class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

target_x = float(sys.argv[1])
target_y = float(sys.argv[2])
api = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
result = api.CGWarpMouseCursorPosition(CGPoint(target_x, target_y))
api.CGAssociateMouseAndMouseCursorPosition(True)
sys.exit(0 if result == 0 else 1)
`;

  return await new Promise(resolve => {
    const child = spawn("python3", ["-c", script, String(clampedX), String(clampedY)], {
      stdio: ["ignore", "ignore", "ignore"],
    });
    child.once("error", () => resolve(false));
    child.once("close", code => resolve(code === 0));
  });
}

async function commitNearestSystemSnapTarget(): Promise<SnapCommitResponse> {
  const cursor = screen.getCursorScreenPoint();
  const response = await listSystemSnapTargets();
  if (!response.ok) {
    const reason = response.reason ? ` (${response.reason})` : "";
    return {
      ok: false,
      message: `System target discovery unavailable${reason}. Grant Accessibility permissions.`,
    };
  }
  if (!response.targets.length) {
    return {
      ok: false,
      message: "No snap targets found in the frontmost app window.",
    };
  }

  const now = Date.now();
  const ranked = response.targets
    .map(target => {
      const cx = target.left + target.width * 0.5;
      const cy = target.top + target.height * 0.5;
      const d = Math.hypot(cx - cursor.x, cy - cursor.y);
      let score = d;
      if (target.id === lastSnappedTargetId) {
        const cooldownLeft = Math.max(0, SNAP_TARGET_COOLDOWN_MS - (now - lastSnappedTargetAt));
        score += cooldownLeft > 0 ? 120 : 18;
      }
      return { target, cx, cy, score };
    })
    .sort((a, b) => a.score - b.score);

  const best = ranked[0];
  if (!best) {
    return {
      ok: false,
      message: "No ranked snap target available.",
    };
  }

  const moved = await moveCursorToScreenPoint(best.cx, best.cy);
  if (!moved) {
    return {
      ok: false,
      message: "Failed to move system cursor. Verify OS permissions.",
    };
  }

  lastSnappedTargetId = best.target.id;
  lastSnappedTargetAt = now;
  return {
    ok: true,
    message: `Snapped to ${best.target.label}`,
    target: {
      id: best.target.id,
      x: best.cx,
      y: best.cy,
    },
  };
}

function parseSystemTargetLines(raw: string): SystemSnapTarget[] {
  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);
  const targets: SystemSnapTarget[] = [];
  let seq = 0;
  for (const line of lines) {
    const parts = line.split("\t");
    if (parts.length < 6) {
      continue;
    }
    const kind = parts[0]?.trim().toLowerCase() || "control";
    const label = parts[1]?.trim() || kind;
    const left = Number(parts[2]);
    const top = Number(parts[3]);
    const width = Number(parts[4]);
    const height = Number(parts[5]);
    if (
      !Number.isFinite(left)
      || !Number.isFinite(top)
      || !Number.isFinite(width)
      || !Number.isFinite(height)
      || width < 8
      || height < 8
    ) {
      continue;
    }
    seq += 1;
    targets.push({
      id: `${kind}-${seq}`,
      kind,
      label,
      left,
      top,
      width,
      height,
    });
  }
  return targets;
}

async function listSystemSnapTargets(): Promise<SystemSnapTargetsResponse> {
  if (process.platform !== "darwin") {
    return {
      ok: false,
      provider: "system-events",
      reason: "unsupported-platform",
      targets: [],
    };
  }

  const script = `
set outLines to {}
tell application "System Events"
  if UI elements enabled is false then
    return "__ERROR__\\tAX_DISABLED"
  end if
  try
    set frontProc to first application process whose frontmost is true
    set frontWin to front window of frontProc
  on error
    return "__ERROR__\\tNO_FRONT_WINDOW"
  end try

  repeat with btn in (buttons of frontWin)
    try
      set p to position of btn
      set s to size of btn
      set end of outLines to ("button\\tbutton\\t" & (item 1 of p) & "\\t" & (item 2 of p) & "\\t" & (item 1 of s) & "\\t" & (item 2 of s))
    end try
  end repeat

  repeat with txt in (text fields of frontWin)
    try
      set p to position of txt
      set s to size of txt
      set end of outLines to ("text_field\\tinput\\t" & (item 1 of p) & "\\t" & (item 2 of p) & "\\t" & (item 1 of s) & "\\t" & (item 2 of s))
    end try
  end repeat

  repeat with cb in (checkboxes of frontWin)
    try
      set p to position of cb
      set s to size of cb
      set end of outLines to ("checkbox\\tcheckbox\\t" & (item 1 of p) & "\\t" & (item 2 of p) & "\\t" & (item 1 of s) & "\\t" & (item 2 of s))
    end try
  end repeat
end tell
set AppleScript's text item delimiters to linefeed
return outLines as text
`;

  return new Promise(resolve => {
    let stdout = "";
    let stderr = "";
    const child = spawn("osascript", ["-e", script], { stdio: ["ignore", "pipe", "pipe"] });
    const timeout = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // no-op
      }
      resolve({
        ok: false,
        provider: "system-events",
        reason: "timeout",
        targets: [],
      });
    }, SYSTEM_TARGET_TIMEOUT_MS);

    child.stdout.on("data", chunk => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", chunk => {
      stderr += chunk.toString();
    });

    child.on("error", error => {
      clearTimeout(timeout);
      resolve({
        ok: false,
        provider: "system-events",
        reason: `spawn-error:${error.message}`,
        targets: [],
      });
    });

    child.on("close", code => {
      clearTimeout(timeout);
      if (code !== 0) {
        resolve({
          ok: false,
          provider: "system-events",
          reason: `osascript-exit:${code}:${stderr.trim()}`,
          targets: [],
        });
        return;
      }
      const text = stdout.trim();
      if (!text) {
        resolve({
          ok: true,
          provider: "system-events",
          reason: "empty",
          targets: [],
        });
        return;
      }
      if (text.startsWith("__ERROR__")) {
        const reason = text.split("\t")[1] || "unknown";
        resolve({
          ok: false,
          provider: "system-events",
          reason,
          targets: [],
        });
        return;
      }
      resolve({
        ok: true,
        provider: "system-events",
        targets: parseSystemTargetLines(text),
      });
    });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function probeTcp(host: string, port: number, timeoutMs = 1000): Promise<boolean> {
  return new Promise(resolve => {
    const socket = new net.Socket();
    let settled = false;

    const finish = (ok: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      try {
        socket.destroy();
      } catch {
        // no-op
      }
      resolve(ok);
    };

    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    socket.connect(port, host);
  });
}

async function waitForBackendTcp(
  host: string,
  port: number,
  timeoutMs: number,
  pid: number | null
): Promise<boolean> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (pid && !isPidAlive(pid)) {
      return false;
    }
    const ok = await probeTcp(host, port, 900);
    if (ok) {
      return true;
    }
    await sleep(250);
  }
  return false;
}

function isPidAlive(pid: number | null): boolean {
  if (!pid || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function killCvBackendPid(pid: number, signal: NodeJS.Signals): boolean {
  if (process.platform === "win32") {
    try {
      const killArgs = ["/PID", String(pid), "/T", "/F"];
      const killer = spawn("taskkill", killArgs, {
        stdio: "ignore",
        windowsHide: true,
      });
      killer.unref();
      return true;
    } catch {
      return false;
    }
  }

  try {
    // Detached processes are their own process-group leaders; kill group first.
    process.kill(-pid, signal);
    return true;
  } catch {
    try {
      process.kill(pid, signal);
      return true;
    } catch {
      return false;
    }
  }
}

function stopCvBackend(options?: { force?: boolean; reason?: string }): { ok: boolean; message: string } {
  const pid = cvBackendPid;
  if (!pid) {
    return { ok: true, message: "CV backend not running." };
  }

  if (!isPidAlive(pid)) {
    cvBackendPid = null;
    return { ok: true, message: `CV backend already exited (pid ${pid}).` };
  }

  const force = options?.force === true;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  const killed = killCvBackendPid(pid, signal);
  if (!killed) {
    return { ok: false, message: `Failed to stop CV backend (pid ${pid}).` };
  }

  cvBackendPid = null;
  const reason = options?.reason ? ` (${options.reason})` : "";
  return { ok: true, message: `Stopped CV backend (pid ${pid})${reason}.` };
}

function findProjectRoot(startDir: string): string | null {
  let dir = path.resolve(startDir);
  for (let i = 0; i < 8; i++) {
    const eyegazeEntry = path.join(dir, "eyegaze", "eye_tracker_service.py");
    if (fs.existsSync(eyegazeEntry)) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  return null;
}

function launchCvBackendInBackground(cameraIndex: number): { ok: boolean; message: string } {
  if (isPidAlive(cvBackendPid)) {
    return { ok: true, message: `CV backend already running (pid ${cvBackendPid}).` };
  }
  cvBackendPid = null;

  const root = findProjectRoot(__dirname);
  if (!root) {
    return { ok: false, message: "Cannot find project root from Electron runtime path." };
  }

  const eyegazeDir = path.join(root, "eyegaze");
  const scriptPath = path.join(eyegazeDir, "eye_tracker_service.py");
  const taskPath = path.join(root, "assets", "models", "face_landmarker.task");
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, message: `Missing backend script: ${scriptPath}` };
  }
  if (!fs.existsSync(taskPath)) {
    return { ok: false, message: `Missing face landmarker task: ${taskPath}` };
  }

  const logsDir = path.join(root, ".dist");
  try {
    fs.mkdirSync(logsDir, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Cannot create logs directory: ${message}` };
  }
  const logPath = path.join(logsDir, "eyegaze-backend.log");

  let pythonBin = path.join(eyegazeDir, ".venv", "bin", "python");
  if (process.platform === "win32") {
    pythonBin = path.join(eyegazeDir, ".venv", "Scripts", "python.exe");
  }
  if (!fs.existsSync(pythonBin)) {
    pythonBin = process.platform === "win32" ? "python" : "python3";
  }

  const args = [
    "-u",
    scriptPath,
    "--camera",
    String(cameraIndex),
    "--http-host",
    "127.0.0.1",
    "--http-port",
    "8767",
    "--http-streaming",
    "--no-cursor-move",
    "--face-landmarker-task",
    taskPath,
  ];

  try {
    const logFd = fs.openSync(logPath, "a");
    try {
      const child = spawn(pythonBin, args, {
        cwd: eyegazeDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      cvBackendPid = child.pid ?? null;
      child.on("exit", () => {
        if (cvBackendPid === child.pid) {
          cvBackendPid = null;
        }
      });
      child.on("error", () => {
        if (cvBackendPid === child.pid) {
          cvBackendPid = null;
        }
      });
      child.unref();
      return {
        ok: true,
        message: `Launched backend in background (pid ${child.pid ?? "unknown"}). Logs: ${logPath}`,
      };
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Failed to spawn backend: ${message}` };
  }
}

async function startCvBackendAndWait(cameraIndex: number): Promise<{ ok: boolean; message: string }> {
  const launch = launchCvBackendInBackground(cameraIndex);
  if (!launch.ok) {
    return launch;
  }

  const ready = await waitForBackendTcp("127.0.0.1", 8767, 45000, cvBackendPid);
  if (!ready) {
    stopCvBackend({ force: true, reason: "startup-timeout" });
    return {
      ok: false,
      message: `CV backend did not become reachable on 127.0.0.1:8767 within 45s. ${launch.message}`.trim(),
    };
  }
  return launch;
}

function createWindow(mode: 'full' | 'overlay' = 'full') {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth } = primaryDisplay.workAreaSize;

  console.log("Creating window with mode:", mode);
  currentMode = mode; // Update the global state

  const commonOptions = {
    backgroundColor: "#0b0b0b",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  };

  if (mode === 'overlay') {
    // Overlay mode
    win = new BrowserWindow({
      ...commonOptions,
      width: screenWidth,
      height: 60,
      x: 0,
      y: 0,
      frame: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: false,
      resizable: false,
      movable: true,
    });
  } else {
    // Full app mode - no native title bar; use app toolbar for minimize/close
    win = new BrowserWindow({
      ...commonOptions,
      width: 1100,
      height: 720,
      frame: false,
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : undefined,
      transparent: false,
      alwaysOnTop: false,
      resizable: true,
      movable: true,
    });
    win.center();
  }

  // Check if we're in development mode
  const isDev = !app.isPackaged;
  
  if (isDev) {
    win.loadURL('http://localhost:5173');
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(__dirname, "../index.html"));
  }

  win.on('closed', () => {
    win = null;
  });

  // Send overlay mode to renderer once it's ready
  win.webContents.on('did-finish-load', () => {
    const isOverlay = mode === 'overlay';
    console.log("Window loaded, sending overlay mode:", isOverlay);
    win?.webContents.send('overlay-mode-changed', isOverlay);
  });
}

// Toggle between overlay and full app mode
ipcMain.handle("app:toggle-overlay", (event) => {
  const newMode: 'full' | 'overlay' = currentMode === 'overlay' ? 'full' : 'overlay';
  const willBeOverlay = newMode === 'overlay';

  console.log("Toggle overlay: currentMode=", currentMode, "-> newMode=", newMode);

  // Use the window that sent this message (so we don't rely on global `win`)
  const senderWindow = BrowserWindow.fromWebContents(event.sender);
  if (!senderWindow || senderWindow.isDestroyed()) {
    console.log("No sender window, creating window in new mode");
    createWindow(newMode);
    return willBeOverlay;
  }

  isToggling = true;

  senderWindow.once('closed', () => {
    setTimeout(() => {
      createWindow(newMode);
      isToggling = false;
      console.log("New window created, mode=", newMode);
    }, 80);
  });

  senderWindow.close();

  return willBeOverlay;
});

// Get current overlay mode
ipcMain.handle("app:get-overlay-mode", () => {
  return currentMode === 'overlay';
});

ipcMain.handle("app:minimize", () => win?.minimize());
ipcMain.handle("app:maximize", () => {
  if (win?.isMaximized()) {
    win?.unmaximize();
  } else {
    win?.maximize();
  }
});
ipcMain.handle("app:close", () => win?.close());

ipcMain.handle("cv:start-backend", async (_event, args?: { camera?: number }) => {
  const requestedCamera = Number.isFinite(args?.camera) ? Number(args?.camera) : 0;
  const safeCamera = Math.max(0, Math.min(9, Math.trunc(requestedCamera)));
  try {
    return await startCvBackendAndWait(safeCamera);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
});

ipcMain.handle("cv:stop-backend", () => {
  return stopCvBackend({ force: true, reason: "ipc" });
});

ipcMain.handle("cv:is-backend-running", async () => {
  try {
    return await probeTcp("127.0.0.1", 8767, 350);
  } catch {
    return false;
  }
});

ipcMain.handle("snap:move-cursor", async (_event, args?: { x?: number; y?: number }) => {
  const x = Number(args?.x);
  const y = Number(args?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return { ok: false, message: "Invalid cursor coordinates." };
  }
  const moved = await moveCursorToScreenPoint(x, y);
  if (!moved) {
    return { ok: false, message: "Failed to move system cursor." };
  }
  return { ok: true, message: "Cursor moved." };
});

ipcMain.handle("snap:commit-nearest", async () => {
  return await commitNearestSystemSnapTarget();
});

ipcMain.handle("snap:get-cursor-screen-point", () => {
  return screen.getCursorScreenPoint();
});

ipcMain.handle("snap:list-system-targets", async () => {
  return await listSystemSnapTargets();
});

app.whenReady().then(() => {
  createWindow('full');

  const snapShortcutRegistered = globalShortcut.register(SNAP_GLOBAL_SHORTCUT, () => {
    void commitNearestSystemSnapTarget();
  });
  if (!snapShortcutRegistered) {
    console.warn(`Failed to register global snap shortcut: ${SNAP_GLOBAL_SHORTCUT}`);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow('full');
    }
  });
});

app.on('window-all-closed', () => {
  // Don't quit if we're in the middle of toggling overlay ↔ full (we're about to create a new window)
  if (isToggling) {
    return;
  }
  stopCvBackend({ force: true, reason: "window-all-closed" });
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on("before-quit", () => {
  globalShortcut.unregisterAll();
  stopCvBackend({ force: true, reason: "before-quit" });
});
