import { app, BrowserWindow, ipcMain, screen } from "electron";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import net from "net";

let win: BrowserWindow | null = null;
let currentMode: 'full' | 'overlay' = 'full';
let isToggling = false;
let cvBackendPid: number | null = null;

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

app.whenReady().then(() => {
  createWindow('full');

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
  stopCvBackend({ force: true, reason: "before-quit" });
});
