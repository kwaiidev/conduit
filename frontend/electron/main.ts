import { app, BrowserWindow, ipcMain, screen } from "electron";
import path from "path";

let win: BrowserWindow | null = null;
let currentMode: 'full' | 'overlay' = 'full';
let isToggling = false;

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
  if (process.platform !== 'darwin') {
    app.quit();
  }
});