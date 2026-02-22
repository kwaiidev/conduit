import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electron", {
  minimize: () => ipcRenderer.invoke("app:minimize"),
  maximize: () => ipcRenderer.invoke("app:maximize"),
  close: () => ipcRenderer.invoke("app:close"),
  toggleOverlay: () => ipcRenderer.invoke("app:toggle-overlay"),
  getOverlayMode: () => ipcRenderer.invoke("app:get-overlay-mode"),
  startCvBackend: (args?: { camera?: number }) => ipcRenderer.invoke("cv:start-backend", args),
  stopCvBackend: () => ipcRenderer.invoke("cv:stop-backend"),
  isCvBackendRunning: () => ipcRenderer.invoke("cv:is-backend-running"),
  moveCursorToScreenPoint: (args: { x: number; y: number }) => ipcRenderer.invoke("snap:move-cursor", args),
  commitNearestSnapTarget: () => ipcRenderer.invoke("snap:commit-nearest"),
  getCursorScreenPoint: () => ipcRenderer.invoke("snap:get-cursor-screen-point"),
  listSystemSnapTargets: () => ipcRenderer.invoke("snap:list-system-targets"),
  onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => {
    ipcRenderer.on('overlay-mode-changed', (_event, isOverlay) => callback(isOverlay));
  },
});
