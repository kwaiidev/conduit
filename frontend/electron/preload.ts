import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electron", {
  minimize: () => ipcRenderer.invoke("app:minimize"),
  maximize: () => ipcRenderer.invoke("app:maximize"),
  close: () => ipcRenderer.invoke("app:close"),
  toggleOverlay: (args?: { targetPath?: string }) => ipcRenderer.invoke("app:toggle-overlay", args),
  getOverlayMode: () => ipcRenderer.invoke("app:get-overlay-mode"),
  consumePendingRoute: () => ipcRenderer.invoke("app:consume-pending-route"),
  startCvBackend: (args?: { camera?: number }) => ipcRenderer.invoke("cv:start-backend", args),
  stopCvBackend: () => ipcRenderer.invoke("cv:stop-backend"),
  isCvBackendRunning: () => ipcRenderer.invoke("cv:is-backend-running"),
  startVoiceBackend: () => ipcRenderer.invoke("voice:start-backend"),
  stopVoiceBackend: () => ipcRenderer.invoke("voice:stop-backend"),
  isVoiceBackendRunning: () => ipcRenderer.invoke("voice:is-backend-running"),
  startSignBackend: () => ipcRenderer.invoke("sign:start-backend"),
  stopSignBackend: () => ipcRenderer.invoke("sign:stop-backend"),
  isSignBackendRunning: () => ipcRenderer.invoke("sign:is-backend-running"),
  startEegBackend: () => ipcRenderer.invoke("eeg:start-backend"),
  stopEegBackend: () => ipcRenderer.invoke("eeg:stop-backend"),
  isEegBackendRunning: () => ipcRenderer.invoke("eeg:is-backend-running"),
  moveCursorToScreenPoint: (args: { x: number; y: number }) => ipcRenderer.invoke("snap:move-cursor", args),
  commitNearestSnapTarget: () => ipcRenderer.invoke("snap:commit-nearest"),
  commitSystemSnapTarget: (args: { x: number; y: number }) => ipcRenderer.invoke("snap:commit-target", args),
  getCursorScreenPoint: () => ipcRenderer.invoke("snap:get-cursor-screen-point"),
  listSystemSnapTargets: () => ipcRenderer.invoke("snap:list-system-targets"),
  onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => {
    ipcRenderer.on('overlay-mode-changed', (_event, isOverlay) => callback(isOverlay));
  },
});
