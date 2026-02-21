import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electron", {
  minimize: () => ipcRenderer.invoke("app:minimize"),
  maximize: () => ipcRenderer.invoke("app:maximize"),
  close: () => ipcRenderer.invoke("app:close"),
  toggleOverlay: () => ipcRenderer.invoke("app:toggle-overlay"),
  getOverlayMode: () => ipcRenderer.invoke("app:get-overlay-mode"),
  onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => {
    ipcRenderer.on('overlay-mode-changed', (_event, isOverlay) => callback(isOverlay));
  },
});