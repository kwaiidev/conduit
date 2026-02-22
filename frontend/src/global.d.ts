export {};

declare global {
  interface Window {
    electron: {
      minimize: () => Promise<void>;
      maximize: () => Promise<void>;
      close: () => Promise<void>;
      toggleOverlay: () => Promise<boolean>;
      getOverlayMode: () => Promise<boolean>;
      startCvBackend: (args?: { camera?: number }) => Promise<{ ok: boolean; message: string }>;
      onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => void;
    };
  }
}
