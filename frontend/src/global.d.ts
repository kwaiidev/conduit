export {};

declare global {
  type ElectronSystemSnapTarget = {
    id: string;
    kind: string;
    label: string;
    left: number;
    top: number;
    width: number;
    height: number;
  };

  type ElectronSystemSnapTargetsResponse = {
    ok: boolean;
    provider: string;
    reason?: string;
    targets: ElectronSystemSnapTarget[];
  };

  interface Window {
    electron: {
      minimize: () => Promise<void>;
      maximize: () => Promise<void>;
      close: () => Promise<void>;
      toggleOverlay: () => Promise<boolean>;
      getOverlayMode: () => Promise<boolean>;
      startCvBackend: (args?: { camera?: number }) => Promise<{ ok: boolean; message: string }>;
      stopCvBackend: () => Promise<{ ok: boolean; message: string }>;
      isCvBackendRunning: () => Promise<boolean>;
      moveCursorToScreenPoint: (args: { x: number; y: number }) => Promise<{ ok: boolean; message: string }>;
      commitNearestSnapTarget: () => Promise<{ ok: boolean; message: string; target?: { id: string; x: number; y: number } }>;
      getCursorScreenPoint: () => Promise<{ x: number; y: number }>;
      listSystemSnapTargets: () => Promise<ElectronSystemSnapTargetsResponse>;
      onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => void;
    };
  }
}
