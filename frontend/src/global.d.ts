export {};

declare global {
  type ElectronSystemSnapTarget = {
    id: string;
    kind: string;
    label: string;
    app: string;
    appPid: number;
    left: number;
    top: number;
    width: number;
    height: number;
    supportsPress: boolean;
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
      toggleOverlay: (args?: { targetPath?: string }) => Promise<boolean>;
      getOverlayMode: () => Promise<boolean>;
      consumePendingRoute: () => Promise<string | null>;
      startCvBackend: (args?: { camera?: number }) => Promise<{ ok: boolean; message: string }>;
      stopCvBackend: () => Promise<{ ok: boolean; message: string }>;
      isCvBackendRunning: () => Promise<boolean>;
      getCvStatusUrl: () => Promise<string>;
      startVoiceBackend: () => Promise<{ ok: boolean; message: string }>;
      stopVoiceBackend: () => Promise<{ ok: boolean; message: string }>;
      isVoiceBackendRunning: () => Promise<boolean>;
      startSignBackend: () => Promise<{ ok: boolean; message: string }>;
      stopSignBackend: () => Promise<{ ok: boolean; message: string }>;
      isSignBackendRunning: () => Promise<boolean>;
      startEegBackend: () => Promise<{ ok: boolean; message: string }>;
      stopEegBackend: () => Promise<{ ok: boolean; message: string }>;
      isEegBackendRunning: () => Promise<boolean>;
      moveCursorToScreenPoint: (args: { x: number; y: number }) => Promise<{ ok: boolean; message: string }>;
      commitNearestSnapTarget: () => Promise<{ ok: boolean; message: string; target?: { id: string; x: number; y: number } }>;
      commitSystemSnapTarget: (
        args: { x: number; y: number }
      ) => Promise<{ ok: boolean; message: string; reason?: string; target?: { id: string; x: number; y: number; label?: string; app?: string } }>;
      getCursorScreenPoint: () => Promise<{ x: number; y: number }>;
      listSystemSnapTargets: () => Promise<ElectronSystemSnapTargetsResponse>;
      onOverlayModeChanged: (callback: (isOverlay: boolean) => void) => void;
    };
  }
}
