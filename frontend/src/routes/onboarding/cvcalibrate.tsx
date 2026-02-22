import React from "react";
import { motion } from "motion/react";

const CV_API_BASE = "http://127.0.0.1:8767";
const STATUS_POLL_MS = 500;

type CalibrationPhase = "idle" | "booting" | "ready" | "error";

type CameraStatusPayload = {
  camera_ready?: boolean;
  camera_status?: string;
  camera_last_valid_frame_ms?: number;
};

async function callCvApi(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 2500);
  const headers = new Headers(init?.headers ?? undefined);
  if (init?.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  try {
    const response = await fetch(`${CV_API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers,
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response;
  } finally {
    window.clearTimeout(timeout);
  }
}

function describeCameraStatus(raw: string | undefined): string {
  if (!raw) {
    return "Waiting for camera service...";
  }
  const status = raw.toLowerCase();
  if (status === "ready") {
    return "Camera feed is stable. Hold your gaze on the target.";
  }
  if (status.startsWith("warming_up")) {
    return "Camera is warming up exposure and focus.";
  }
  if (status.startsWith("recovering")) {
    return "Recovering camera stream...";
  }
  if (status.includes("too_dark")) {
    return "Frame too dark. Increase lighting and face the camera.";
  }
  if (status.includes("overexposed")) {
    return "Frame too bright. Reduce backlight and retry.";
  }
  if (status.includes("low_contrast") || status.includes("low_dynamic_range")) {
    return "Waiting for a stable frame with enough contrast.";
  }
  if (status.includes("desaturated")) {
    return "Camera feed looks gray. Reinitializing until color/contrast returns.";
  }
  if (status.includes("stale")) {
    return "Camera frame appears frozen. Restarting capture automatically.";
  }
  if (status.includes("camera_open_failed")) {
    return "Camera open failed. Check OS camera permissions.";
  }
  if (status.includes("frame_missing")) {
    return "No frame yet. Hold still while camera initializes.";
  }
  return raw.replace(/_/g, " ");
}

export function CVCursorCalibrationCenter() {
  const [isCalibrating, setIsCalibrating] = React.useState(false);
  const [phase, setPhase] = React.useState<CalibrationPhase>("idle");
  const [statusMessage, setStatusMessage] = React.useState("Press Start to initialize the CV camera.");
  const [streamNonce, setStreamNonce] = React.useState(() => Date.now());
  const [streamAttempt, setStreamAttempt] = React.useState(0);
  const [lastFrameLoadMs, setLastFrameLoadMs] = React.useState(0);
  const refreshCooldownRef = React.useRef(0);
  const lastBackendFrameMsRef = React.useRef(0);

  const refreshStream = React.useCallback((reason: string) => {
    const now = Date.now();
    if (now - refreshCooldownRef.current < 700) {
      return;
    }
    refreshCooldownRef.current = now;
    setStreamAttempt(prev => prev + 1);
    setStatusMessage(reason);
  }, []);

  const startCalibration = async () => {
    setIsCalibrating(true);
    setPhase("booting");
    setStatusMessage("Starting camera stream...");
    setStreamNonce(Date.now());
    setStreamAttempt(0);
    setLastFrameLoadMs(0);
    lastBackendFrameMsRef.current = 0;
    try {
      await Promise.all([
        callCvApi("/stream/enable", { method: "POST" }),
        callCvApi("/processing/enable", { method: "POST" }),
      ]);
    } catch (error) {
      setPhase("error");
      setStatusMessage(
        "Cannot reach eye-gaze service at 127.0.0.1:8767. Start eyegaze service and retry."
      );
      console.error("[CV Calibration] startup request failed:", error);
    }
  };

  const stopCalibration = async () => {
    setIsCalibrating(false);
    setPhase("idle");
    setStatusMessage("Calibration paused.");
    try {
      await callCvApi("/processing/disable", { method: "POST" });
    } catch (error) {
      console.error("[CV Calibration] failed to stop processing:", error);
    }
  };

  React.useEffect(() => {
    if (!isCalibrating) {
      return undefined;
    }

    let active = true;

    const pollStatus = async () => {
      try {
        const response = await callCvApi("/status");
        const payload = (await response.json()) as CameraStatusPayload;
        if (!active) {
          return;
        }
        if (typeof payload.camera_last_valid_frame_ms === "number") {
          lastBackendFrameMsRef.current = payload.camera_last_valid_frame_ms;
        }
        const cameraReady = Boolean(payload.camera_ready);
        setPhase(cameraReady ? "ready" : "booting");
        if (cameraReady) {
          const nowMs = Date.now();
          const backendFrameAge = nowMs - (lastBackendFrameMsRef.current || nowMs);
          if (backendFrameAge > 1800) {
            refreshStream("Stream reconnecting to latest camera frame...");
            setPhase("booting");
            return;
          }
          setStatusMessage("Camera ready. Hold your gaze at the center target.");
          return;
        }
        setStatusMessage(describeCameraStatus(payload.camera_status));
      } catch (error) {
        if (!active) {
          return;
        }
        setPhase("error");
        setStatusMessage(
          "Camera status unavailable. Verify eyegaze service is running and camera permission is granted."
        );
        console.error("[CV Calibration] status poll failed:", error);
      }
    };

    void pollStatus();
    const timer = window.setInterval(() => {
      void pollStatus();
    }, STATUS_POLL_MS);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [isCalibrating, refreshStream]);

  React.useEffect(() => {
    if (!isCalibrating) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      const now = Date.now();
      if (lastFrameLoadMs > 0 && now - lastFrameLoadMs < 2500) {
        return;
      }
      refreshStream("Waiting for camera frames. Reconnecting stream...");
    }, 1800);
    return () => window.clearInterval(timer);
  }, [isCalibrating, lastFrameLoadMs, refreshStream]);

  const showOverlay = isCalibrating && phase !== "ready";

  return (
    <div style={styles.container}>
      <div style={styles.headerBlock}>
        <p style={styles.subtitle}>
          Keep your face centered. Calibration starts only after a stable camera frame is available.
        </p>
      </div>

      <div style={styles.previewCard}>
        <div style={styles.previewInner}>
          {isCalibrating ? (
            <>
              <img
                src={`${CV_API_BASE}/video?nonce=${streamNonce}&attempt=${streamAttempt}`}
                alt="Live camera preview for eye calibration"
                style={styles.previewImage}
                draggable={false}
                onLoad={() => setLastFrameLoadMs(Date.now())}
                onError={() => {
                  refreshStream("Camera stream disconnected. Reconnecting...");
                }}
              />
              {showOverlay ? (
                <div style={styles.statusOverlay}>
                  <motion.span
                    style={styles.statusDot}
                    animate={{ scale: [1, 1.25, 1], opacity: [1, 0.55, 1] }}
                    transition={{ duration: 1.1, repeat: Infinity }}
                  />
                  <span style={styles.statusOverlayText}>{statusMessage}</span>
                </div>
              ) : null}
            </>
          ) : (
            <>
              <div style={styles.previewPlaceholder} />
              <div style={styles.placeholderText}>Camera preview appears after Start.</div>
            </>
          )}

          <div style={styles.target}>
            <div style={styles.targetRing} />
            <div style={styles.targetDot} />
            <div style={styles.crosshairV} />
            <div style={styles.crosshairH} />
          </div>
        </div>
      </div>

      <div style={styles.buttonWrapper}>
        {!isCalibrating ? (
          <button type="button" onClick={startCalibration} style={styles.startButton}>
            Start
          </button>
        ) : (
          <button
            type="button"
            onClick={stopCalibration}
            style={styles.statusBar}
            title="Click to stop calibration"
          >
            <motion.span
              style={styles.statusDot}
              animate={{ scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            {phase === "ready" ? "Calibrating" : "Initializing"}
          </button>
        )}
      </div>

      <p style={styles.statusText}>{statusMessage}</p>
    </div>
  );
}

const BUTTON_WIDTH = 180;
const BUTTON_HEIGHT = 42;
const PREVIEW_WIDTH = 520;

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 16,
    width: "100%",
  },
  headerBlock: {
    width: PREVIEW_WIDTH,
    textAlign: "center",
  },
  subtitle: {
    margin: "8px 0 0 0",
    fontSize: 14,
    color: "var(--text-secondary)",
  },
  previewCard: {
    width: PREVIEW_WIDTH,
    borderRadius: 20,
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    padding: 16,
  },
  previewInner: {
    height: 280,
    borderRadius: 16,
    border: "1px solid var(--border)",
    background:
      "radial-gradient(circle at 20% 20%, rgba(0,194,170,0.15), transparent 50%), radial-gradient(circle at 80% 20%, rgba(0,122,255,0.13), transparent 48%), #0f1620",
    display: "grid",
    placeItems: "center",
    position: "relative",
    overflow: "hidden",
  },
  previewImage: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    objectFit: "cover",
    filter: "saturate(1.05)",
  },
  previewPlaceholder: {
    position: "absolute",
    inset: 0,
    background:
      "radial-gradient(circle at 22% 20%, rgba(0,194,170,0.15), transparent 50%), radial-gradient(circle at 80% 18%, rgba(0,122,255,0.14), transparent 45%), #0f1620",
  },
  placeholderText: {
    position: "absolute",
    bottom: 16,
    fontSize: 12,
    fontWeight: 800,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "rgba(255,255,255,0.78)",
  },
  statusOverlay: {
    position: "absolute",
    left: 12,
    right: 12,
    top: 12,
    minHeight: 42,
    borderRadius: 12,
    background: "rgba(10, 15, 20, 0.82)",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "#ffffff",
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "10px 12px",
    zIndex: 2,
  },
  statusOverlayText: {
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "-0.01em",
    color: "rgba(255,255,255,0.9)",
  },
  target: {
    position: "relative",
    width: 90,
    height: 90,
    display: "grid",
    placeItems: "center",
    zIndex: 3,
    pointerEvents: "none",
  },
  targetRing: {
    position: "absolute",
    width: 72,
    height: 72,
    borderRadius: "50%",
    border: "2px solid rgba(255,45,141,0.6)",
    boxShadow: "0 0 0 12px rgba(255,45,141,0.14)",
  },
  targetDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "#FF2D8D",
  },
  crosshairV: {
    position: "absolute",
    width: 2,
    height: 56,
    background: "rgba(255,45,141,0.42)",
    borderRadius: 2,
  },
  crosshairH: {
    position: "absolute",
    height: 2,
    width: 56,
    background: "rgba(255,45,141,0.42)",
    borderRadius: 2,
  },
  buttonWrapper: {
    width: PREVIEW_WIDTH,
    display: "flex",
    justifyContent: "center",
  },
  startButton: {
    width: BUTTON_WIDTH,
    height: BUTTON_HEIGHT,
    borderRadius: 16,
    border: "1px solid rgba(255,45,141,0.35)",
    background: "#FF2D8D",
    color: "white",
    fontWeight: 900,
    cursor: "pointer",
    boxShadow: "0 10px 25px rgba(255, 45, 141, 0.18)",
  },
  statusBar: {
    position: "relative",
    width: BUTTON_WIDTH,
    height: BUTTON_HEIGHT,
    borderRadius: 16,
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    fontWeight: 900,
    cursor: "pointer",
    overflow: "hidden",
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#FF2D8D",
    boxShadow: "0 0 0 6px rgba(255,45,141,0.12)",
    zIndex: 1,
    flexShrink: 0,
  },
  statusText: {
    margin: 0,
    minHeight: 22,
    fontSize: 13,
    color: "var(--text-secondary)",
    textAlign: "center",
    width: PREVIEW_WIDTH,
  },
};
