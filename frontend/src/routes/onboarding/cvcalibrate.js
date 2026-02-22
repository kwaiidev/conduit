import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import React from "react";
import { motion } from "motion/react";
const CV_API_BASE = "http://127.0.0.1:8767";
const STATUS_POLL_MS = 500;
const CENTER_LOCK_MS = 1800;
const CENTER_NORM_RADIUS = 0.2;
const CENTER_PROGRESS_DECAY = 0.08;
async function callCvApi(path, init) {
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
    }
    finally {
        window.clearTimeout(timeout);
    }
}
async function setCvState(state) {
    await callCvApi("/state", {
        method: "POST",
        body: JSON.stringify(state),
    });
}
async function setCvStateWithRetry(state, retries = 6) {
    let lastError = null;
    for (let attempt = 0; attempt < retries; attempt += 1) {
        try {
            await setCvState(state);
            return;
        }
        catch (error) {
            lastError = error;
            await new Promise(resolve => window.setTimeout(resolve, 300));
        }
    }
    throw lastError ?? new Error("Unable to set CV state.");
}
function describeCameraStatus(raw) {
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
    const [backendReachable, setBackendReachable] = React.useState(false);
    const [isOnCenter, setIsOnCenter] = React.useState(false);
    const [centerLocked, setCenterLocked] = React.useState(false);
    const [phase, setPhase] = React.useState("idle");
    const [statusMessage, setStatusMessage] = React.useState("Press Start to initialize the CV camera.");
    const [streamNonce, setStreamNonce] = React.useState(() => Date.now());
    const [streamAttempt, setStreamAttempt] = React.useState(0);
    const [lastFrameLoadMs, setLastFrameLoadMs] = React.useState(0);
    const [centerLockProgress, setCenterLockProgress] = React.useState(0);
    const refreshCooldownRef = React.useRef(0);
    const lastBackendFrameMsRef = React.useRef(0);
    const centerReadySinceMsRef = React.useRef(null);
    const statusErrorStreakRef = React.useRef(0);
    const refreshStream = React.useCallback((reason) => {
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
        setBackendReachable(false);
        setIsOnCenter(false);
        setCenterLocked(false);
        setPhase("booting");
        setStatusMessage("Starting camera stream...");
        setStreamNonce(Date.now());
        setStreamAttempt(0);
        setLastFrameLoadMs(0);
        setCenterLockProgress(0);
        centerReadySinceMsRef.current = null;
        statusErrorStreakRef.current = 0;
        lastBackendFrameMsRef.current = 0;
        let launchDebugMessage = "";
        try {
            setStatusMessage("Launching CV backend...");
            const launchResult = await window.electron.startCvBackend({ camera: 0 });
            launchDebugMessage = launchResult.message ?? "";
            if (!launchResult.ok) {
                throw new Error(launchResult.message || "Failed to launch backend.");
            }
            console.info("[CV Calibration] backend launch:", launchResult.message);
            setBackendReachable(true);
            await setCvStateWithRetry({
                streaming: true,
                processing: true,
                mouse_control: false,
            });
            setStatusMessage("Camera online. Look at the center target to calibrate.");
        }
        catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            setBackendReachable(false);
            setIsOnCenter(false);
            setCenterLocked(false);
            setPhase("error");
            setStatusMessage(`Cannot start eye-gaze service. ${detail}`.trim());
            if (launchDebugMessage) {
                console.error("[CV Calibration] backend launch details:", launchDebugMessage);
            }
            console.error("[CV Calibration] startup request failed:", error);
        }
    };
    const stopCalibration = async () => {
        setIsCalibrating(false);
        setBackendReachable(false);
        setIsOnCenter(false);
        setCenterLocked(false);
        setPhase("idle");
        setStatusMessage("Calibration paused.");
        setCenterLockProgress(0);
        centerReadySinceMsRef.current = null;
        statusErrorStreakRef.current = 0;
        try {
            await setCvState({
                processing: false,
                mouse_control: false,
            });
        }
        catch (error) {
            console.error("[CV Calibration] failed to stop processing:", error);
        }
    };
    React.useEffect(() => {
        if (!isCalibrating || !backendReachable) {
            return undefined;
        }
        let active = true;
        const pollStatus = async () => {
            try {
                const response = await callCvApi("/status");
                const payload = (await response.json());
                if (!active) {
                    return;
                }
                statusErrorStreakRef.current = 0;
                if (typeof payload.camera_last_valid_frame_ms === "number") {
                    lastBackendFrameMsRef.current = payload.camera_last_valid_frame_ms;
                }
                const cameraReady = Boolean(payload.camera_ready);
                if (cameraReady) {
                    if (centerLocked) {
                        setIsOnCenter(true);
                        setCenterLockProgress(1);
                        setPhase("ready");
                        setStatusMessage("Center lock complete. Calibration is ready.");
                        return;
                    }
                    const nowMs = Date.now();
                    const backendFrameAge = nowMs - (lastBackendFrameMsRef.current || nowMs);
                    if (backendFrameAge > 1800) {
                        centerReadySinceMsRef.current = null;
                        setCenterLockProgress(0);
                        setIsOnCenter(false);
                        refreshStream("Stream reconnecting to latest camera frame...");
                        setPhase("booting");
                        return;
                    }
                    const event = payload.last_event ?? null;
                    const intent = event?.intent ?? "";
                    const payloadXNorm = event?.payload?.x_norm;
                    const payloadYNorm = event?.payload?.y_norm;
                    const payloadTargetX = event?.payload?.target_x;
                    const payloadTargetY = event?.payload?.target_y;
                    let xNorm = payloadXNorm;
                    let yNorm = payloadYNorm;
                    if ((typeof xNorm !== "number" || typeof yNorm !== "number")
                        && typeof payloadTargetX === "number"
                        && typeof payloadTargetY === "number"
                        && typeof payload.screen_width === "number"
                        && typeof payload.screen_height === "number"
                        && payload.screen_width > 1
                        && payload.screen_height > 1) {
                        xNorm = Math.max(0, Math.min(1, payloadTargetX / (payload.screen_width - 1)));
                        yNorm = Math.max(0, Math.min(1, payloadTargetY / (payload.screen_height - 1)));
                    }
                    const hasNorm = typeof xNorm === "number" && typeof yNorm === "number";
                    if (!hasNorm) {
                        centerReadySinceMsRef.current = null;
                        setIsOnCenter(false);
                        setCenterLockProgress(prev => Math.max(0, prev - CENTER_PROGRESS_DECAY));
                        setPhase("booting");
                        if (intent === "noop") {
                            setStatusMessage("Face not detected yet. Keep your face centered and look at the target.");
                        }
                        else {
                            setStatusMessage("Camera ready. Keep your eyes open and look at the center target.");
                        }
                        return;
                    }
                    const dx = Number(xNorm) - 0.5;
                    const dy = Number(yNorm) - 0.5;
                    const dist = Math.hypot(dx, dy);
                    const onCenter = dist <= CENTER_NORM_RADIUS;
                    setIsOnCenter(onCenter);
                    if (onCenter) {
                        if (centerReadySinceMsRef.current === null) {
                            centerReadySinceMsRef.current = nowMs;
                        }
                        const elapsed = nowMs - centerReadySinceMsRef.current;
                        const progress = Math.max(0, Math.min(1, elapsed / CENTER_LOCK_MS));
                        setCenterLockProgress(progress);
                        if (progress >= 1) {
                            setCenterLocked(true);
                            setPhase("ready");
                            setStatusMessage("Center lock complete. Calibration is ready.");
                        }
                        else {
                            setPhase("booting");
                            const secondsLeft = Math.max(1, Math.ceil((CENTER_LOCK_MS - elapsed) / 1000));
                            setStatusMessage(`Hold your gaze on center (${secondsLeft}s)`);
                        }
                    }
                    else {
                        centerReadySinceMsRef.current = null;
                        setCenterLockProgress(prev => Math.max(0, prev - CENTER_PROGRESS_DECAY));
                        setPhase("booting");
                        setStatusMessage("Move gaze to the center target.");
                    }
                    return;
                }
                centerReadySinceMsRef.current = null;
                setIsOnCenter(false);
                setCenterLockProgress(0);
                setPhase("booting");
                setStatusMessage(describeCameraStatus(payload.camera_status));
            }
            catch (error) {
                if (!active) {
                    return;
                }
                statusErrorStreakRef.current += 1;
                setIsOnCenter(false);
                centerReadySinceMsRef.current = null;
                setCenterLockProgress(prev => Math.max(0, prev - CENTER_PROGRESS_DECAY));
                setPhase("booting");
                setStatusMessage("Reconnecting to camera service...");
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
    }, [isCalibrating, backendReachable, centerLocked, refreshStream]);
    React.useEffect(() => {
        if (!isCalibrating || !backendReachable) {
            return undefined;
        }
        const onKeyDown = (event) => {
            if (event.repeat) {
                return;
            }
            if (event.key.toLowerCase() !== "c") {
                return;
            }
            event.preventDefault();
            void (async () => {
                try {
                    await callCvApi("/calibrate/center", { method: "POST" });
                    setStatusMessage("Manual center calibration requested (C). Hold gaze at center.");
                }
                catch (error) {
                    console.error("[CV Calibration] center calibration request failed:", error);
                    setStatusMessage("Center calibration request failed. Check camera service.");
                }
            })();
        };
        window.addEventListener("keydown", onKeyDown);
        return () => {
            window.removeEventListener("keydown", onKeyDown);
        };
    }, [isCalibrating, backendReachable]);
    React.useEffect(() => {
        if (!isCalibrating || !backendReachable) {
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
    }, [isCalibrating, backendReachable, lastFrameLoadMs, refreshStream]);
    const showOverlay = isCalibrating && (phase !== "ready" || !backendReachable);
    return (_jsxs("div", { style: styles.container, children: [_jsx("div", { style: styles.headerBlock, children: _jsx("p", { style: styles.subtitle, children: "Keep your face centered. Press C anytime during this step to request center calibration." }) }), _jsx("div", { style: styles.previewCard, children: _jsxs("div", { style: styles.previewInner, children: [isCalibrating && backendReachable ? (_jsxs(_Fragment, { children: [_jsx("img", { src: `${CV_API_BASE}/video?nonce=${streamNonce}&attempt=${streamAttempt}`, alt: "Live camera preview for eye calibration", style: styles.previewImage, draggable: false, onLoad: () => setLastFrameLoadMs(Date.now()), onError: () => {
                                        refreshStream("Camera stream disconnected. Reconnecting...");
                                    } }), showOverlay ? (_jsxs("div", { style: styles.statusOverlay, children: [_jsx(motion.span, { style: styles.statusDot, animate: { scale: [1, 1.25, 1], opacity: [1, 0.55, 1] }, transition: { duration: 1.1, repeat: Infinity } }), _jsx("span", { style: styles.statusOverlayText, children: statusMessage })] })) : null] })) : (_jsxs(_Fragment, { children: [_jsx("div", { style: styles.previewPlaceholder }), _jsx("div", { style: styles.placeholderText, children: isCalibrating ? "Launching camera backend..." : "Camera preview appears after Start." })] })), _jsxs("div", { style: styles.target, children: [_jsx("div", { style: {
                                        ...styles.targetProgress,
                                        background: `conic-gradient(rgba(0,194,170,0.95) ${Math.round(centerLockProgress * 360)}deg, rgba(255,255,255,0.14) 0deg)`,
                                    } }), _jsx("div", { style: styles.targetRing }), _jsx("div", { style: {
                                        ...styles.targetDot,
                                        background: isOnCenter || centerLocked ? "#00c2aa" : "#FF2D8D",
                                    } }), _jsx("div", { style: styles.crosshairV }), _jsx("div", { style: styles.crosshairH })] })] }) }), _jsx("div", { style: styles.buttonWrapper, children: !isCalibrating ? (_jsx("button", { type: "button", onClick: startCalibration, style: styles.startButton, children: "Start" })) : (_jsxs("button", { type: "button", onClick: stopCalibration, style: styles.statusBar, title: "Click to stop calibration", children: [_jsx(motion.span, { style: styles.statusDot, animate: { scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }, transition: { duration: 1.2, repeat: Infinity } }), !backendReachable
                            ? "Starting"
                            : centerLocked || phase === "ready"
                                ? "Center Locked"
                                : isOnCenter
                                    ? "Hold Center"
                                    : "Find Center"] })) }), _jsx("p", { style: styles.statusText, children: statusMessage })] }));
}
const BUTTON_WIDTH = 180;
const BUTTON_HEIGHT = 42;
const PREVIEW_WIDTH = 520;
const styles = {
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
        background: "radial-gradient(circle at 20% 20%, rgba(0,194,170,0.15), transparent 50%), radial-gradient(circle at 80% 20%, rgba(0,122,255,0.13), transparent 48%), #0f1620",
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
        background: "radial-gradient(circle at 22% 20%, rgba(0,194,170,0.15), transparent 50%), radial-gradient(circle at 80% 18%, rgba(0,122,255,0.14), transparent 45%), #0f1620",
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
    targetProgress: {
        position: "absolute",
        width: 78,
        height: 78,
        borderRadius: "50%",
        maskImage: "radial-gradient(circle, transparent 59%, black 61%)",
        WebkitMaskImage: "radial-gradient(circle, transparent 59%, black 61%)",
        transition: "background 120ms linear",
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
