import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from "react";
import { motion } from "motion/react";
import { disableEEG, enableEEG } from "../../lib/eeg";
export function EegTrainingCard({ instruction }) {
    const [isTraining, setIsTraining] = React.useState(false);
    const [isStarting, setIsStarting] = React.useState(false);
    const [statusMessage, setStatusMessage] = React.useState("Press Start to launch EEG service and begin.");
    const startTraining = React.useCallback(async () => {
        setIsStarting(true);
        setStatusMessage("Launching EEG backend...");
        try {
            const launchResult = await window.electron.startEegBackend();
            if (!launchResult.ok) {
                setStatusMessage(`Cannot start EEG backend: ${launchResult.message}`);
                return;
            }
            await enableEEG();
            setIsTraining(true);
            setStatusMessage("EEG service active. Perform the exercise for 3 seconds.");
        }
        catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            setStatusMessage(`EEG start failed: ${detail}`);
        }
        finally {
            setIsStarting(false);
        }
    }, []);
    const stopTraining = React.useCallback(async () => {
        setIsTraining(false);
        setStatusMessage("Training paused.");
        try {
            await disableEEG();
        }
        catch {
            // no-op: keep UI responsive even if backend endpoint is temporarily unavailable
        }
    }, []);
    return (_jsxs("div", { style: styles.container, children: [_jsxs("div", { style: styles.headerBlock, children: [_jsx("p", { style: styles.subtitle, children: instruction }), _jsx("p", { style: styles.statusText, children: statusMessage })] }), _jsx("div", { style: styles.gifCard, children: _jsx("div", { style: styles.gifInner, children: _jsx("div", { style: styles.gifText, children: "EEG Training Cue" }) }) }), _jsx("div", { style: styles.buttonWrapper, children: !isTraining ? (_jsx("button", { type: "button", onClick: startTraining, style: {
                        ...styles.startButton,
                        ...(isStarting ? styles.buttonDisabled : {}),
                    }, disabled: isStarting, children: isStarting ? "Starting..." : "Start" })) : (_jsxs("button", { type: "button", onClick: stopTraining, style: styles.statusBar, children: [_jsx(motion.span, { style: styles.statusDot, animate: { scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }, transition: { duration: 1.2, repeat: Infinity } }), "Training...", _jsx("div", { style: styles.fakeFill })] })) })] }));
}
const BUTTON_WIDTH = 180;
const BUTTON_HEIGHT = 42;
const GIF_WIDTH = 420;
const styles = {
    container: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 24,
        width: "100%",
    },
    headerBlock: {
        width: GIF_WIDTH,
        textAlign: "center",
    },
    subtitle: {
        margin: 0,
        fontSize: 14,
        color: "var(--text-secondary)",
    },
    statusText: {
        margin: "10px 0 0 0",
        fontSize: 12,
        color: "var(--text-secondary)",
    },
    gifCard: {
        width: GIF_WIDTH,
        borderRadius: 20,
        border: "1px solid var(--border)",
        background: "var(--bg-secondary)",
        padding: 16,
    },
    gifInner: {
        height: 240,
        borderRadius: 16,
        border: "1px dashed var(--border)",
        background: "var(--bg-tertiary)",
        display: "grid",
        placeItems: "center",
    },
    gifText: {
        fontSize: 12,
        fontWeight: 900,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
        color: "var(--text-secondary)",
    },
    buttonWrapper: {
        width: GIF_WIDTH,
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
    buttonDisabled: {
        cursor: "default",
        opacity: 0.65,
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
    },
    fakeFill: {
        position: "absolute",
        inset: 0,
        width: "60%",
        background: "rgba(255,45,141,0.12)",
        zIndex: 0,
    },
};
