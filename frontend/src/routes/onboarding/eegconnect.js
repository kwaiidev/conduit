import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React from "react";
import { disableEEG, enableEEG } from "../../lib/eeg";
import { setModalityPreference } from "../../state/modalityPreferences";
export function EEGConnectStep() {
    const [isBooting, setIsBooting] = React.useState(false);
    const [isReady, setIsReady] = React.useState(false);
    const [statusMessage, setStatusMessage] = React.useState("Press Start to launch the Muse Bluetooth stream and wait for EEG tracking confirmation.");
    const startService = React.useCallback(async () => {
        setIsBooting(true);
        setStatusMessage("Starting Muse Bluetooth stream and waiting for EEG stream confirmation...");
        try {
            const launchResult = await window.electron.startEegBackend();
            if (!launchResult.ok) {
                setIsReady(false);
                setStatusMessage(`Unable to launch EEG backend: ${launchResult.message}`);
                setModalityPreference("eeg-select", false);
                return;
            }
            await enableEEG();
            setIsReady(true);
            setStatusMessage("EEG stream confirmed. Backend is active.");
            setModalityPreference("eeg-select", true);
        }
        catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            setIsReady(false);
            setStatusMessage(`EEG startup failed: ${detail}`);
            setModalityPreference("eeg-select", false);
        }
        finally {
            setIsBooting(false);
        }
    }, []);
    const pauseSession = React.useCallback(async () => {
        setIsReady(false);
        setStatusMessage("EEG session paused.");
        setModalityPreference("eeg-select", false);
        try {
            await disableEEG();
        }
        catch {
            // no-op
        }
    }, []);
    React.useEffect(() => {
        let cancelled = false;
        const check = async () => {
            try {
                const running = await window.electron.isEegBackendRunning();
                if (cancelled || !running) {
                    return;
                }
                setIsReady(true);
                setStatusMessage("EEG stream already confirmed. Continue to baseline calibration.");
                setModalityPreference("eeg-select", true);
            }
            catch {
                // no-op
            }
        };
        void check();
        return () => {
            cancelled = true;
        };
    }, []);
    return (_jsx("div", { style: styles.container, children: _jsxs("div", { style: styles.card, children: [_jsx("h3", { style: styles.title, children: "EEG Service" }), _jsx("p", { style: styles.description, children: "This step turns on Muse Bluetooth, waits for headset streaming, and then enables EEG tracking." }), _jsx("p", { style: styles.status, children: statusMessage }), _jsx("div", { style: styles.controls, children: !isReady ? (_jsx("button", { type: "button", onClick: startService, style: {
                            ...styles.startButton,
                            ...(isBooting ? styles.disabledButton : {}),
                        }, disabled: isBooting, children: isBooting ? "Starting..." : "Start" })) : (_jsx("button", { type: "button", onClick: pauseSession, style: styles.stopButton, children: "Pause" })) })] }) }));
}
const styles = {
    container: {
        display: "flex",
        justifyContent: "center",
        width: "100%",
    },
    card: {
        width: 520,
        borderRadius: 18,
        border: "1px solid var(--border)",
        background: "var(--bg-secondary)",
        padding: 20,
        display: "flex",
        flexDirection: "column",
        gap: 12,
    },
    title: {
        margin: 0,
        fontSize: 18,
        color: "var(--text-primary)",
    },
    description: {
        margin: 0,
        fontSize: 14,
        color: "var(--text-secondary)",
    },
    status: {
        margin: 0,
        fontSize: 12,
        color: "var(--text-secondary)",
        minHeight: 18,
    },
    controls: {
        display: "flex",
        gap: 10,
        justifyContent: "center",
    },
    startButton: {
        height: 40,
        minWidth: 140,
        borderRadius: 14,
        border: "1px solid rgba(255,45,141,0.35)",
        background: "#FF2D8D",
        color: "#fff",
        fontWeight: 800,
        cursor: "pointer",
    },
    stopButton: {
        height: 40,
        minWidth: 140,
        borderRadius: 14,
        border: "1px solid var(--border)",
        background: "var(--bg-tertiary)",
        color: "var(--text-primary)",
        fontWeight: 700,
        cursor: "pointer",
    },
    disabledButton: {
        opacity: 0.65,
        cursor: "default",
    },
};
