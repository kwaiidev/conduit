import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import React from "react";
const SERVICE_LABEL = {
    voice: "Voice",
    sign: "Sign",
};
export function ServiceStartupStep({ service }) {
    const [isStarting, setIsStarting] = React.useState(false);
    const [isReady, setIsReady] = React.useState(false);
    const [statusMessage, setStatusMessage] = React.useState(`Press Start to launch the ${SERVICE_LABEL[service]} backend terminal.`);
    const checkRunning = React.useCallback(async () => {
        if (service === "voice") {
            return await window.electron.isVoiceBackendRunning();
        }
        return await window.electron.isSignBackendRunning();
    }, [service]);
    const startBackend = React.useCallback(async () => {
        setIsStarting(true);
        setStatusMessage(`Launching ${SERVICE_LABEL[service]} backend...`);
        try {
            const launchResult = service === "voice"
                ? await window.electron.startVoiceBackend()
                : await window.electron.startSignBackend();
            if (!launchResult.ok) {
                setStatusMessage(`Unable to launch ${SERVICE_LABEL[service]} backend: ${launchResult.message}`);
                setIsReady(false);
                return;
            }
            setIsReady(true);
            setStatusMessage(`${SERVICE_LABEL[service]} backend running.`);
        }
        catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            setIsReady(false);
            setStatusMessage(`${SERVICE_LABEL[service]} startup failed: ${detail}`);
        }
        finally {
            setIsStarting(false);
        }
    }, [service]);
    React.useEffect(() => {
        let cancelled = false;
        const init = async () => {
            try {
                const running = await checkRunning();
                if (cancelled || !running) {
                    return;
                }
                setIsReady(true);
                setStatusMessage(`${SERVICE_LABEL[service]} backend already running.`);
            }
            catch {
                // no-op
            }
        };
        void init();
        return () => {
            cancelled = true;
        };
    }, [checkRunning, service]);
    return (_jsx("div", { style: styles.container, children: _jsxs("div", { style: styles.card, children: [_jsxs("h3", { style: styles.title, children: [SERVICE_LABEL[service], " Service"] }), _jsx("p", { style: styles.description, children: "Launch this backend from onboarding so the modality can be enabled later from Home/Overlay." }), _jsx("p", { style: styles.status, children: statusMessage }), _jsx("button", { type: "button", onClick: startBackend, style: {
                        ...styles.startButton,
                        ...(isStarting ? styles.disabledButton : {}),
                        ...(isReady ? styles.readyButton : {}),
                    }, disabled: isStarting, children: isStarting ? "Starting..." : isReady ? "Running" : "Start" })] }) }));
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
    readyButton: {
        border: "1px solid var(--border)",
        background: "var(--bg-tertiary)",
        color: "var(--text-primary)",
    },
    disabledButton: {
        opacity: 0.65,
        cursor: "default",
    },
};
