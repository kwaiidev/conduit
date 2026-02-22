const BASE = "http://localhost:8766";
async function ensureVoiceBackendRunning() {
    if (typeof window === "undefined" || !window.electron?.startVoiceBackend) {
        return;
    }
    if (window.electron.isVoiceBackendRunning) {
        try {
            const alreadyRunning = await window.electron.isVoiceBackendRunning();
            if (alreadyRunning) {
                return;
            }
        }
        catch {
            // Fall through to launch attempt.
        }
    }
    const launchResult = await window.electron.startVoiceBackend();
    if (!launchResult.ok) {
        throw new Error(launchResult.message || "Failed to start voice backend.");
    }
}
async function postVoiceJson(path) {
    const response = await fetch(`${BASE}${path}`, { method: "POST" });
    if (!response.ok) {
        throw new Error(`Voice request failed (${response.status}) for ${path}`);
    }
    return (await response.json());
}
function isExpectedPttReason(reason) {
    return reason === "already recording" || reason === "not recording";
}
export async function enableVoice() {
    await ensureVoiceBackendRunning();
    await postVoiceJson("/changestate?state=1");
    const ptt = await postVoiceJson("/ptt/start");
    if (ptt.success === false && !isExpectedPttReason(ptt.reason)) {
        throw new Error(ptt.reason || "Failed to start voice recording.");
    }
}
export async function disableVoice() {
    try {
        const ptt = await postVoiceJson("/ptt/stop");
        if (ptt.success === false && !isExpectedPttReason(ptt.reason)) {
            throw new Error(ptt.reason || "Failed to stop voice recording.");
        }
    }
    finally {
        await postVoiceJson("/changestate?state=0");
    }
}
export async function getVoiceReady() {
    let processRunning = false;
    if (typeof window !== "undefined" && window.electron?.isVoiceBackendRunning) {
        try {
            processRunning = await window.electron.isVoiceBackendRunning();
            if (!processRunning) {
                return false;
            }
        }
        catch {
            processRunning = false;
        }
    }
    try {
        const res = await fetch(`${BASE}/changestate`, { signal: AbortSignal.timeout(1500) });
        const data = (await res.json());
        if (typeof data.state === "number") {
            return data.state === 1;
        }
        if (typeof data.status === "string") {
            return data.status.toLowerCase() === "active";
        }
        return processRunning;
    }
    catch {
        return processRunning;
    }
}
