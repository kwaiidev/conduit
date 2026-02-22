const BASE = "http://127.0.0.1:8770";
async function setState(state) {
    await Promise.all([
        fetch(`${BASE}/changestate?state=${state}`),
    ]);
}
export async function enableEEG() {
    await setState(1);
}
export async function disableEEG() {
    await setState(0);
}
export async function getEEGReady() {
    try {
        const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(1500) });
        const data = (await res.json());
        return data.state === 1 && Boolean(data.stream_connected);
    }
    catch {
        return false;
    }
}
export async function getEEGPrediction() {
    try {
        const res = await fetch(`${BASE}/prediction`, { signal: AbortSignal.timeout(800) });
        if (!res.ok) {
            return null;
        }
        const data = (await res.json());
        const prediction = data.prediction;
        const confidence = data.confidence;
        const ageMs = data.age_ms ?? data.ageMs;
        const active = data.state === 1 && Boolean(data.stream_connected);
        if (typeof prediction !== "number" ||
            typeof confidence !== "number" ||
            typeof ageMs !== "number") {
            return null;
        }
        return {
            prediction,
            confidence,
            ageMs,
            active,
        };
    }
    catch {
        return null;
    }
}
