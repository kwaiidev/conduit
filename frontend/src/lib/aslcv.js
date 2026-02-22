const BASE = "http://localhost:8765";
async function setState(state) {
    await Promise.all([
        fetch(`${BASE}/changestate?state=${state}`, { method: "POST" }),
        fetch(`${BASE}/sessionstate?state=${state}`, { method: "POST" }),
    ]);
}
export async function enableASL() {
    await setState(1);
}
export async function disableASL() {
    await setState(0);
}
export async function getASLReady() {
    try {
        const res = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(1500) });
        const data = await res.json();
        return Boolean(data.detection_ready);
    }
    catch {
        return false;
    }
}
