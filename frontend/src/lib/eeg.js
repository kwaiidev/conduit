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
