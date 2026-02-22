const BASE = "http://localhost:8765";

async function setState(state: 0 | 1): Promise<void> {
  await Promise.all([
    fetch(`${BASE}/changestate?state=${state}`, { method: "POST" }),
    fetch(`${BASE}/sessionstate?state=${state}`, { method: "POST" }),
  ]);
}

export async function enableASL(): Promise<void> {
  await setState(1);
}

export async function disableASL(): Promise<void> {
  await setState(0);
}

export async function getASLReady(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(1500) });
    const data = await res.json();
    return Boolean(data.detection_ready);
  } catch {
    return false;
  }
}
