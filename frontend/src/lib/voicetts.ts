const BASE = "http://localhost:8766";

export async function enableVoice(): Promise<void> {
  await fetch(`${BASE}/ptt/start`, { method: "POST" });
}

export async function disableVoice(): Promise<void> {
  await fetch(`${BASE}/ptt/stop`, { method: "POST" });
}

export async function getVoiceReady(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/status`, { signal: AbortSignal.timeout(1500) });
    const data = await res.json();
    return Boolean(data.is_recording);
  } catch {
    return false;
  }
}
