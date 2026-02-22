const BASE = "http://127.0.0.1:8770";

async function setState(state: 0 | 1): Promise<void> {
  await Promise.all([
    fetch(`${BASE}/changestate?state=${state}`),
  ]);
}

export async function enableEEG(): Promise<void> {
  await setState(1);
}

export async function disableEEG(): Promise<void> {
  await setState(0);
}
