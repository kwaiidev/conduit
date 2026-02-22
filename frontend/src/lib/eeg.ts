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

type EegHealthPayload = {
  stream_connected?: boolean;
  state?: number;
};

type EegPredictionPayload = {
  prediction?: number;
  confidence?: number;
  age_ms?: number;
  ageMs?: number;
  state?: number;
  stream_connected?: boolean;
};

export type EegPredictionState = {
  prediction: number;
  confidence: number;
  ageMs: number;
  active: boolean;
};

export async function getEEGReady(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(1500) });
    const data = (await res.json()) as EegHealthPayload;
    return data.state === 1 && Boolean(data.stream_connected);
  } catch {
    return false;
  }
}

export async function getEEGPrediction(): Promise<EegPredictionState | null> {
  try {
    const res = await fetch(`${BASE}/prediction`, { signal: AbortSignal.timeout(800) });
    if (!res.ok) {
      return null;
    }

    const data = (await res.json()) as EegPredictionPayload;
    const prediction = data.prediction;
    const confidence = data.confidence;
    const ageMs = data.age_ms ?? data.ageMs;
    const active = data.state === 1 && Boolean(data.stream_connected);

    if (
      typeof prediction !== "number" ||
      typeof confidence !== "number" ||
      typeof ageMs !== "number"
    ) {
      return null;
    }

    return {
      prediction,
      confidence,
      ageMs,
      active,
    };
  } catch {
    return null;
  }
}
