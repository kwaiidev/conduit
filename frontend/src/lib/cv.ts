const BASE = "http://127.0.0.1:8767";
const REQUEST_TIMEOUT_MS = 2500;

type CvStateUpdate = {
  processing?: boolean;
  streaming?: boolean;
  mouse_control?: boolean;
};

type CvStatusPayload = {
  processing?: boolean;
  mouse_control?: boolean;
};

async function callCvApi(path: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const headers = new Headers(init?.headers ?? undefined);
  if (init?.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  try {
    const response = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers,
    });
    if (!response.ok) {
      throw new Error(`CV API ${path} failed with HTTP ${response.status}`);
    }
    return response;
  } finally {
    window.clearTimeout(timeout);
  }
}

async function setCvState(state: CvStateUpdate): Promise<void> {
  await callCvApi("/state", {
    method: "POST",
    body: JSON.stringify(state),
  });
}

async function ensureCvBackendRunning(): Promise<void> {
  const launchResult = await window.electron.startCvBackend({ camera: 0 });
  if (!launchResult.ok) {
    throw new Error(launchResult.message || "Failed to start eye-gaze backend.");
  }
}

export async function getCvReady(): Promise<boolean> {
  try {
    const response = await callCvApi("/status");
    const payload = (await response.json()) as CvStatusPayload;
    return Boolean(payload.processing) && Boolean(payload.mouse_control);
  } catch {
    return false;
  }
}

export async function enableCvCursorControl(): Promise<void> {
  await ensureCvBackendRunning();
  await setCvState({
    streaming: true,
    processing: true,
    mouse_control: true,
  });
}

export async function disableCvCursorControl(): Promise<void> {
  try {
    await setCvState({
      streaming: false,
      processing: false,
      mouse_control: false,
    });
  } catch (error) {
    // If backend is offline, treat disable as already satisfied.
    console.warn("[CV] disable request skipped:", error);
  }
}
