import React from "react";

type SnapTarget = {
  id: string;
  element: HTMLElement | null;
  source: "dom" | "system";
  left: number;
  top: number;
  width: number;
  height: number;
  cx: number;
  cy: number;
  clusterId: number;
  priority: number;
  label: string;
};

type SystemSnapTargetPayload = {
  id: string;
  kind: string;
  label: string;
  app?: string;
  appPid?: number;
  left: number;
  top: number;
  width: number;
  height: number;
  supportsPress?: boolean;
};

type SystemSnapTargetsResponse = {
  ok: boolean;
  provider: string;
  reason?: string;
  targets: SystemSnapTargetPayload[];
};

type CvStatusResponse = {
  screen_width?: number;
  screen_height?: number;
  last_event?: {
    intent?: string;
    payload?: {
      target_x?: number;
      target_y?: number;
      x_norm?: number;
      y_norm?: number;
    };
  } | null;
};

type FrameState = {
  hasPointer: boolean;
  pointerX: number;
  pointerY: number;
  prevPointerX: number;
  prevPointerY: number;
  displayX: number;
  displayY: number;
  velocityX: number;
  velocityY: number;
  lastTs: number;
  suggestedId: string | null;
  lockedId: string | null;
  dwellMs: number;
  cooldownUntilByTarget: Map<string, number>;
  detachUntilMs: number;
  snapBoostUntilMs: number;
};

type UiState = {
  enabled: boolean;
  x: number;
  y: number;
  suggestedId: string | null;
  lockedId: string | null;
  detachProgress: number;
  modeText: string;
};

const TARGET_SELECTOR = [
  "button",
  "a[href]",
  "input:not([type='hidden']):not([disabled])",
  "textarea:not([disabled])",
  "select:not([disabled])",
  "summary",
  "[role='button']",
  "[role='link']",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

const DISCOVERY_INTERVAL_MS = 220;
const SYSTEM_DISCOVERY_INTERVAL_MS = 1200;
const CV_DISCOVERY_INTERVAL_MS = 90;
const CLUSTER_RADIUS_PX = 96;
const DETECTION_RADIUS_PX = 260;
const HOLD_RADIUS_PX = 78;
const RELEASE_RADIUS_PX = 132;
const SWITCH_HYSTERESIS = 0.14;
const LOCK_SWITCH_HYSTERESIS = 0.22;
const MIN_HOTKEY_INTERVAL_MS = 140;
const AUTO_LOCK_DWELL_MS = 260;
const SLOW_SPEED_PX_PER_MS = 0.24;
const TARGET_COOLDOWN_MS = 640;
const DETACH_ANIM_MS = 220;

const CV_POINTER_STICK_MS = 180;
const DEFAULT_CV_STATUS_URL = "http://127.0.0.1:8767/status";
const CV_BACKEND_CHECK_INTERVAL_MS = 900;

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(max, value));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function distance(aX: number, aY: number, bX: number, bY: number): number {
  return Math.hypot(bX - aX, bY - aY);
}

function normalize(x: number, y: number): { x: number; y: number } {
  const len = Math.hypot(x, y);
  if (len < 1e-6) {
    return { x: 0, y: 0 };
  }
  return { x: x / len, y: y / len };
}

function getWindowScreenOffset(): { x: number; y: number } {
  const maybeWindow = window as Window & {
    screenLeft?: number;
    screenTop?: number;
  };
  const rawX = Number(
    typeof maybeWindow.screenX === "number"
      ? maybeWindow.screenX
      : maybeWindow.screenLeft
  );
  const rawY = Number(
    typeof maybeWindow.screenY === "number"
      ? maybeWindow.screenY
      : maybeWindow.screenTop
  );
  return {
    x: Number.isFinite(rawX) ? rawX : 0,
    y: Number.isFinite(rawY) ? rawY : 0,
  };
}

function mapScreenPointToLocal(screenX: number, screenY: number): { x: number; y: number } {
  const offset = getWindowScreenOffset();
  return {
    x: screenX - offset.x,
    y: screenY - offset.y,
  };
}

function clampPointToViewport(x: number, y: number): { x: number; y: number; clamped: boolean } {
  const maxX = Math.max(0, window.innerWidth);
  const maxY = Math.max(0, window.innerHeight);
  const clampedX = clamp(x, 0, maxX);
  const clampedY = clamp(y, 0, maxY);
  return {
    x: clampedX,
    y: clampedY,
    clamped: Math.abs(clampedX - x) > 0.5 || Math.abs(clampedY - y) > 0.5,
  };
}

function scorePriority(el: HTMLElement): number {
  const tag = el.tagName.toLowerCase();
  if (tag === "button") {
    return 0.26;
  }
  if (tag === "a") {
    return 0.22;
  }
  if (tag === "input" || tag === "textarea" || tag === "select") {
    return 0.2;
  }
  if (el.getAttribute("role") === "button" || el.getAttribute("role") === "link") {
    return 0.18;
  }
  return 0.1;
}

function scoreSystemPriority(kind: string): number {
  const normalized = kind.toLowerCase();
  if (normalized === "button") {
    return 0.18;
  }
  if (normalized === "link") {
    return 0.15;
  }
  if (normalized === "text_field" || normalized === "input") {
    return 0.14;
  }
  return 0.1;
}

function extractLabel(el: HTMLElement): string {
  const aria = el.getAttribute("aria-label");
  if (aria && aria.trim()) {
    return aria.trim();
  }
  const title = el.getAttribute("title");
  if (title && title.trim()) {
    return title.trim();
  }
  const text = (el.textContent || "").replace(/\s+/g, " ").trim();
  if (text) {
    return text.slice(0, 60);
  }
  const placeholder = (el as HTMLInputElement).placeholder;
  if (placeholder && placeholder.trim()) {
    return placeholder.trim();
  }
  return el.tagName.toLowerCase();
}

function isEditableElement(node: EventTarget | null): boolean {
  const el = node as HTMLElement | null;
  if (!el) {
    return false;
  }
  if (el.isContentEditable) {
    return true;
  }
  const tag = el.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") {
    return true;
  }
  return false;
}

function isVisibleTarget(el: HTMLElement): boolean {
  if (el.closest("[data-snap-ignore='true']")) {
    return false;
  }
  const modalOpen = document.documentElement.getAttribute("data-snap-modal-open") === "true";
  if (modalOpen && !el.closest("[data-snap-modal-root='true']")) {
    return false;
  }
  const rect = el.getBoundingClientRect();
  if (rect.width < 8 || rect.height < 8) {
    return false;
  }
  if (rect.bottom < 0 || rect.right < 0 || rect.top > window.innerHeight || rect.left > window.innerWidth) {
    return false;
  }
  const style = window.getComputedStyle(el);
  if (style.display === "none" || style.visibility === "hidden") {
    return false;
  }
  if (style.pointerEvents === "none") {
    return false;
  }
  if (Number(style.opacity || 1) < 0.06) {
    return false;
  }
  return true;
}

function buildClusters(targets: Omit<SnapTarget, "clusterId">[]): SnapTarget[] {
  const clusters: Array<{ cx: number; cy: number; count: number }> = [];
  return targets.map(target => {
    let bestCluster = -1;
    let bestDist = Number.POSITIVE_INFINITY;
    for (let i = 0; i < clusters.length; i += 1) {
      const c = clusters[i];
      const d = distance(target.cx, target.cy, c.cx, c.cy);
      if (d < CLUSTER_RADIUS_PX && d < bestDist) {
        bestDist = d;
        bestCluster = i;
      }
    }
    if (bestCluster === -1) {
      clusters.push({ cx: target.cx, cy: target.cy, count: 1 });
      bestCluster = clusters.length - 1;
    } else {
      const c = clusters[bestCluster];
      c.count += 1;
      c.cx += (target.cx - c.cx) / c.count;
      c.cy += (target.cy - c.cy) / c.count;
    }
    return {
      ...target,
      clusterId: bestCluster,
    };
  });
}

function targetScore(
  target: SnapTarget,
  pointerX: number,
  pointerY: number,
  velocityX: number,
  velocityY: number,
  suggestedId: string | null,
  lockedId: string | null,
  lockedClusterId: number | null,
  cooldownUntilByTarget: Map<string, number>,
  now: number
): number {
  const d = distance(pointerX, pointerY, target.cx, target.cy);
  if (d > DETECTION_RADIUS_PX * 1.25) {
    return Number.POSITIVE_INFINITY;
  }

  let score = d / DETECTION_RADIUS_PX;
  score -= target.priority;

  const speed = Math.hypot(velocityX, velocityY);
  if (speed > 0.04) {
    const velocityNorm = normalize(velocityX, velocityY);
    const toTarget = normalize(target.cx - pointerX, target.cy - pointerY);
    const alignment = velocityNorm.x * toTarget.x + velocityNorm.y * toTarget.y;
    score -= alignment * 0.24;
    if (alignment < -0.25) {
      score += 0.08;
    }
  }

  if (target.id === suggestedId) {
    score -= 0.09;
  }
  if (target.id === lockedId) {
    score -= 0.22;
  }
  if (lockedClusterId !== null && target.clusterId === lockedClusterId) {
    score -= 0.07;
  }

  const cooldownUntil = cooldownUntilByTarget.get(target.id) || 0;
  if (cooldownUntil > now) {
    score += 0.42;
  }

  return score;
}

function cycleInCluster(targets: SnapTarget[], lockedId: string | null): SnapTarget | null {
  if (!lockedId) {
    return null;
  }
  const current = targets.find(t => t.id === lockedId);
  if (!current) {
    return null;
  }
  const peers = targets
    .filter(t => t.clusterId === current.clusterId)
    .sort((a, b) => (a.cx - b.cx) || (a.cy - b.cy));
  if (peers.length <= 1) {
    return current;
  }
  const idx = peers.findIndex(t => t.id === lockedId);
  const nextIdx = idx < 0 ? 0 : (idx + 1) % peers.length;
  return peers[nextIdx];
}

export default function SnapCursorLayer() {
  const [targets, setTargets] = React.useState<SnapTarget[]>([]);
  const [providerLabel, setProviderLabel] = React.useState("DOM");
  const [ui, setUi] = React.useState<UiState>(() => ({
    enabled: true,
    x: window.innerWidth / 2,
    y: window.innerHeight / 2,
    suggestedId: null,
    lockedId: null,
    detachProgress: 0,
    modeText: "SNAP ASSIST ON",
  }));

  const idMapRef = React.useRef(new WeakMap<HTMLElement, string>());
  const nextIdRef = React.useRef(1);
  const domTargetsRef = React.useRef<Omit<SnapTarget, "clusterId">[]>([]);
  const systemTargetsRef = React.useRef<Omit<SnapTarget, "clusterId">[]>([]);
  const targetsRef = React.useRef<SnapTarget[]>([]);
  const frameRef = React.useRef<FrameState>({
    hasPointer: false,
    pointerX: window.innerWidth / 2,
    pointerY: window.innerHeight / 2,
    prevPointerX: window.innerWidth / 2,
    prevPointerY: window.innerHeight / 2,
    displayX: window.innerWidth / 2,
    displayY: window.innerHeight / 2,
    velocityX: 0,
    velocityY: 0,
    lastTs: performance.now(),
    suggestedId: null,
    lockedId: null,
    dwellMs: 0,
    cooldownUntilByTarget: new Map<string, number>(),
    detachUntilMs: 0,
    snapBoostUntilMs: 0,
  });

  const enabledRef = React.useRef(true);
  const commitRequestedRef = React.useRef(false);
  const cycleRequestedRef = React.useRef(false);
  const releaseRequestedRef = React.useRef(false);
  const lastHotkeyAtRef = React.useRef(0);
  const cvPointerActiveUntilRef = React.useRef(0);
  const cvBackendReadyRef = React.useRef(false);
  const cvBackendLastCheckRef = React.useRef(-CV_BACKEND_CHECK_INTERVAL_MS);
  const cvStatusUrlRef = React.useRef(DEFAULT_CV_STATUS_URL);

  React.useEffect(() => {
    if (typeof window.electron?.getCvStatusUrl !== "function") {
      return;
    }
    let active = true;
    void window.electron.getCvStatusUrl()
      .then(url => {
        if (!active) {
          return;
        }
        if (typeof url === "string" && url.trim()) {
          cvStatusUrlRef.current = url.trim();
        }
      })
      .catch(() => {
        // no-op
      });
    return () => {
      active = false;
    };
  }, []);

  const publishTargets = React.useCallback(() => {
    const domTargets = domTargetsRef.current;
    const systemTargets = systemTargetsRef.current;
    const clustered = buildClusters([...domTargets, ...systemTargets]);
    targetsRef.current = clustered;
    setTargets(clustered);
    if (domTargets.length > 0 && systemTargets.length > 0) {
      setProviderLabel("DOM + SYSTEM");
    } else if (systemTargets.length > 0) {
      setProviderLabel("SYSTEM");
    } else {
      setProviderLabel("DOM");
    }
  }, []);

  const ensureIdForElement = React.useCallback((el: HTMLElement): string => {
    const cached = idMapRef.current.get(el);
    if (cached) {
      return cached;
    }
    const id = `snap-target-${nextIdRef.current}`;
    nextIdRef.current += 1;
    idMapRef.current.set(el, id);
    return id;
  }, []);

  const refreshDomTargets = React.useCallback(() => {
    const found: Omit<SnapTarget, "clusterId">[] = [];
    const nodes = Array.from(document.querySelectorAll<HTMLElement>(TARGET_SELECTOR));
    for (const el of nodes) {
      if (!isVisibleTarget(el)) {
        continue;
      }
      const rect = el.getBoundingClientRect();
      const id = ensureIdForElement(el);
      found.push({
        id,
        element: el,
        source: "dom",
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        cx: rect.left + rect.width * 0.5,
        cy: rect.top + rect.height * 0.5,
        priority: scorePriority(el),
        label: extractLabel(el),
      });
    }
    domTargetsRef.current = found;
    publishTargets();
  }, [ensureIdForElement, publishTargets]);

  React.useEffect(() => {
    refreshDomTargets();
    const interval = window.setInterval(() => {
      refreshDomTargets();
    }, DISCOVERY_INTERVAL_MS);

    const observer = new MutationObserver(() => {
      refreshDomTargets();
    });
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style", "disabled", "aria-hidden", "tabindex"],
    });

    const onResize = () => refreshDomTargets();
    window.addEventListener("resize", onResize);
    window.addEventListener("scroll", onResize, true);

    return () => {
      window.clearInterval(interval);
      observer.disconnect();
      window.removeEventListener("resize", onResize);
      window.removeEventListener("scroll", onResize, true);
    };
  }, [refreshDomTargets]);

  React.useEffect(() => {
    if (typeof window.electron?.listSystemSnapTargets !== "function") {
      setProviderLabel("DOM");
      return undefined;
    }

    let active = true;
    const refreshSystemTargets = async () => {
      try {
        const response = (await window.electron.listSystemSnapTargets()) as SystemSnapTargetsResponse;
        if (!active || !response || !Array.isArray(response.targets)) {
          return;
        }
        const offset = getWindowScreenOffset();
        const mapped: Omit<SnapTarget, "clusterId">[] = [];
        for (const target of response.targets) {
          const width = Number(target.width);
          const height = Number(target.height);
          const screenLeft = Number(target.left);
          const screenTop = Number(target.top);
          if (!Number.isFinite(width) || !Number.isFinite(height) || width < 8 || height < 8) {
            continue;
          }
          if (!Number.isFinite(screenLeft) || !Number.isFinite(screenTop)) {
            continue;
          }

          const left = screenLeft - offset.x;
          const top = screenTop - offset.y;

          const cleanKind = typeof target.kind === "string" && target.kind.trim()
            ? target.kind.trim()
            : "control";
          const cleanLabelRaw = typeof target.label === "string" && target.label.trim()
            ? target.label.trim()
            : cleanKind;
          const appName = typeof target.app === "string" && target.app.trim() ? target.app.trim() : "";
          const cleanLabel = appName ? `${cleanLabelRaw} · ${appName}` : cleanLabelRaw;
          mapped.push({
            id: `system-${target.id}`,
            element: null,
            source: "system",
            left,
            top,
            width,
            height,
            cx: left + width * 0.5,
            cy: top + height * 0.5,
            priority: scoreSystemPriority(cleanKind),
            label: cleanLabel,
          });
        }
        systemTargetsRef.current = mapped;
        publishTargets();
      } catch {
        if (!active) {
          return;
        }
        systemTargetsRef.current = [];
        publishTargets();
      }
    };

    void refreshSystemTargets();
    const interval = window.setInterval(() => {
      void refreshSystemTargets();
    }, SYSTEM_DISCOVERY_INTERVAL_MS);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [publishTargets]);

  React.useEffect(() => {
    const onMouseMove = (event: MouseEvent) => {
      if (performance.now() < cvPointerActiveUntilRef.current) {
        return;
      }
      const frame = frameRef.current;
      frame.hasPointer = true;
      frame.pointerX = event.clientX;
      frame.pointerY = event.clientY;
    };
    window.addEventListener("mousemove", onMouseMove, { passive: true });
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
    };
  }, []);

  React.useEffect(() => {
    if (typeof window.electron?.getCursorScreenPoint !== "function") {
      return undefined;
    }
    let active = true;
    let pending = false;
    const poll = async () => {
      if (!active || pending) {
        return;
      }
      pending = true;
      try {
        const point = await window.electron.getCursorScreenPoint();
        if (!active || !point) {
          return;
        }
        const localPoint = mapScreenPointToLocal(Number(point.x), Number(point.y));
        const localX = localPoint.x;
        const localY = localPoint.y;
        const frame = frameRef.current;
        if (performance.now() < cvPointerActiveUntilRef.current) {
          return;
        }
        if (Number.isFinite(localX) && Number.isFinite(localY)) {
          frame.hasPointer = true;
          frame.pointerX = localX;
          frame.pointerY = localY;
        }
      } catch {
        // no-op
      } finally {
        pending = false;
      }
    };
    void poll();
    const interval = window.setInterval(() => {
      void poll();
    }, 33);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  React.useEffect(() => {
    let active = true;
    let pending = false;
    const pollCvPointer = async () => {
      if (!active || pending) {
        return;
      }
      if (!enabledRef.current) {
        return;
      }
      pending = true;
      try {
        if (typeof window.electron?.isCvBackendRunning === "function") {
          const now = performance.now();
          const needsHealthCheck = now - cvBackendLastCheckRef.current >= CV_BACKEND_CHECK_INTERVAL_MS;
          if (needsHealthCheck) {
            cvBackendLastCheckRef.current = now;
            try {
              cvBackendReadyRef.current = await window.electron.isCvBackendRunning();
            } catch {
              cvBackendReadyRef.current = false;
            }
          }
          if (!cvBackendReadyRef.current) {
            return;
          }
        }

        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), 700);
        try {
          const statusUrl = cvStatusUrlRef.current || DEFAULT_CV_STATUS_URL;
          const response = await fetch(statusUrl, { signal: controller.signal });
          if (!response.ok) {
            cvBackendReadyRef.current = false;
            return;
          }
          const payload = (await response.json()) as CvStatusResponse;
          if (!active || !payload?.last_event) {
            return;
          }
          const event = payload.last_event;
          if (event.intent !== "gaze_target" || !event.payload) {
            return;
          }

          let mappedPoint: { x: number; y: number } | null = null;
          const targetX = Number(event.payload.target_x);
          const targetY = Number(event.payload.target_y);
          if (Number.isFinite(targetX) && Number.isFinite(targetY)) {
            mappedPoint = mapScreenPointToLocal(targetX, targetY);
          } else {
            const xNorm = Number(event.payload.x_norm);
            const yNorm = Number(event.payload.y_norm);
            const hasNorm = Number.isFinite(xNorm) && Number.isFinite(yNorm);
            const screenWidth = Number(payload.screen_width);
            const screenHeight = Number(payload.screen_height);
            if (hasNorm && Number.isFinite(screenWidth) && Number.isFinite(screenHeight) && screenWidth > 1 && screenHeight > 1) {
              const screenX = clamp01(xNorm) * (screenWidth - 1);
              const screenY = clamp01(yNorm) * (screenHeight - 1);
              mappedPoint = mapScreenPointToLocal(screenX, screenY);
            } else if (hasNorm) {
              mappedPoint = {
                x: clamp01(xNorm) * window.innerWidth,
                y: clamp01(yNorm) * window.innerHeight,
              };
            }
          }

          if (!mappedPoint || !Number.isFinite(mappedPoint.x) || !Number.isFinite(mappedPoint.y)) {
            return;
          }
          const frame = frameRef.current;
          frame.hasPointer = true;
          frame.pointerX = mappedPoint.x;
          frame.pointerY = mappedPoint.y;
          cvPointerActiveUntilRef.current = performance.now() + CV_POINTER_STICK_MS;
        } finally {
          window.clearTimeout(timeout);
        }
      } catch {
        cvBackendReadyRef.current = false;
      } finally {
        pending = false;
      }
    };

    void pollCvPointer();
    const interval = window.setInterval(() => {
      void pollCvPointer();
    }, CV_DISCOVERY_INTERVAL_MS);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  React.useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const now = performance.now();
      if (event.ctrlKey && event.shiftKey && event.code === "KeyS") {
        event.preventDefault();
        enabledRef.current = !enabledRef.current;
        if (!enabledRef.current) {
          frameRef.current.lockedId = null;
          frameRef.current.suggestedId = null;
          frameRef.current.dwellMs = 0;
        }
        setUi(prev => ({
          ...prev,
          enabled: enabledRef.current,
          modeText: enabledRef.current ? "SNAP ASSIST ON" : "SNAP ASSIST OFF",
        }));
        return;
      }

      if (!enabledRef.current) {
        return;
      }
      if (isEditableElement(event.target)) {
        return;
      }
      if (now - lastHotkeyAtRef.current < MIN_HOTKEY_INTERVAL_MS) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "g") {
        event.preventDefault();
        commitRequestedRef.current = true;
        lastHotkeyAtRef.current = now;
      } else if (event.key === "]") {
        event.preventDefault();
        cycleRequestedRef.current = true;
        lastHotkeyAtRef.current = now;
      } else if (event.key === "Escape") {
        event.preventDefault();
        releaseRequestedRef.current = true;
        lastHotkeyAtRef.current = now;
      } else if (event.key === "Enter") {
        const lockedId = frameRef.current.lockedId;
        if (!lockedId) {
          return;
        }
        const target = targetsRef.current.find(t => t.id === lockedId);
        if (!target) {
          return;
        }
        if (target.source === "system") {
          event.preventDefault();
          const offset = getWindowScreenOffset();
          if (typeof window.electron?.commitSystemSnapTarget === "function") {
            const x = target.cx + offset.x;
            const y = target.cy + offset.y;
            void window.electron.commitSystemSnapTarget({ x, y });
          } else if (typeof window.electron?.commitNearestSnapTarget === "function") {
            void window.electron.commitNearestSnapTarget();
          } else if (typeof window.electron?.moveCursorToScreenPoint === "function") {
            const x = target.cx + offset.x;
            const y = target.cy + offset.y;
            void window.electron.moveCursorToScreenPoint({ x, y });
          }
          lastHotkeyAtRef.current = now;
          return;
        }
        if (!target.element) {
          return;
        }
        event.preventDefault();
        target.element.click();
        lastHotkeyAtRef.current = now;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, []);

  React.useEffect(() => {
    if (ui.enabled) {
      document.documentElement.classList.add("snap-cursor-active");
    } else {
      document.documentElement.classList.remove("snap-cursor-active");
    }
    return () => {
      document.documentElement.classList.remove("snap-cursor-active");
    };
  }, [ui.enabled]);

  React.useEffect(() => {
    let rafId = 0;

    const releaseLock = (now: number) => {
      const frame = frameRef.current;
      if (!frame.lockedId) {
        return;
      }
      frame.cooldownUntilByTarget.set(frame.lockedId, now + TARGET_COOLDOWN_MS);
      frame.lockedId = null;
      frame.dwellMs = 0;
      frame.detachUntilMs = now + DETACH_ANIM_MS;
    };

    const commitLock = (target: SnapTarget, now: number) => {
      const frame = frameRef.current;
      frame.lockedId = target.id;
      frame.suggestedId = target.id;
      frame.dwellMs = 0;
      frame.snapBoostUntilMs = now + 170;
      if (target.source === "system" && typeof window.electron?.moveCursorToScreenPoint === "function") {
        const offset = getWindowScreenOffset();
        const x = target.cx + offset.x;
        const y = target.cy + offset.y;
        frame.pointerX = target.cx;
        frame.pointerY = target.cy;
        frame.prevPointerX = target.cx;
        frame.prevPointerY = target.cy;
        void window.electron.moveCursorToScreenPoint({ x, y });
      }
      if (target.element) {
        target.element.focus({ preventScroll: true });
        try {
          target.element.scrollIntoView({ block: "nearest", inline: "nearest" });
        } catch {
          // no-op
        }
      }
    };

    const loop = (ts: number) => {
      const frame = frameRef.current;
      const now = ts;
      const dt = Math.max(1, Math.min(50, now - frame.lastTs));
      frame.lastTs = now;

      const enabled = enabledRef.current;
      if (!enabled) {
        setUi(prev => ({
          ...prev,
          enabled: false,
          x: frame.displayX,
          y: frame.displayY,
          suggestedId: null,
          lockedId: null,
          detachProgress: 0,
        }));
        rafId = window.requestAnimationFrame(loop);
        return;
      }

      if (!frame.hasPointer) {
        frame.lockedId = null;
        frame.suggestedId = null;
        frame.dwellMs = 0;
        frame.displayX += (frame.pointerX - frame.displayX) * 0.2;
        frame.displayY += (frame.pointerY - frame.displayY) * 0.2;
        setUi({
          enabled: true,
          x: frame.displayX,
          y: frame.displayY,
          suggestedId: null,
          lockedId: null,
          detachProgress: 0,
          modeText: "SCAN",
        });
        rafId = window.requestAnimationFrame(loop);
        return;
      }

      frame.velocityX = (frame.pointerX - frame.prevPointerX) / dt;
      frame.velocityY = (frame.pointerY - frame.prevPointerY) / dt;
      frame.prevPointerX = frame.pointerX;
      frame.prevPointerY = frame.pointerY;

      for (const [id, until] of frame.cooldownUntilByTarget.entries()) {
        if (until <= now) {
          frame.cooldownUntilByTarget.delete(id);
        }
      }

      const targetsLive = targetsRef.current;
      const lockedTarget = frame.lockedId
        ? targetsLive.find(t => t.id === frame.lockedId) || null
        : null;
      const lockedClusterId = lockedTarget?.clusterId ?? null;

      const ranked = targetsLive
        .map(target => ({
          target,
          score: targetScore(
            target,
            frame.pointerX,
            frame.pointerY,
            frame.velocityX,
            frame.velocityY,
            frame.suggestedId,
            frame.lockedId,
            lockedClusterId,
            frame.cooldownUntilByTarget,
            now
          ),
        }))
        .filter(item => Number.isFinite(item.score))
        .sort((a, b) => a.score - b.score);

      let best = ranked[0] || null;

      if (frame.suggestedId && best) {
        const currentRank = ranked.find(r => r.target.id === frame.suggestedId);
        if (currentRank) {
          const isSameCluster = currentRank.target.clusterId === best.target.clusterId;
          const hysteresis = frame.lockedId ? LOCK_SWITCH_HYSTERESIS : SWITCH_HYSTERESIS;
          if (isSameCluster && best.score > currentRank.score - hysteresis) {
            best = currentRank;
          }
        }
      }

      if (releaseRequestedRef.current) {
        releaseRequestedRef.current = false;
        releaseLock(now);
      }

      if (cycleRequestedRef.current && frame.lockedId) {
        cycleRequestedRef.current = false;
        const nextTarget = cycleInCluster(targetsLive, frame.lockedId);
        if (nextTarget) {
          commitLock(nextTarget, now);
        }
      } else {
        cycleRequestedRef.current = false;
      }

      if (commitRequestedRef.current) {
        commitRequestedRef.current = false;
        if (best) {
          commitLock(best.target, now);
        }
      }

      if (!frame.lockedId && best) {
        const distToBest = distance(frame.pointerX, frame.pointerY, best.target.cx, best.target.cy);
        const speed = Math.hypot(frame.velocityX, frame.velocityY);
        if (distToBest < HOLD_RADIUS_PX && speed < SLOW_SPEED_PX_PER_MS) {
          frame.dwellMs += dt;
          if (frame.dwellMs >= AUTO_LOCK_DWELL_MS) {
            commitLock(best.target, now);
          }
        } else {
          frame.dwellMs = Math.max(0, frame.dwellMs - dt * 1.9);
        }
      } else {
        frame.dwellMs = Math.max(0, frame.dwellMs - dt * 1.5);
      }

      const liveLockedTarget = frame.lockedId
        ? targetsLive.find(t => t.id === frame.lockedId) || null
        : null;
      if (frame.lockedId && !liveLockedTarget) {
        releaseLock(now);
      }

      const speed = Math.hypot(frame.velocityX, frame.velocityY);
      if (liveLockedTarget) {
        const distToLock = distance(frame.pointerX, frame.pointerY, liveLockedTarget.cx, liveLockedTarget.cy);
        const toLock = normalize(liveLockedTarget.cx - frame.pointerX, liveLockedTarget.cy - frame.pointerY);
        const velNorm = normalize(frame.velocityX, frame.velocityY);
        const movingAway = velNorm.x * toLock.x + velNorm.y * toLock.y < -0.35;

        if ((distToLock > RELEASE_RADIUS_PX && speed > SLOW_SPEED_PX_PER_MS * 1.6) || (movingAway && speed > 0.28)) {
          releaseLock(now);
        }
      }

      frame.suggestedId = frame.lockedId || best?.target.id || null;

      const liveSuggested = frame.suggestedId
        ? targetsLive.find(t => t.id === frame.suggestedId) || null
        : null;

      let desiredX = frame.pointerX;
      let desiredY = frame.pointerY;
      if (liveSuggested) {
        const distToSuggested = distance(frame.pointerX, frame.pointerY, liveSuggested.cx, liveSuggested.cy);
        const isLocked = frame.lockedId === liveSuggested.id;
        const magnet = isLocked
          ? 0.88
          : clamp01((DETECTION_RADIUS_PX - distToSuggested) / DETECTION_RADIUS_PX) * 0.34;
        desiredX = frame.pointerX + (liveSuggested.cx - frame.pointerX) * magnet;
        desiredY = frame.pointerY + (liveSuggested.cy - frame.pointerY) * magnet;
      }

      const snapBoost = now < frame.snapBoostUntilMs ? 0.55 : 0.32;
      frame.displayX += (desiredX - frame.displayX) * snapBoost;
      frame.displayY += (desiredY - frame.displayY) * snapBoost;

      const detachProgress = clamp01((frame.detachUntilMs - now) / DETACH_ANIM_MS);
      const modeText = frame.lockedId
        ? "LOCKED"
        : liveSuggested
          ? "SUGGEST"
          : "SCAN";

      setUi({
        enabled: true,
        x: frame.displayX,
        y: frame.displayY,
        suggestedId: frame.suggestedId,
        lockedId: frame.lockedId,
        detachProgress,
        modeText,
      });

      rafId = window.requestAnimationFrame(loop);
    };

    rafId = window.requestAnimationFrame(loop);
    return () => {
      window.cancelAnimationFrame(rafId);
    };
  }, []);

  const suggested = ui.suggestedId ? targets.find(t => t.id === ui.suggestedId) || null : null;
  const locked = ui.lockedId ? targets.find(t => t.id === ui.lockedId) || null : null;
  const suggestedIsSystem = suggested?.source === "system";
  const pointerVisual = clampPointToViewport(ui.x, ui.y);
  const suggestedVisual = (() => {
    if (!suggested) {
      return null;
    }
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const isInViewport = (
      suggested.left + suggested.width >= 0
      && suggested.top + suggested.height >= 0
      && suggested.left <= viewportWidth
      && suggested.top <= viewportHeight
    );
    if (isInViewport) {
      return {
        left: suggested.left,
        top: suggested.top,
        width: suggested.width,
        height: suggested.height,
        isOffscreen: false,
      };
    }
    const anchor = clampPointToViewport(suggested.cx, suggested.cy);
    const proxyWidth = Math.min(220, Math.max(120, Math.min(suggested.width + 28, 180)));
    const proxyHeight = 36;
    return {
      left: clamp(anchor.x - proxyWidth * 0.5, 6, Math.max(6, viewportWidth - proxyWidth - 6)),
      top: clamp(anchor.y - proxyHeight * 0.5, 6, Math.max(6, viewportHeight - proxyHeight - 6)),
      width: proxyWidth,
      height: proxyHeight,
      isOffscreen: true,
    };
  })();

  return (
    <div data-snap-ignore="true" style={styles.root}>
      {suggested && suggestedVisual ? (
        <div
          style={{
            ...styles.targetHint,
            left: suggestedVisual.left,
            top: suggestedVisual.top,
            width: suggestedVisual.width,
            height: suggestedVisual.height,
            borderStyle: suggestedVisual.isOffscreen ? "dashed" : "solid",
            background: suggestedVisual.isOffscreen
              ? "linear-gradient(135deg, rgba(82,128,255,0.18), rgba(8,14,22,0.22))"
              : styles.targetHint.background,
            borderColor: locked
              ? "rgba(0,194,170,0.95)"
              : suggestedIsSystem
                ? "rgba(82,128,255,0.85)"
                : "rgba(255,45,141,0.78)",
            boxShadow: locked
              ? "0 0 0 8px rgba(0,194,170,0.16), 0 8px 22px rgba(0,194,170,0.22)"
              : suggestedIsSystem
                ? "0 0 0 6px rgba(82,128,255,0.15), 0 6px 18px rgba(82,128,255,0.2)"
                : "0 0 0 6px rgba(255,45,141,0.13), 0 6px 18px rgba(255,45,141,0.18)",
          }}
        >
          <div style={suggestedVisual.isOffscreen ? styles.targetLabelInline : styles.targetLabel}>
            {suggested.label}
            {suggestedVisual.isOffscreen ? " (offscreen)" : ""}
            {suggestedIsSystem ? " (system)" : ""}
          </div>
        </div>
      ) : null}

      <div
        style={{
          ...styles.cursorCore,
          left: pointerVisual.x,
          top: pointerVisual.y,
          transform: `translate(-50%, -50%) scale(${locked ? 1.22 : suggested ? 1.08 : 1})`,
          borderStyle: pointerVisual.clamped ? "dashed" : "solid",
          opacity: pointerVisual.clamped ? 0.86 : 1,
          borderColor: locked
            ? "rgba(0,194,170,0.96)"
            : suggestedIsSystem
              ? "rgba(82,128,255,0.9)"
              : "rgba(255,45,141,0.85)",
          boxShadow: locked
            ? "0 0 0 12px rgba(0,194,170,0.22), 0 0 0 2px rgba(255,255,255,0.45) inset"
            : suggestedIsSystem
              ? "0 0 0 9px rgba(82,128,255,0.16), 0 0 0 2px rgba(255,255,255,0.35) inset"
              : "0 0 0 9px rgba(255,45,141,0.14), 0 0 0 2px rgba(255,255,255,0.35) inset",
        }}
      >
        <span
          style={{
            ...styles.cursorDot,
            background: locked ? "#00c2aa" : suggestedIsSystem ? "#5280ff" : "#FF2D8D",
          }}
        />
      </div>

      {ui.detachProgress > 0 ? (
        <div
          style={{
            ...styles.detachPulse,
            left: pointerVisual.x,
            top: pointerVisual.y,
            opacity: ui.detachProgress,
            transform: `translate(-50%, -50%) scale(${1 + (1 - ui.detachProgress) * 0.55})`,
          }}
        />
      ) : null}

      <div style={styles.hudPill}>
        <span style={styles.hudMode}>{ui.modeText}</span>
        <span style={styles.hudHint}>{providerLabel} | G lock | ] cycle | Esc release | Ctrl+Shift+S toggle | Cmd/Ctrl+Shift+G global</span>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    position: "fixed",
    inset: 0,
    pointerEvents: "none",
    zIndex: 2147483000,
  },
  targetHint: {
    position: "fixed",
    borderRadius: 14,
    border: "2px solid rgba(255,45,141,0.8)",
    background: "linear-gradient(135deg, rgba(255,45,141,0.12), rgba(255,255,255,0.02))",
    transition: "left 90ms linear, top 90ms linear, width 90ms linear, height 90ms linear, border-color 140ms ease, box-shadow 140ms ease",
  },
  targetLabel: {
    position: "absolute",
    top: -24,
    left: 0,
    padding: "3px 8px",
    borderRadius: 999,
    fontSize: 10,
    fontWeight: 700,
    color: "white",
    letterSpacing: "0.03em",
    background: "rgba(10,14,18,0.86)",
    border: "1px solid rgba(255,255,255,0.2)",
    whiteSpace: "nowrap",
    maxWidth: 220,
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  targetLabelInline: {
    position: "absolute",
    top: "50%",
    left: 10,
    transform: "translateY(-50%)",
    padding: "3px 8px",
    borderRadius: 999,
    fontSize: 10,
    fontWeight: 700,
    color: "white",
    letterSpacing: "0.03em",
    background: "rgba(10,14,18,0.8)",
    border: "1px solid rgba(255,255,255,0.2)",
    whiteSpace: "nowrap",
    maxWidth: 200,
    overflow: "hidden",
    textOverflow: "ellipsis",
  },
  cursorCore: {
    position: "fixed",
    width: 28,
    height: 28,
    borderRadius: "50%",
    border: "2px solid rgba(255,45,141,0.85)",
    display: "grid",
    placeItems: "center",
    background: "rgba(14,16,20,0.2)",
    transition: "transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease",
  },
  cursorDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#FF2D8D",
    boxShadow: "0 0 12px rgba(255,45,141,0.6)",
  },
  detachPulse: {
    position: "fixed",
    width: 34,
    height: 34,
    borderRadius: "50%",
    border: "2px solid rgba(255,255,255,0.7)",
  },
  hudPill: {
    position: "fixed",
    right: 14,
    bottom: 14,
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 10px",
    borderRadius: 999,
    border: "1px solid rgba(255,255,255,0.16)",
    background: "rgba(12,16,20,0.78)",
    color: "white",
    backdropFilter: "blur(6px)",
    WebkitBackdropFilter: "blur(6px)",
  },
  hudMode: {
    fontSize: 11,
    fontWeight: 800,
    letterSpacing: "0.08em",
    color: "#00c2aa",
  },
  hudHint: {
    fontSize: 11,
    fontWeight: 600,
    color: "rgba(255,255,255,0.86)",
  },
};
