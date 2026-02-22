import { app, BrowserWindow, ipcMain, screen, globalShortcut } from "electron";
import path from "path";
import fs from "fs";
import { spawn } from "child_process";
import net from "net";
import http from "http";

let win: BrowserWindow | null = null;
let currentMode: 'full' | 'overlay' = 'full';
let pendingFullRoute: string | null = null;
let isToggling = false;
let cvBackendPid: number | null = null;
let voiceBackendPid: number | null = null;
let signBackendPid: number | null = null;
let eegBackendPid: number | null = null;
const SYSTEM_TARGET_TIMEOUT_MS = 1600;
const SYSTEM_ACTION_TIMEOUT_MS = 1200;
const SNAP_GLOBAL_SHORTCUT = "CommandOrControl+Shift+G";
const SNAP_TARGET_COOLDOWN_MS = 520;
const SYSTEM_TARGET_MAX_COUNT = 900;
const parsePort = (value: string | undefined, fallback: number): number => {
  const parsed = Number.parseInt(value || "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};
const CAMERA_BROKER_STREAM_URL = process.env.CAMERA_BROKER_URL || "http://localhost:9001/stream";
const CV_HTTP_PORT = parsePort(process.env.CV_HTTP_PORT, 8767);
const CV_EVENT_PORT = parsePort(process.env.CV_EVENT_PORT, 8768);
const EEG_HTTP_PORT = parsePort(process.env.EEG_HTTP_PORT, 8770);
const EEG_HEALTH_PATH = "/health";
const EEG_SERVICE_ID = "muse-eeg-realtime";

let lastSnappedTargetId: string | null = null;
let lastSnappedTargetAt = 0;

type SystemSnapTarget = {
  id: string;
  kind: string;
  label: string;
  app: string;
  appPid: number;
  left: number;
  top: number;
  width: number;
  height: number;
  supportsPress: boolean;
};

type SystemSnapTargetsResponse = {
  ok: boolean;
  provider: string;
  reason?: string;
  targets: SystemSnapTarget[];
};

type SnapCommitResponse = {
  ok: boolean;
  message: string;
  provider?: string;
  method?: string;
  reason?: string;
  target?: {
    id: string;
    x: number;
    y: number;
    label?: string;
    app?: string;
  };
};

async function moveCursorToScreenPoint(x: number, y: number): Promise<boolean> {
  if (process.platform !== "darwin") {
    return false;
  }
  const clampedX = Number.isFinite(x) ? Math.max(0, x) : 0;
  const clampedY = Number.isFinite(y) ? Math.max(0, y) : 0;
  const script = `
import ctypes
import sys

class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

target_x = float(sys.argv[1])
target_y = float(sys.argv[2])
api = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
result = api.CGWarpMouseCursorPosition(CGPoint(target_x, target_y))
api.CGAssociateMouseAndMouseCursorPosition(True)
sys.exit(0 if result == 0 else 1)
`;

  return await new Promise(resolve => {
    const child = spawn("python3", ["-c", script, String(clampedX), String(clampedY)], {
      stdio: ["ignore", "ignore", "ignore"],
    });
    child.once("error", () => resolve(false));
    child.once("close", code => resolve(code === 0));
  });
}

async function clickCursorAtScreenPoint(x: number, y: number): Promise<boolean> {
  if (process.platform !== "darwin") {
    return false;
  }
  const clampedX = Number.isFinite(x) ? Math.max(0, x) : 0;
  const clampedY = Number.isFinite(y) ? Math.max(0, y) : 0;
  const script = `
import ctypes
import sys
import time

class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

target_x = float(sys.argv[1])
target_y = float(sys.argv[2])
api = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
api.CGEventCreateMouseEvent.restype = ctypes.c_void_p
api.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]

kCGHIDEventTap = 0
kCGEventMouseMoved = 5
kCGEventLeftMouseDown = 1
kCGEventLeftMouseUp = 2
kCGMouseButtonLeft = 0

pt = CGPoint(target_x, target_y)

def post(event_type):
    evt = api.CGEventCreateMouseEvent(None, event_type, pt, kCGMouseButtonLeft)
    if not evt:
        return False
    api.CGEventPost(kCGHIDEventTap, evt)
    return True

api.CGWarpMouseCursorPosition(pt)
api.CGAssociateMouseAndMouseCursorPosition(True)
ok = post(kCGEventMouseMoved)
time.sleep(0.006)
ok = post(kCGEventLeftMouseDown) and ok
time.sleep(0.006)
ok = post(kCGEventLeftMouseUp) and ok
sys.exit(0 if ok else 1)
`;

  return await new Promise(resolve => {
    const child = spawn("python3", ["-c", script, String(clampedX), String(clampedY)], {
      stdio: ["ignore", "ignore", "ignore"],
    });
    child.once("error", () => resolve(false));
    child.once("close", code => resolve(code === 0));
  });
}

function stableHash32(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
    hash >>>= 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function normalizeAccessibilityKind(kindRaw: string): string {
  const trimmed = kindRaw.trim();
  if (!trimmed) {
    return "control";
  }
  const noPrefix = trimmed.startsWith("AX") ? trimmed.slice(2) : trimmed;
  return noPrefix
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/^_+|_+$/g, "")
    .toLowerCase() || "control";
}

function sanitizeLabel(labelRaw: string, fallback: string): string {
  const collapsed = labelRaw.replace(/\s+/g, " ").trim();
  if (!collapsed) {
    return fallback;
  }
  return collapsed.slice(0, 120);
}

type OsaScriptResult = {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
};

async function runOsaScript(script: string, args: string[], timeoutMs: number): Promise<OsaScriptResult> {
  return await new Promise(resolve => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const child = spawn("osascript", ["-e", script, ...args], { stdio: ["ignore", "pipe", "pipe"] });
    const finish = (payload: OsaScriptResult) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(payload);
    };
    const timeout = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // no-op
      }
      finish({
        code: null,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut: true,
      });
    }, timeoutMs);

    child.stdout.on("data", chunk => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", chunk => {
      stderr += chunk.toString();
    });

    child.on("error", error => {
      clearTimeout(timeout);
      finish({
        code: null,
        stdout: stdout.trim(),
        stderr: `${stderr.trim()} ${error.message}`.trim(),
        timedOut: false,
      });
    });

    child.on("close", code => {
      clearTimeout(timeout);
      finish({
        code,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        timedOut: false,
      });
    });
  });
}

function parseSystemError(raw: string): string {
  const line = raw
    .split(/\r?\n/)
    .map(part => part.trim())
    .find(Boolean);
  if (!line) {
    return "unknown";
  }
  if (!line.startsWith("__ERROR__")) {
    return line;
  }
  const parts = line.split("\t").slice(1).filter(Boolean);
  if (!parts.length) {
    return "unknown";
  }
  return parts.join(":");
}

function parseSystemTargetLines(raw: string): SystemSnapTarget[] {
  const lines = raw
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);
  const targets: SystemSnapTarget[] = [];
  const seenIds = new Set<string>();
  for (const line of lines) {
    const parts = line.split("\t");
    if (parts.length < 9) {
      continue;
    }
    const appPid = Number(parts[0]);
    const app = parts[1]?.trim() || "App";
    const kind = normalizeAccessibilityKind(parts[2] || "control");
    const label = sanitizeLabel(parts[3] || "", kind);
    const left = Number(parts[4]);
    const top = Number(parts[5]);
    const width = Number(parts[6]);
    const height = Number(parts[7]);
    const supportsPress = (parts[8] || "0").trim() === "1";
    if (
      !Number.isFinite(appPid)
      || appPid <= 0
      || !Number.isFinite(left)
      || !Number.isFinite(top)
      || !Number.isFinite(width)
      || !Number.isFinite(height)
      || width < 8
      || height < 8
    ) {
      continue;
    }
    if (appPid === process.pid) {
      continue;
    }
    const idSeed = [
      appPid,
      app,
      kind,
      Math.round(left),
      Math.round(top),
      Math.round(width),
      Math.round(height),
      label.toLowerCase(),
    ].join("|");
    const id = `ax-${stableHash32(idSeed)}`;
    if (seenIds.has(id)) {
      continue;
    }
    seenIds.add(id);
    targets.push({
      id,
      kind,
      label,
      app,
      appPid,
      left,
      top,
      width,
      height,
      supportsPress,
    });
    if (targets.length >= SYSTEM_TARGET_MAX_COUNT) {
      break;
    }
  }
  return targets;
}

async function listSystemSnapTargets(): Promise<SystemSnapTargetsResponse> {
  if (process.platform !== "darwin") {
    return {
      ok: false,
      provider: "macos-accessibility",
      reason: "unsupported-platform",
      targets: [],
    };
  }

  const script = `
on clean_text(value_in)
  if value_in is missing value then return ""
  set asText to (value_in as text)
  set AppleScript's text item delimiters to {tab, return, linefeed}
  set pieces to text items of asText
  set AppleScript's text item delimiters to " "
  set joinedText to pieces as text
  set AppleScript's text item delimiters to ""
  return joinedText
end clean_text

on run argv
  set selfPid to -1
  if (count of argv) > 0 then
    try
      set selfPid to (item 1 of argv as integer)
    end try
  end if

  set outLines to {}
  set maxPerWindow to 220
  set maxMenuItems to 140
  set maxTotal to 900
  set allowedRoles to {"AXButton", "AXCheckBox", "AXRadioButton", "AXLink", "AXPopUpButton", "AXMenuButton", "AXDisclosureTriangle", "AXTextField", "AXTextArea", "AXComboBox", "AXMenuItem", "AXScrollBar", "AXTabButton"}

  tell application "System Events"
    if UI elements enabled is false then
      return "__ERROR__" & tab & "AX_DISABLED"
    end if

    set procList to (application processes whose background only is false)
    repeat with proc in procList
      set procName to ""
      set procPid to -1
      set frontProc to false
      try
        set procName to name of proc
      end try
      try
        set procPid to unix id of proc
      end try
      try
        set frontProc to frontmost of proc
      end try
      if procPid is selfPid then
      else
        repeat with win in windows of proc
          set addedForWindow to 0
          try
            repeat with el in (entire contents of win)
              if (count of outLines) >= maxTotal then exit repeat
              if addedForWindow >= maxPerWindow then exit repeat
              set roleName to ""
              try
                set roleName to role of el
              end try
              if allowedRoles contains roleName then
                try
                  set p to position of el
                  set s to size of el
                  set leftPos to item 1 of p
                  set topPos to item 2 of p
                  set w to item 1 of s
                  set h to item 2 of s
                  if (w > 7) and (h > 7) then
                    set labelText to ""
                    try
                      set labelText to value of attribute "AXDescription" of el
                    end try
                    if my clean_text(labelText) is "" then
                      try
                        set labelText to value of attribute "AXTitle" of el
                      end try
                    end if
                    if my clean_text(labelText) is "" then
                      try
                        set labelText to name of el
                      end try
                    end if
                    if my clean_text(labelText) is "" then set labelText to roleName

                    set pressCapable to "0"
                    try
                      set actionNames to actions of el
                      if actionNames contains "AXPress" then set pressCapable to "1"
                    end try

                    set lineText to (procPid as text) & tab & my clean_text(procName) & tab & my clean_text(roleName) & tab & my clean_text(labelText) & tab & (leftPos as text) & tab & (topPos as text) & tab & (w as text) & tab & (h as text) & tab & pressCapable
                    set end of outLines to lineText
                    set addedForWindow to addedForWindow + 1
                  end if
                end try
              end if
            end repeat
          end try
          if (count of outLines) >= maxTotal then exit repeat
        end repeat

        if frontProc then
          set addedForMenu to 0
          try
            repeat with el in (entire contents of menu bar 1 of proc)
              if (count of outLines) >= maxTotal then exit repeat
              if addedForMenu >= maxMenuItems then exit repeat
              set roleName to ""
              try
                set roleName to role of el
              end try
              if allowedRoles contains roleName then
                try
                  set p to position of el
                  set s to size of el
                  set leftPos to item 1 of p
                  set topPos to item 2 of p
                  set w to item 1 of s
                  set h to item 2 of s
                  if (w > 7) and (h > 7) then
                    set labelText to ""
                    try
                      set labelText to value of attribute "AXDescription" of el
                    end try
                    if my clean_text(labelText) is "" then
                      try
                        set labelText to value of attribute "AXTitle" of el
                      end try
                    end if
                    if my clean_text(labelText) is "" then
                      try
                        set labelText to name of el
                      end try
                    end if
                    if my clean_text(labelText) is "" then set labelText to roleName

                    set pressCapable to "0"
                    try
                      set actionNames to actions of el
                      if actionNames contains "AXPress" then set pressCapable to "1"
                    end try

                    set lineText to (procPid as text) & tab & my clean_text(procName) & tab & my clean_text(roleName) & tab & my clean_text(labelText) & tab & (leftPos as text) & tab & (topPos as text) & tab & (w as text) & tab & (h as text) & tab & pressCapable
                    set end of outLines to lineText
                    set addedForMenu to addedForMenu + 1
                  end if
                end try
              end if
            end repeat
          end try
        end if
      end if
      if (count of outLines) >= maxTotal then exit repeat
    end repeat
  end tell

  set AppleScript's text item delimiters to linefeed
  return outLines as text
end run
`;

  const result = await runOsaScript(script, [String(process.pid)], SYSTEM_TARGET_TIMEOUT_MS);
  if (result.timedOut) {
    return {
      ok: false,
      provider: "macos-accessibility",
      reason: "timeout",
      targets: [],
    };
  }
  if (result.code !== 0) {
    const detail = result.stderr ? `:${result.stderr}` : "";
    return {
      ok: false,
      provider: "macos-accessibility",
      reason: `osascript-exit:${result.code}${detail}`,
      targets: [],
    };
  }

  const text = result.stdout.trim();
  if (!text) {
    return {
      ok: true,
      provider: "macos-accessibility",
      reason: "empty",
      targets: [],
    };
  }
  if (text.startsWith("__ERROR__")) {
    return {
      ok: false,
      provider: "macos-accessibility",
      reason: parseSystemError(text),
      targets: [],
    };
  }
  return {
    ok: true,
    provider: "macos-accessibility",
    targets: parseSystemTargetLines(text),
  };
}

type SystemTargetActionResult = {
  ok: boolean;
  method: string;
  reason?: string;
  app?: string;
  kind?: string;
  label?: string;
  left?: number;
  top?: number;
  width?: number;
  height?: number;
};

function parseSystemTargetActionResult(raw: string): SystemTargetActionResult {
  const line = raw
    .split(/\r?\n/)
    .map(part => part.trim())
    .find(Boolean);
  if (!line) {
    return {
      ok: false,
      method: "none",
      reason: "empty",
    };
  }
  if (line.startsWith("__OK__")) {
    const parts = line.split("\t");
    return {
      ok: true,
      method: parts[1] || "AXPress",
      app: parts[2] || undefined,
      kind: normalizeAccessibilityKind(parts[3] || ""),
      label: parts[4] || undefined,
      left: Number(parts[5]),
      top: Number(parts[6]),
      width: Number(parts[7]),
      height: Number(parts[8]),
    };
  }
  return {
    ok: false,
    method: "none",
    reason: parseSystemError(line),
  };
}

async function activateSystemTargetAtScreenPoint(x: number, y: number): Promise<SystemTargetActionResult> {
  if (process.platform !== "darwin") {
    return {
      ok: false,
      method: "none",
      reason: "unsupported-platform",
    };
  }
  const script = `
on clean_text(value_in)
  if value_in is missing value then return ""
  set asText to (value_in as text)
  set AppleScript's text item delimiters to {tab, return, linefeed}
  set pieces to text items of asText
  set AppleScript's text item delimiters to " "
  set joinedText to pieces as text
  set AppleScript's text item delimiters to ""
  return joinedText
end clean_text

on point_score(targetX, targetY, leftPos, topPos, w, h)
  set centerX to leftPos + (w / 2)
  set centerY to topPos + (h / 2)
  set dx to centerX - targetX
  set dy to centerY - targetY
  set distanceScore to (dx * dx + dy * dy) ^ 0.5
  if targetX >= leftPos and targetX <= (leftPos + w) and targetY >= topPos and targetY <= (topPos + h) then
    return distanceScore * 0.18
  end if
  return distanceScore + 20
end point_score

on run argv
  if (count of argv) < 3 then
    return "__ERROR__" & tab & "INVALID_ARGS"
  end if
  set targetX to item 1 of argv as real
  set targetY to item 2 of argv as real
  set selfPid to item 3 of argv as integer

  set allowedRoles to {"AXButton", "AXCheckBox", "AXRadioButton", "AXLink", "AXPopUpButton", "AXMenuButton", "AXDisclosureTriangle", "AXTextField", "AXTextArea", "AXComboBox", "AXMenuItem", "AXScrollBar", "AXTabButton"}
  set bestScore to 1000000000
  set bestElem to missing value
  set bestApp to ""
  set bestRole to ""
  set bestLabel to ""
  set bestLeft to 0
  set bestTop to 0
  set bestWidth to 0
  set bestHeight to 0

  tell application "System Events"
    if UI elements enabled is false then
      return "__ERROR__" & tab & "AX_DISABLED"
    end if

    set procList to (application processes whose background only is false)
    repeat with proc in procList
      set procPid to -1
      set procName to ""
      try
        set procPid to unix id of proc
      end try
      if procPid is selfPid then
      else
        try
          set procName to name of proc
        end try
        repeat with win in windows of proc
          try
            repeat with el in (entire contents of win)
              set roleName to ""
              try
                set roleName to role of el
              end try
              if allowedRoles contains roleName then
                try
                  set p to position of el
                  set s to size of el
                  set leftPos to item 1 of p
                  set topPos to item 2 of p
                  set w to item 1 of s
                  set h to item 2 of s
                  if (w > 7) and (h > 7) then
                    set score to my point_score(targetX, targetY, leftPos, topPos, w, h)
                    if score < bestScore then
                      set bestScore to score
                      set bestElem to el
                      set bestApp to my clean_text(procName)
                      set bestRole to my clean_text(roleName)

                      set labelText to ""
                      try
                        set labelText to value of attribute "AXDescription" of el
                      end try
                      if my clean_text(labelText) is "" then
                        try
                          set labelText to value of attribute "AXTitle" of el
                        end try
                      end if
                      if my clean_text(labelText) is "" then
                        try
                          set labelText to name of el
                        end try
                      end if
                      if my clean_text(labelText) is "" then set labelText to roleName

                      set bestLabel to my clean_text(labelText)
                      set bestLeft to leftPos
                      set bestTop to topPos
                      set bestWidth to w
                      set bestHeight to h
                    end if
                  end if
                end try
              end if
            end repeat
          end try
        end repeat
      end if
    end repeat

    if bestElem is missing value then
      return "__ERROR__" & tab & "NO_TARGET"
    end if

    set methodText to ""
    try
      perform action "AXPress" of bestElem
      set methodText to "AXPress"
    on error pressErr
      try
        set focused of bestElem to true
        set methodText to "AXFocus"
      on error focusErr
        return "__ERROR__" & tab & "PRESS_FAILED" & tab & my clean_text(pressErr) & tab & my clean_text(focusErr)
      end try
    end try
  end tell

  return "__OK__" & tab & methodText & tab & bestApp & tab & bestRole & tab & bestLabel & tab & (bestLeft as text) & tab & (bestTop as text) & tab & (bestWidth as text) & tab & (bestHeight as text)
end run
`;

  const result = await runOsaScript(
    script,
    [String(Math.max(0, x)), String(Math.max(0, y)), String(process.pid)],
    SYSTEM_ACTION_TIMEOUT_MS
  );
  if (result.timedOut) {
    return {
      ok: false,
      method: "none",
      reason: "timeout",
    };
  }
  if (result.code !== 0) {
    return {
      ok: false,
      method: "none",
      reason: `osascript-exit:${result.code}:${result.stderr || "unknown"}`,
    };
  }
  return parseSystemTargetActionResult(result.stdout);
}

async function commitSystemTargetAtPoint(targetX: number, targetY: number): Promise<SnapCommitResponse> {
  const response = await listSystemSnapTargets();
  if (!response.ok) {
    const reason = response.reason ? ` (${response.reason})` : "";
    return {
      ok: false,
      provider: response.provider,
      reason: response.reason,
      message: `System target discovery unavailable${reason}. Grant Accessibility permissions.`,
    };
  }
  if (!response.targets.length) {
    return {
      ok: false,
      provider: response.provider,
      reason: "no-targets",
      message: "No system snap targets found.",
    };
  }

  const now = Date.now();
  const ranked = response.targets
    .map(target => {
      const cx = target.left + target.width * 0.5;
      const cy = target.top + target.height * 0.5;
      const d = Math.hypot(cx - targetX, cy - targetY);
      let score = d;
      if (target.id === lastSnappedTargetId) {
        const cooldownLeft = Math.max(0, SNAP_TARGET_COOLDOWN_MS - (now - lastSnappedTargetAt));
        score += cooldownLeft > 0 ? 120 : 18;
      }
      return { target, cx, cy, score };
    })
    .sort((a, b) => a.score - b.score);

  const best = ranked[0];
  if (!best) {
    return {
      ok: false,
      provider: response.provider,
      reason: "no-ranked-target",
      message: "No ranked snap target available.",
    };
  }

  const axResult = await activateSystemTargetAtScreenPoint(best.cx, best.cy);
  if (axResult.ok) {
    lastSnappedTargetId = best.target.id;
    lastSnappedTargetAt = now;
    return {
      ok: true,
      provider: response.provider,
      method: axResult.method,
      message: `Activated ${best.target.label} via ${axResult.method}.`,
      target: {
        id: best.target.id,
        x: best.cx,
        y: best.cy,
        label: best.target.label,
        app: best.target.app,
      },
    };
  }

  const fallbackClicked = await clickCursorAtScreenPoint(best.cx, best.cy);
  if (fallbackClicked) {
    lastSnappedTargetId = best.target.id;
    lastSnappedTargetAt = now;
    return {
      ok: true,
      provider: response.provider,
      method: "cursor-click-fallback",
      reason: axResult.reason,
      message: `Activated ${best.target.label} via cursor fallback.`,
      target: {
        id: best.target.id,
        x: best.cx,
        y: best.cy,
        label: best.target.label,
        app: best.target.app,
      },
    };
  }

  return {
    ok: false,
    provider: response.provider,
    reason: axResult.reason || "activation-failed",
    message: "Failed to activate target via accessibility API and cursor fallback.",
    target: {
      id: best.target.id,
      x: best.cx,
      y: best.cy,
      label: best.target.label,
      app: best.target.app,
    },
  };
}

async function commitNearestSystemSnapTarget(): Promise<SnapCommitResponse> {
  const cursor = screen.getCursorScreenPoint();
  return await commitSystemTargetAtPoint(cursor.x, cursor.y);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function probeTcp(host: string, port: number, timeoutMs = 1000): Promise<boolean> {
  return new Promise(resolve => {
    const socket = new net.Socket();
    let settled = false;

    const finish = (ok: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      try {
        socket.destroy();
      } catch {
        // no-op
      }
      resolve(ok);
    };

    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    socket.connect(port, host);
  });
}

type HttpProbeResult = {
  ok: boolean;
  statusCode: number;
  body: string;
};

function probeHttp(
  host: string,
  port: number,
  pathName: string,
  timeoutMs = 1000
): Promise<HttpProbeResult> {
  return new Promise(resolve => {
    const req = http.request(
      {
        host,
        port,
        path: pathName,
        method: "GET",
        timeout: timeoutMs,
        headers: { Accept: "application/json" },
      },
      res => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", chunk => {
          body += chunk;
          if (body.length > 8192) {
            body = body.slice(0, 8192);
          }
        });
        res.on("end", () => {
          resolve({
            ok: true,
            statusCode: typeof res.statusCode === "number" ? res.statusCode : 0,
            body,
          });
        });
      }
    );

    req.on("timeout", () => {
      req.destroy();
      resolve({ ok: false, statusCode: 0, body: "" });
    });
    req.on("error", () => resolve({ ok: false, statusCode: 0, body: "" }));
    req.end();
  });
}

async function probeEegBackendHealth(timeoutMs = 1000): Promise<boolean> {
  const probe = await probeHttp("127.0.0.1", EEG_HTTP_PORT, EEG_HEALTH_PATH, timeoutMs);
  if (!probe.ok || probe.statusCode !== 200) {
    return false;
  }

  try {
    const parsed = JSON.parse(probe.body) as { service?: string };
    return parsed.service === EEG_SERVICE_ID;
  } catch {
    return false;
  }
}

async function waitForBackendTcp(
  host: string,
  port: number,
  timeoutMs: number,
  pid: number | null
): Promise<boolean> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (pid && !isPidAlive(pid)) {
      return false;
    }
    const ok = await probeTcp(host, port, 900);
    if (ok) {
      return true;
    }
    await sleep(250);
  }
  return false;
}

async function waitForEegBackendReady(
  timeoutMs: number,
  pid: number | null
): Promise<boolean> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (pid && !isPidAlive(pid)) {
      return false;
    }
    const ok = await probeEegBackendHealth(900);
    if (ok) {
      return true;
    }
    await sleep(250);
  }
  return false;
}

function isPidAlive(pid: number | null): boolean {
  if (!pid || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function killManagedBackendPid(pid: number, signal: NodeJS.Signals): boolean {
  if (process.platform === "win32") {
    try {
      const killArgs = ["/PID", String(pid), "/T", "/F"];
      const killer = spawn("taskkill", killArgs, {
        stdio: "ignore",
        windowsHide: true,
      });
      killer.unref();
      return true;
    } catch {
      return false;
    }
  }

  try {
    // Detached processes are their own process-group leaders; kill group first.
    process.kill(-pid, signal);
    return true;
  } catch {
    try {
      process.kill(pid, signal);
      return true;
    } catch {
      return false;
    }
  }
}

function stopCvBackend(options?: { force?: boolean; reason?: string }): { ok: boolean; message: string } {
  const pid = cvBackendPid;
  if (!pid) {
    return { ok: true, message: "CV backend not running." };
  }

  if (!isPidAlive(pid)) {
    cvBackendPid = null;
    return { ok: true, message: `CV backend already exited (pid ${pid}).` };
  }

  const force = options?.force === true;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  const killed = killManagedBackendPid(pid, signal);
  if (!killed) {
    return { ok: false, message: `Failed to stop CV backend (pid ${pid}).` };
  }

  cvBackendPid = null;
  const reason = options?.reason ? ` (${options.reason})` : "";
  return { ok: true, message: `Stopped CV backend (pid ${pid})${reason}.` };
}

function findProjectRoot(startDir: string): string | null {
  let dir = path.resolve(startDir);
  for (let i = 0; i < 8; i++) {
    const eyegazeEntry = path.join(dir, "eyegaze", "eye_tracker_service.py");
    if (fs.existsSync(eyegazeEntry)) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  return null;
}

function resolvePythonBinary(serviceDir: string): string {
  let pythonBin = path.join(serviceDir, ".venv", "bin", "python");
  if (process.platform === "win32") {
    pythonBin = path.join(serviceDir, ".venv", "Scripts", "python.exe");
  }
  if (!fs.existsSync(pythonBin)) {
    pythonBin = process.platform === "win32" ? "python" : "python3";
  }
  return pythonBin;
}

function launchCvBackendInBackground(cameraIndex: number): { ok: boolean; message: string } {
  if (isPidAlive(cvBackendPid)) {
    return { ok: true, message: `CV backend already running (pid ${cvBackendPid}).` };
  }
  cvBackendPid = null;

  const root = findProjectRoot(__dirname);
  if (!root) {
    return { ok: false, message: "Cannot find project root from Electron runtime path." };
  }

  const eyegazeDir = path.join(root, "eyegaze");
  const scriptPath = path.join(eyegazeDir, "eye_tracker_service.py");
  const taskPath = path.join(root, "assets", "models", "face_landmarker.task");
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, message: `Missing backend script: ${scriptPath}` };
  }
  if (!fs.existsSync(taskPath)) {
    return { ok: false, message: `Missing face landmarker task: ${taskPath}` };
  }

  const logsDir = path.join(root, ".dist");
  try {
    fs.mkdirSync(logsDir, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Cannot create logs directory: ${message}` };
  }
  const logPath = path.join(logsDir, "eyegaze-backend.log");

  const pythonBin = resolvePythonBinary(eyegazeDir);

  const cameraSource = CAMERA_BROKER_STREAM_URL || String(cameraIndex);
  const args = [
    "-u",
    scriptPath,
    "--camera",
    cameraSource,
    "--port",
    String(CV_EVENT_PORT),
    "--http-host",
    "127.0.0.1",
    "--http-port",
    String(CV_HTTP_PORT),
    "--http-streaming",
    "--cursor-mode",
    "legacy_3d",
    "--cursor-move",
    "--face-landmarker-task",
    taskPath,
    "--cursor-ramp-deadzone-px",
    "0",
    "--cursor-ramp-full-speed-px",
    "24",
    "--cursor-ramp-min-scale",
    "0.95",
    "--cursor-ramp-min-step-px",
    "0",
    "--cursor-gain",
    "9.5",
    "--cursor-bottom-gain-mult",
    "9",
    "--cursor-max-speed-px-s",
    "3600",
    "--one-euro-cutoff",
    "0.2",
    "--one-euro-beta",
    "0.02",
    "--one-euro-d-cutoff",
    "2.2",
  ];

  try {
    const logFd = fs.openSync(logPath, "a");
    try {
      const child = spawn(pythonBin, args, {
        cwd: eyegazeDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      cvBackendPid = child.pid ?? null;
      child.on("exit", () => {
        if (cvBackendPid === child.pid) {
          cvBackendPid = null;
        }
      });
      child.on("error", () => {
        if (cvBackendPid === child.pid) {
          cvBackendPid = null;
        }
      });
      child.unref();
      return {
        ok: true,
        message: `Launched backend in background (pid ${child.pid ?? "unknown"}). Logs: ${logPath}`,
      };
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Failed to spawn backend: ${message}` };
  }
}

async function startCvBackendAndWait(cameraIndex: number): Promise<{ ok: boolean; message: string }> {
  const launch = launchCvBackendInBackground(cameraIndex);
  if (!launch.ok) {
    return launch;
  }

  const ready = await waitForBackendTcp("127.0.0.1", CV_HTTP_PORT, 45000, cvBackendPid);
  if (!ready) {
    stopCvBackend({ force: true, reason: "startup-timeout" });
    return {
      ok: false,
      message: `CV backend did not become reachable on 127.0.0.1:${CV_HTTP_PORT} within 45s. ${launch.message}`.trim(),
    };
  }
  return launch;
}

function stopVoiceBackend(options?: { force?: boolean; reason?: string }): { ok: boolean; message: string } {
  const pid = voiceBackendPid;
  if (!pid) {
    return { ok: true, message: "Voice backend not running." };
  }

  if (!isPidAlive(pid)) {
    voiceBackendPid = null;
    return { ok: true, message: `Voice backend already exited (pid ${pid}).` };
  }

  const force = options?.force === true;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  const killed = killManagedBackendPid(pid, signal);
  if (!killed) {
    return { ok: false, message: `Failed to stop voice backend (pid ${pid}).` };
  }

  voiceBackendPid = null;
  const reason = options?.reason ? ` (${options.reason})` : "";
  return { ok: true, message: `Stopped voice backend (pid ${pid})${reason}.` };
}

function launchVoiceBackendInBackground(): { ok: boolean; message: string } {
  if (isPidAlive(voiceBackendPid)) {
    return { ok: true, message: `Voice backend already running (pid ${voiceBackendPid}).` };
  }
  voiceBackendPid = null;

  const root = findProjectRoot(__dirname);
  if (!root) {
    return { ok: false, message: "Cannot find project root from Electron runtime path." };
  }

  const voiceDir = path.join(root, "VoiceTTS");
  const scriptPath = path.join(voiceDir, "tts.py");
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, message: `Missing voice backend script: ${scriptPath}` };
  }

  const logsDir = path.join(root, ".dist");
  try {
    fs.mkdirSync(logsDir, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Cannot create logs directory: ${message}` };
  }
  const logPath = path.join(logsDir, "voice-backend.log");

  const pythonBin = resolvePythonBinary(voiceDir);
  const args = ["-u", scriptPath];

  try {
    const logFd = fs.openSync(logPath, "a");
    try {
      const child = spawn(pythonBin, args, {
        cwd: voiceDir,
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      voiceBackendPid = child.pid ?? null;
      child.on("exit", () => {
        if (voiceBackendPid === child.pid) {
          voiceBackendPid = null;
        }
      });
      child.on("error", () => {
        if (voiceBackendPid === child.pid) {
          voiceBackendPid = null;
        }
      });
      child.unref();
      return {
        ok: true,
        message: `Launched voice backend in background (pid ${child.pid ?? "unknown"}). Logs: ${logPath}`,
      };
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Failed to spawn voice backend: ${message}` };
  }
}

async function startVoiceBackendAndWait(): Promise<{ ok: boolean; message: string }> {
  const launch = launchVoiceBackendInBackground();
  if (!launch.ok) {
    return launch;
  }

  const ready = await waitForBackendTcp("127.0.0.1", 8766, 45000, voiceBackendPid);
  if (!ready) {
    stopVoiceBackend({ force: true, reason: "startup-timeout" });
    return {
      ok: false,
      message: `Voice backend did not become reachable on 127.0.0.1:8766 within 45s. ${launch.message}`.trim(),
    };
  }
  return launch;
}

function stopSignBackend(options?: { force?: boolean; reason?: string }): { ok: boolean; message: string } {
  const pid = signBackendPid;
  if (!pid) {
    return { ok: true, message: "Sign backend not running." };
  }

  if (!isPidAlive(pid)) {
    signBackendPid = null;
    return { ok: true, message: `Sign backend already exited (pid ${pid}).` };
  }

  const force = options?.force === true;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  const killed = killManagedBackendPid(pid, signal);
  if (!killed) {
    return { ok: false, message: `Failed to stop sign backend (pid ${pid}).` };
  }

  signBackendPid = null;
  const reason = options?.reason ? ` (${options.reason})` : "";
  return { ok: true, message: `Stopped sign backend (pid ${pid})${reason}.` };
}

function launchSignBackendInBackground(): { ok: boolean; message: string } {
  if (isPidAlive(signBackendPid)) {
    return { ok: true, message: `Sign backend already running (pid ${signBackendPid}).` };
  }
  signBackendPid = null;

  const root = findProjectRoot(__dirname);
  if (!root) {
    return { ok: false, message: "Cannot find project root from Electron runtime path." };
  }

  const signDir = path.join(root, "ASLCV", "Sign-Language-Recognition");
  const scriptPath = path.join(signDir, "app", "api.py");
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, message: `Missing sign backend script: ${scriptPath}` };
  }

  const logsDir = path.join(root, ".dist");
  try {
    fs.mkdirSync(logsDir, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Cannot create logs directory: ${message}` };
  }
  const logPath = path.join(logsDir, "sign-backend.log");

  const pythonBin = resolvePythonBinary(signDir);
  const args = ["-u", scriptPath];

  try {
    const logFd = fs.openSync(logPath, "a");
    try {
      const child = spawn(pythonBin, args, {
        cwd: signDir,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          CAMERA_BROKER_URL: CAMERA_BROKER_STREAM_URL,
        },
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      signBackendPid = child.pid ?? null;
      child.on("exit", () => {
        if (signBackendPid === child.pid) {
          signBackendPid = null;
        }
      });
      child.on("error", () => {
        if (signBackendPid === child.pid) {
          signBackendPid = null;
        }
      });
      child.unref();
      return {
        ok: true,
        message: `Launched sign backend in background (pid ${child.pid ?? "unknown"}). Logs: ${logPath}`,
      };
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Failed to spawn sign backend: ${message}` };
  }
}

async function startSignBackendAndWait(): Promise<{ ok: boolean; message: string }> {
  const launch = launchSignBackendInBackground();
  if (!launch.ok) {
    return launch;
  }

  const ready = await waitForBackendTcp("127.0.0.1", 8765, 60000, signBackendPid);
  if (!ready) {
    stopSignBackend({ force: true, reason: "startup-timeout" });
    return {
      ok: false,
      message: `Sign backend did not become reachable on 127.0.0.1:8765 within 60s. ${launch.message}`.trim(),
    };
  }
  return launch;
}

function stopEegBackend(options?: { force?: boolean; reason?: string }): { ok: boolean; message: string } {
  const pid = eegBackendPid;
  if (!pid) {
    return { ok: true, message: "EEG backend not running." };
  }

  if (!isPidAlive(pid)) {
    eegBackendPid = null;
    return { ok: true, message: `EEG backend already exited (pid ${pid}).` };
  }

  const force = options?.force === true;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  const killed = killManagedBackendPid(pid, signal);
  if (!killed) {
    return { ok: false, message: `Failed to stop EEG backend (pid ${pid}).` };
  }

  eegBackendPid = null;
  const reason = options?.reason ? ` (${options.reason})` : "";
  return { ok: true, message: `Stopped EEG backend (pid ${pid})${reason}.` };
}

function launchEegBackendInBackground(): { ok: boolean; message: string } {
  if (isPidAlive(eegBackendPid)) {
    return { ok: true, message: `EEG backend already running (pid ${eegBackendPid}).` };
  }
  eegBackendPid = null;

  const root = findProjectRoot(__dirname);
  if (!root) {
    return { ok: false, message: "Cannot find project root from Electron runtime path." };
  }

  const eegDir = path.join(root, "muse");
  const scriptPath = path.join(eegDir, "realtime_muse.py");
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, message: `Missing EEG backend script: ${scriptPath}` };
  }

  const logsDir = path.join(root, ".dist");
  try {
    fs.mkdirSync(logsDir, { recursive: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Cannot create logs directory: ${message}` };
  }
  const logPath = path.join(logsDir, "eeg-backend.log");

  const pythonBin = resolvePythonBinary(eegDir);
  const args = ["-u", scriptPath];

  try {
    const logFd = fs.openSync(logPath, "a");
    try {
      const child = spawn(pythonBin, args, {
        cwd: eegDir,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          EEG_STREAM_PORT: String(EEG_HTTP_PORT),
        },
        detached: true,
        stdio: ["ignore", logFd, logFd],
      });
      eegBackendPid = child.pid ?? null;
      child.on("exit", () => {
        if (eegBackendPid === child.pid) {
          eegBackendPid = null;
        }
      });
      child.on("error", () => {
        if (eegBackendPid === child.pid) {
          eegBackendPid = null;
        }
      });
      child.unref();
      return {
        ok: true,
        message: `Launched EEG backend in background (pid ${child.pid ?? "unknown"}). Logs: ${logPath}`,
      };
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message: `Failed to spawn EEG backend: ${message}` };
  }
}

async function startEegBackendAndWait(): Promise<{ ok: boolean; message: string }> {
  const launch = launchEegBackendInBackground();
  if (!launch.ok) {
    return launch;
  }

  const ready = await waitForEegBackendReady(30000, eegBackendPid);
  if (!ready) {
    stopEegBackend({ force: true, reason: "startup-timeout" });
    return {
      ok: false,
      message: `EEG backend did not pass health checks on 127.0.0.1:${EEG_HTTP_PORT}${EEG_HEALTH_PATH} within 30s. ${launch.message}`.trim(),
    };
  }
  return launch;
}

function createWindow(mode: 'full' | 'overlay' = 'full') {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth } = primaryDisplay.workAreaSize;

  console.log("Creating window with mode:", mode);
  currentMode = mode; // Update the global state

  const commonOptions = {
    backgroundColor: "#0b0b0b",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  };

  if (mode === 'overlay') {
    // Overlay mode
    win = new BrowserWindow({
      ...commonOptions,
      width: screenWidth,
      height: 60,
      x: 0,
      y: 0,
      frame: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: false,
      resizable: false,
      movable: true,
    });
  } else {
    // Full app mode - no native title bar; use app toolbar for minimize/close
    win = new BrowserWindow({
      ...commonOptions,
      width: 1100,
      height: 720,
      frame: false,
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : undefined,
      transparent: false,
      alwaysOnTop: false,
      resizable: true,
      movable: true,
    });
    win.center();
  }

  const createdWindow = win;
  if (!createdWindow) {
    return;
  }

  // Check if we're in development mode
  const isDev = !app.isPackaged;
  
  if (isDev) {
    createdWindow.loadURL('http://localhost:5173');
    createdWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    createdWindow.loadFile(path.join(__dirname, "../index.html"));
  }

  createdWindow.on('closed', () => {
    if (win === createdWindow) {
      win = null;
    }
  });

  // Send overlay mode to renderer once it's ready
  createdWindow.webContents.on('did-finish-load', () => {
    const isOverlay = mode === 'overlay';
    console.log("Window loaded, sending overlay mode:", isOverlay);
    createdWindow.webContents.send('overlay-mode-changed', isOverlay);
  });
}

// Toggle between overlay and full app mode
ipcMain.handle("app:toggle-overlay", (event, args?: { targetPath?: string }) => {
  const newMode: 'full' | 'overlay' = currentMode === 'overlay' ? 'full' : 'overlay';
  const willBeOverlay = newMode === 'overlay';
  const requestedTargetPath = args?.targetPath;

  if (newMode === "full" && typeof requestedTargetPath === "string" && requestedTargetPath.startsWith("/")) {
    pendingFullRoute = requestedTargetPath;
  } else {
    pendingFullRoute = null;
  }

  console.log("Toggle overlay: currentMode=", currentMode, "-> newMode=", newMode);

  // Build the replacement window first, then close the sender window.
  // This avoids race conditions where the close event never reaches our swap path.
  const senderWindow = BrowserWindow.fromWebContents(event.sender);
  isToggling = true;
  try {
    createWindow(newMode);
    if (senderWindow && !senderWindow.isDestroyed()) {
      senderWindow.destroy();
    }
    console.log("New window created, mode=", newMode);
    return willBeOverlay;
  } finally {
    isToggling = false;
  }
});

// Get current overlay mode
ipcMain.handle("app:get-overlay-mode", () => {
  return currentMode === 'overlay';
});

ipcMain.handle("app:consume-pending-route", () => {
  const route = pendingFullRoute;
  pendingFullRoute = null;
  return route;
});

ipcMain.handle("app:minimize", () => win?.minimize());
ipcMain.handle("app:maximize", () => {
  if (win?.isMaximized()) {
    win?.unmaximize();
  } else {
    win?.maximize();
  }
});
ipcMain.handle("app:close", () => win?.close());

ipcMain.handle("cv:start-backend", async (_event, args?: { camera?: number }) => {
  const requestedCamera = Number.isFinite(args?.camera) ? Number(args?.camera) : 0;
  const safeCamera = Math.max(0, Math.min(9, Math.trunc(requestedCamera)));
  try {
    return await startCvBackendAndWait(safeCamera);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
});

ipcMain.handle("cv:stop-backend", () => {
  return stopCvBackend({ force: true, reason: "ipc" });
});

ipcMain.handle("cv:is-backend-running", async () => {
  try {
    return await probeTcp("127.0.0.1", CV_HTTP_PORT, 350);
  } catch {
    return false;
  }
});

ipcMain.handle("voice:start-backend", async () => {
  try {
    return await startVoiceBackendAndWait();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
});

ipcMain.handle("voice:stop-backend", () => {
  return stopVoiceBackend({ force: true, reason: "ipc" });
});

ipcMain.handle("voice:is-backend-running", async () => {
  try {
    return await probeTcp("127.0.0.1", 8766, 350);
  } catch {
    return false;
  }
});

ipcMain.handle("sign:start-backend", async () => {
  try {
    return await startSignBackendAndWait();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
});

ipcMain.handle("sign:stop-backend", () => {
  return stopSignBackend({ force: true, reason: "ipc" });
});

ipcMain.handle("sign:is-backend-running", async () => {
  try {
    return await probeTcp("127.0.0.1", 8765, 350);
  } catch {
    return false;
  }
});

ipcMain.handle("eeg:start-backend", async () => {
  try {
    return await startEegBackendAndWait();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, message };
  }
});

ipcMain.handle("eeg:stop-backend", () => {
  return stopEegBackend({ force: true, reason: "ipc" });
});

ipcMain.handle("eeg:is-backend-running", async () => {
  try {
    return await probeEegBackendHealth(350);
  } catch {
    return false;
  }
});

ipcMain.handle("snap:move-cursor", async (_event, args?: { x?: number; y?: number }) => {
  const x = Number(args?.x);
  const y = Number(args?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return { ok: false, message: "Invalid cursor coordinates." };
  }
  const moved = await moveCursorToScreenPoint(x, y);
  if (!moved) {
    return { ok: false, message: "Failed to move system cursor." };
  }
  return { ok: true, message: "Cursor moved." };
});

ipcMain.handle("snap:commit-nearest", async () => {
  return await commitNearestSystemSnapTarget();
});

ipcMain.handle("snap:commit-target", async (_event, args?: { x?: number; y?: number }) => {
  const x = Number(args?.x);
  const y = Number(args?.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return {
      ok: false,
      message: "Invalid target coordinates.",
      reason: "invalid-target-coordinates",
    } satisfies SnapCommitResponse;
  }
  return await commitSystemTargetAtPoint(x, y);
});

ipcMain.handle("snap:get-cursor-screen-point", () => {
  return screen.getCursorScreenPoint();
});

ipcMain.handle("snap:list-system-targets", async () => {
  return await listSystemSnapTargets();
});

app.whenReady().then(() => {
  createWindow('full');

  const snapShortcutRegistered = globalShortcut.register(SNAP_GLOBAL_SHORTCUT, () => {
    void commitNearestSystemSnapTarget();
  });
  if (!snapShortcutRegistered) {
    console.warn(`Failed to register global snap shortcut: ${SNAP_GLOBAL_SHORTCUT}`);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow('full');
    }
  });
});

app.on('window-all-closed', () => {
  // Don't quit if we're in the middle of toggling overlay ↔ full (we're about to create a new window)
  if (isToggling) {
    return;
  }
  stopCvBackend({ force: true, reason: "window-all-closed" });
  stopVoiceBackend({ force: true, reason: "window-all-closed" });
  stopSignBackend({ force: true, reason: "window-all-closed" });
  stopEegBackend({ force: true, reason: "window-all-closed" });
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on("before-quit", () => {
  globalShortcut.unregisterAll();
  stopCvBackend({ force: true, reason: "before-quit" });
  stopVoiceBackend({ force: true, reason: "before-quit" });
  stopSignBackend({ force: true, reason: "before-quit" });
  stopEegBackend({ force: true, reason: "before-quit" });
});
