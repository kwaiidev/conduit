import React, { useState, useEffect } from "react";
import {
  Zap,
  Mic,
  MicOff,
  Keyboard,
  Info,
  Settings,
  Power,
  Maximize2,
  Hand,
} from "lucide-react";
import { enableASL, disableASL, getASLReady } from "../lib/aslcv";
import { enableVoice, disableVoice, getVoiceReady } from "../lib/voicetts";

const SIGN_TEXT_PREFERENCE_KEY = "conduit-modality-intent-sign-text";

function getModalityPreference(): boolean | null {
  if (typeof window === "undefined" || !window.localStorage) {
    return null;
  }
  const value = localStorage.getItem(SIGN_TEXT_PREFERENCE_KEY);
  if (value === null) {
    return null;
  }
  return value === "1";
}

function setModalityPreference(enabled: boolean): void {
  if (typeof window === "undefined" || !window.localStorage) {
    return;
  }
  localStorage.setItem(SIGN_TEXT_PREFERENCE_KEY, enabled ? "1" : "0");
}

export default function OverlayBar() {
  const isMacOS =
    typeof navigator !== "undefined" &&
    /Mac|iPhone|iPad|iPod/.test(navigator.platform);
  const [signalStrength] = useState(98);
  const [leftJawSignal, setLeftJawSignal] = useState(false);
  const [rightJawSignal, setRightJawSignal] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false);
  const [opticOpen, setOpticOpen] = useState(false);
  const [aslOn, setAslOn] = useState(false);
  const [isSwitchingOverlay, setIsSwitchingOverlay] = useState(false);

  // Sync toggles with real API state on mount
  useEffect(() => {
    void (async () => {
      const [voiceReady, aslReady] = await Promise.all([
        getVoiceReady(),
        getASLReady(),
      ]);
      setVoiceOn(voiceReady);

      const userIntent = getModalityPreference();
      setAslOn(userIntent === null ? false : userIntent && aslReady);
    })();
  }, []);

  const toggleVoice = async () => {
    const next = !voiceOn;
    setVoiceOn(next);
    try {
      if (next) {
        await enableVoice();
      } else {
        await disableVoice();
      }
    } catch (e) {
      console.error("Voice toggle failed:", e);
      setVoiceOn(!next);
    }
  };

  const toggleASL = async () => {
    const next = !aslOn;
    setAslOn(next);
    setModalityPreference(next);
    try {
      if (next) {
        await enableASL();
      } else {
        await disableASL();
      }
    } catch (e) {
      console.error("ASL toggle failed:", e);
      setAslOn(!next);
      setModalityPreference(!next);
    }
  };


  // Simulate jaw signals for demo (hum on/off soft pink). Replace with real EEG/sensor data.
  useEffect(() => {
    const t = setInterval(() => {
      setLeftJawSignal((s) => !s);
    }, 2000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    const t = setInterval(() => {
      setRightJawSignal((s) => !s);
    }, 2500);
    return () => clearInterval(t);
  }, []);

  const exitOverlay = async () => {
    if (isSwitchingOverlay) {
      return;
    }
    if (typeof window.electron?.toggleOverlay !== "function") {
      console.error("Overlay toggle is unavailable in this runtime.");
      return;
    }
    setIsSwitchingOverlay(true);
    try {
      await window.electron.toggleOverlay({ targetPath: "/home" });
    } catch (e) {
      console.error("Toggle overlay:", e);
    } finally {
      setIsSwitchingOverlay(false);
    }
  };

  return (
    <div style={{ ...styles.overlay, ...(isMacOS ? styles.overlayMac : {}) }}>
      {/* Logo / Signal */}
      <div style={styles.signalSection}>
        <div style={styles.signalIcons}>
          <span style={styles.greenDot} />
          <Zap size={18} style={{ color: "#FF2D8D" }} />
        </div>
        <span style={styles.signalLabel}>SIGNAL</span>
        <span style={styles.signalValue}>{signalStrength}%</span>
      </div>

      {/* Jaw sensors */}
      <div style={styles.jawSection}>
        <JawButton
          label="L-JAW"
          active={leftJawSignal}
          crescentSide="right"
        />
        <JawButton
          label="R-JAW"
          active={rightJawSignal}
          crescentSide="left"
        />
      </div>

      {/* Voice toggle */}
      <button
        type="button"
        onClick={toggleVoice}
        style={styles.iconButton}
        title={voiceOn ? "Voice on" : "Voice off"}
      >
        {voiceOn ? (
          <Mic size={22} style={{ color: "var(--pink, #FF2D8D)" }} />
        ) : (
          <MicOff size={22} style={{ color: "#9ca3af" }} />
        )}
        <span style={styles.iconLabel}>VOICE</span>
      </button>

      {/* Optic keyboard toggle */}
<button
  type="button"
  onClick={() => setOpticOpen((o) => !o)}
  style={styles.iconButton}
  title={opticOpen ? "Close optic keyboard" : "Open optic keyboard"}
>
  <Keyboard
    size={22}
    style={{ color: opticOpen ? "var(--pink, #FF2D8D)" : "#9ca3af" }}
  />
  <span style={styles.iconLabel}>OPTIC</span>
</button>

{/* ASL to Text toggle */}
<button
  type="button"
  onClick={toggleASL}
  style={styles.iconButton}
  title={aslOn ? "ASL to Text enabled" : "Enable ASL to Text"}
>
  <Hand
    size={22}
    style={{ color: aslOn ? "var(--pink, #FF2D8D)" : "#9ca3af" }}
  />
  <span style={styles.iconLabel}>ASL</span>
</button>
      {/* Mode: SAFE PASSIVE */}
      <div style={styles.modePill}>
        <span style={styles.modeDot} />
        <span style={styles.modeText}>SAFE PASSIVE</span>
      </div>

      {/* Utility icons */}
      <button type="button" style={styles.utilityButton} title="Information">
        <Info size={18} style={{ color: "#6b7280" }} />
      </button>
      <button type="button" style={styles.utilityButton} title="Settings">
        <Settings size={18} style={{ color: "#6b7280" }} />
      </button>
      <button type="button" style={styles.powerButton} title="Power">
        <Power size={20} style={{ color: "#fff" }} />
      </button>

      {/* Expand to full window */}
      <button
        type="button"
        onClick={exitOverlay}
        style={{
          ...styles.expandButton,
          ...(isSwitchingOverlay ? styles.expandButtonDisabled : {}),
        }}
        title={isSwitchingOverlay ? "Switching back to full window..." : "Expand to full window"}
        disabled={isSwitchingOverlay}
      >
        <Maximize2 size={18} />
        <span>{isSwitchingOverlay ? "Switching..." : "Expand"}</span>
      </button>
    </div>
  );
}

function JawButton({
  label,
  active,
  crescentSide,
}: {
  label: string;
  active: boolean;
  crescentSide?: "left" | "right";
}) {
  return (
    <div style={styles.jawButtonWrap}>
      <div
        style={{
          ...styles.jawCircle,
          ...(active ? styles.jawCircleActive : {}),
        }}
        className={active ? "overlay-jaw-active" : ""}
      >
        {crescentSide && (
          <span
            style={{
              ...styles.crescent,
              ...(crescentSide === "left" ? styles.crescentLeft : styles.crescentRight),
            }}
          />
        )}
      </div>
      <span
        style={{
          ...styles.jawLabel,
          ...(active ? styles.jawLabelActive : {}),
        }}
      >
        {label}
      </span>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-start",
    gap: 24,
    padding: "12px 20px",
    height: 72,
    background: "#faf9f7",
    borderRadius: 16,
    boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
    border: "1px solid rgba(0,0,0,0.06)",
    margin: 8,
  },
  overlayMac: {
    padding: "6px 20px",
    transform: "translateY(-12%)",
  },
  signalSection: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: 2,
    minWidth: 56,
  },
  signalIcons: {
    display: "flex",
    alignItems: "center",
    gap: 6,
  },
  greenDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#22c55e",
    animation: "overlay-pulse 1.5s ease-in-out infinite",
  },
  signalLabel: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.08em",
    color: "#9ca3af",
    textTransform: "uppercase",
  },
  signalValue: {
    fontSize: 20,
    fontWeight: 700,
    color: "#111",
  },
  jawSection: {
    display: "flex",
    alignItems: "center",
    gap: 20,
  },
  jawButtonWrap: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
  },
  jawCircle: {
    width: 44,
    height: 44,
    borderRadius: "50%",
    background: "#1e293b",
    position: "relative",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "box-shadow 0.3s ease, background 0.3s ease",
  },
  jawCircleActive: {
    background: "#1e293b",
    boxShadow: "0 0 0 0 rgba(255, 45, 141, 0.4)",
  },
  crescent: {
    position: "absolute",
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "rgba(255,255,255,0.95)",
  },
  crescentLeft: {
    left: 4,
    boxShadow: "4px 0 0 0 #1e293b",
  },
  crescentRight: {
    right: 4,
    boxShadow: "-4px 0 0 0 #1e293b",
  },
  jawLabel: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.04em",
    color: "#9ca3af",
  },
  jawLabelActive: {
    color: "#111",
  },
  iconButton: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 4,
    padding: "6px 8px",
    background: "transparent",
    border: "none",
    cursor: "pointer",
    borderRadius: 8,
    transition: "background 0.2s",
  },
  iconLabel: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.04em",
    color: "#9ca3af",
  },
  modePill: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 14px",
    background: "#fff7ed",
    border: "1px solid #fed7aa",
    borderRadius: 12,
    marginLeft: "auto",
  },
  modeDot: {
    width: 6,
    height: 6,
    borderRadius: "50%",
    background: "#ea580c",
  },
  modeText: {
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: "0.06em",
    color: "#ea580c",
    textTransform: "uppercase",
  },
  utilityButton: {
    width: 36,
    height: 36,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "transparent",
    border: "none",
    borderRadius: "50%",
    cursor: "pointer",
  },
  powerButton: {
    width: 44,
    height: 44,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#1e293b",
    border: "none",
    borderRadius: "50%",
    cursor: "pointer",
  },
  expandButton: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "8px 14px",
    background: "#FF2D8D",
    border: "none",
    borderRadius: 10,
    color: "#fff",
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  expandButtonDisabled: {
    opacity: 0.7,
    cursor: "wait",
  },
};
