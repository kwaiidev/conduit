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

export default function OverlayBar() {
  const [signalStrength] = useState(98);
  const [leftJawSignal, setLeftJawSignal] = useState(false);
  const [middleJawSignal, setMiddleJawSignal] = useState(false);
  const [rightJawSignal, setRightJawSignal] = useState(false);
  const [voiceOn, setVoiceOn] = useState(false);
  const [opticOpen, setOpticOpen] = useState(false);
  const [aslOn, setAslOn] = useState(false);
  

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
  useEffect(() => {
    const t = setInterval(() => {
      setMiddleJawSignal((s) => !s);
    }, 3000);
    return () => clearInterval(t);
  }, []);

  const exitOverlay = async () => {
    try {
      await window.electron?.toggleOverlay();
    } catch (e) {
      console.error("Toggle overlay:", e);
    }
  };

  return (
    <div style={styles.overlay}>
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
          label="M-JAW"
          active={middleJawSignal}
          size="small"
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
        onClick={() => setVoiceOn((v) => !v)}
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
  onClick={() => setAslOn((a) => !a)}
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
        style={styles.expandButton}
        title="Expand to full window"
      >
        <Maximize2 size={18} />
        <span>Expand</span>
      </button>
    </div>
  );
}

function JawButton({
  label,
  active,
  crescentSide,
  size = "large",
}: {
  label: string;
  active: boolean;
  crescentSide?: "left" | "right";
  size?: "large" | "small";
}) {
  const isSmall = size === "small";

  return (
    <div style={styles.jawButtonWrap}>
      <div
        style={{
          ...styles.jawCircle,
          ...(isSmall ? styles.jawCircleSmall : {}),
          ...(active ? styles.jawCircleActive : {}),
        }}
        className={active ? "overlay-jaw-active" : ""}
      >
        {!isSmall && crescentSide && (
          <span
            style={{
              ...styles.crescent,
              ...(crescentSide === "left" ? styles.crescentLeft : styles.crescentRight),
            }}
          />
        )}
        {isSmall && active && <span style={styles.smallGlow} />}
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
  jawCircleSmall: {
    width: 28,
    height: 28,
    background: "#e5e7eb",
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
  smallGlow: {
    position: "absolute",
    width: "100%",
    height: "100%",
    borderRadius: "50%",
    background: "rgba(255, 45, 141, 0.35)",
    animation: "overlay-jaw-hum 1.2s ease-in-out infinite",
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
};
