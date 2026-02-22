import React from "react";
import { motion } from "motion/react";
import { disableEEG, enableEEG } from "../../lib/eeg";
import { useTheme } from "../../context/ThemeContext";
import leftDark from "../../assets/leftDark.gif";
import leftLight from "../../assets/leftLight.gif";
import rightDark from "../../assets/rightDark.gif";
import rightLight from "../../assets/rightLight.gif";

type EegTrainingCardProps = {
  instruction: string;
  cue?: "left" | "right";
};

const CUE_ASSETS = {
  left: {
    light: leftLight,
    dark: leftDark,
    alt: "Left jaw clench EEG cue",
  },
  right: {
    light: rightLight,
    dark: rightDark,
    alt: "Right jaw clench EEG cue",
  },
} as const;

export function EegTrainingCard({ instruction, cue }: EegTrainingCardProps) {
  const { isDark } = useTheme();
  const [isTraining, setIsTraining] = React.useState(false);
  const [isStarting, setIsStarting] = React.useState(false);
  const [statusMessage, setStatusMessage] = React.useState("Press Start to launch EEG service and begin.");
  const cueAsset = cue ? (isDark ? CUE_ASSETS[cue].dark : CUE_ASSETS[cue].light) : null;
  const cueAlt = cue ? CUE_ASSETS[cue].alt : "";

  const startTraining = React.useCallback(async () => {
    setIsStarting(true);
    setStatusMessage("Launching EEG backend and waiting for stream confirmation...");
    try {
      const launchResult = await window.electron.startEegBackend();
      if (!launchResult.ok) {
        setStatusMessage(`Cannot start EEG backend: ${launchResult.message}`);
        return;
      }
      await enableEEG();
      setIsTraining(true);
      setStatusMessage("EEG service active. Perform the exercise for 3 seconds.");
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      setStatusMessage(`EEG start failed: ${detail}`);
    } finally {
      setIsStarting(false);
    }
  }, []);

  const stopTraining = React.useCallback(async () => {
    setIsTraining(false);
    setStatusMessage("Training paused.");
    try {
      await disableEEG();
    } catch {
      // no-op: keep UI responsive even if backend endpoint is temporarily unavailable
    }
  }, []);

  return (
    <div style={styles.container}>
      <div style={styles.headerBlock}>
        <p style={styles.subtitle}>{instruction}</p>
        <p style={styles.statusText}>{statusMessage}</p>
      </div>

      <div style={styles.gifCard}>
        {cueAsset ? (
          <div style={styles.gifInner}>
            <img src={cueAsset} alt={cueAlt} style={styles.gifImage} />
          </div>
        ) : null}

        <div style={styles.feedGrid}>
          <img
            src="http://localhost:8770/waves"
            alt="EEG waves calibration feed"
            style={styles.feedImage}
          />
          <img
            src="http://localhost:8770/topo"
            alt="EEG topomap calibration feed"
            style={styles.feedImage}
          />
        </div>
      </div>

      <div style={styles.buttonWrapper}>
        {!isTraining ? (
          <button
            type="button"
            onClick={startTraining}
            style={{
              ...styles.startButton,
              ...(isStarting ? styles.buttonDisabled : {}),
            }}
            disabled={isStarting}
          >
            {isStarting ? "Starting..." : "Start"}
          </button>
        ) : (
          <button
            type="button"
            onClick={stopTraining}
            style={styles.statusBar}
          >
            <motion.span
              style={styles.statusDot}
              animate={{ scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            Training...
            <div style={styles.fakeFill} />
          </button>
        )}
      </div>
    </div>
  );
}

const BUTTON_WIDTH = 180;
const BUTTON_HEIGHT = 42;
const CARD_MAX_WIDTH = 500;

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 24,
    width: "100%",
  },

  headerBlock: {
    width: "100%",
    maxWidth: CARD_MAX_WIDTH,
    textAlign: "center",
  },

  subtitle: {
    margin: 0,
    fontSize: 14,
    color: "var(--text-secondary)",
  },

  statusText: {
    margin: "10px 0 0 0",
    fontSize: 12,
    color: "var(--text-secondary)",
  },

  gifCard: {
    width: "100%",
    maxWidth: CARD_MAX_WIDTH,
    borderRadius: 20,
    border: "1px solid var(--pink-border)",
    background: "var(--bg-secondary)",
    padding: 16,
    display: "grid",
    gap: 12,
  },

  feedGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: 12,
  },

  feedImage: {
    width: "100%",
    height: 84,
    borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--bg-primary)",
    objectFit: "cover",
    overflow: "hidden",
  },

  gifInner: {
    minHeight: 250,
    borderRadius: 16,
    border: "1px solid transparent",
    background: "linear-gradient(180deg, rgba(255,45,141,0.12), rgba(255,45,141,0.04))",
    display: "grid",
    placeItems: "center",
    padding: 8,
    overflow: "hidden",
  },

  gifText: {
    fontSize: 12,
    fontWeight: 900,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "var(--text-secondary)",
  },

  gifImage: {
    width: "88%",
    height: "auto",
    maxWidth: 360,
    maxHeight: 220,
    borderRadius: 12,
    objectFit: "contain",
    display: "block",
  },

  buttonWrapper: {
    width: "100%",
    maxWidth: CARD_MAX_WIDTH,
    display: "flex",
    justifyContent: "center",
  },

  startButton: {
    width: BUTTON_WIDTH,
    height: BUTTON_HEIGHT,
    borderRadius: 16,
    border: "1px solid rgba(255,45,141,0.35)",
    background: "#FF2D8D",
    color: "white",
    fontWeight: 900,
    cursor: "pointer",
    boxShadow: "0 10px 25px rgba(255, 45, 141, 0.18)",
  },

  buttonDisabled: {
    cursor: "default",
    opacity: 0.65,
  },

  statusBar: {
    position: "relative",
    width: BUTTON_WIDTH,
    height: BUTTON_HEIGHT,
    borderRadius: 16,
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    color: "var(--text-primary)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    fontWeight: 900,
    cursor: "pointer",
    overflow: "hidden",
  },

  statusDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: "#FF2D8D",
    boxShadow: "0 0 0 6px rgba(255,45,141,0.12)",
  },

  fakeFill: {
    position: "absolute",
    inset: 0,
    width: "60%",
    background: "rgba(255,45,141,0.12)",
    zIndex: 0,
  },
};
