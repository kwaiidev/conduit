import React from "react";
import { motion } from "motion/react";

export function CVCursorCalibrationCenter() {
  const [isCalibrating, setIsCalibrating] = React.useState(false);

  const startCalibration = async () => {
    setIsCalibrating(true);

    // TODO: replace with your real API call
    // await fetch("/api/cv/calibrate-center", { method: "POST" });

    // no countdown logic; just UI toggle for now
  };

  return (
    <div style={styles.container}>
      {/* Centered Header (aligned to GIF width) */}
      <div style={styles.headerBlock}>
        <p style={styles.subtitle}>
          Look at the center target and hold still. We’ll calibrate your CV cursor.
        </p>
      </div>

      {/* Center Target Placeholder */}
      <div style={styles.gifCard}>
        <div style={styles.gifInner}>
          <div style={styles.target}>
            <div style={styles.targetRing} />
            <div style={styles.targetDot} />
            <div style={styles.crosshairV} />
            <div style={styles.crosshairH} />
          </div>

          <div style={styles.gifText}>CENTER TARGET</div>
        </div>
      </div>

      {/* Centered Start Button OR Status Bar */}
      <div style={styles.buttonWrapper}>
        {!isCalibrating ? (
          <button type="button" onClick={startCalibration} style={styles.startButton}>
            Start
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setIsCalibrating(false)}
            style={styles.statusBar}
            title="Click to stop (toggle)"
          >
            <motion.span
              style={styles.statusDot}
              animate={{ scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            Calibrating…
            <div style={styles.fakeFill} />
          </button>
        )}
      </div>
    </div>
  );
}

const BUTTON_WIDTH = 180;
const BUTTON_HEIGHT = 42;
const GIF_WIDTH = 420;

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 24,
    width: "100%",
  },

  headerBlock: {
    width: GIF_WIDTH,
    textAlign: "center",
  },

  subtitle: {
    margin: "8px 0 0 0",
    fontSize: 14,
    color: "var(--text-secondary)",
  },

  gifCard: {
    width: GIF_WIDTH,
    borderRadius: 20,
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    padding: 16,
  },

  gifInner: {
    height: 240,
    borderRadius: 16,
    border: "1px dashed var(--border)",
    background: "var(--bg-tertiary)",
    display: "grid",
    placeItems: "center",
    position: "relative",
    overflow: "hidden",
  },

  gifText: {
    position: "absolute",
    bottom: 14,
    fontSize: 12,
    fontWeight: 900,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "var(--text-secondary)",
    opacity: 0.9,
  },

  /* Center target */
  target: {
    position: "relative",
    width: 90,
    height: 90,
    display: "grid",
    placeItems: "center",
  },
  targetRing: {
    position: "absolute",
    width: 72,
    height: 72,
    borderRadius: "50%",
    border: "2px solid rgba(255,45,141,0.55)",
    boxShadow: "0 0 0 12px rgba(255,45,141,0.10)",
  },
  targetDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    background: "#FF2D8D",
  },
  crosshairV: {
    position: "absolute",
    width: 2,
    height: 56,
    background: "rgba(255,45,141,0.35)",
    borderRadius: 2,
  },
  crosshairH: {
    position: "absolute",
    height: 2,
    width: 56,
    background: "rgba(255,45,141,0.35)",
    borderRadius: 2,
  },

  buttonWrapper: {
    width: GIF_WIDTH,
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
    zIndex: 1,
  },

  fakeFill: {
    position: "absolute",
    inset: 0,
    width: "55%",
    background: "rgba(255,45,141,0.12)",
    zIndex: 0,
  },
};