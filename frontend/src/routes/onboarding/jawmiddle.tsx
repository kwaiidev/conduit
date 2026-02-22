import React from "react";
import { motion } from "motion/react";

export function JawClenchTrainingMiddle() {
  const [isTraining, setIsTraining] = React.useState(false);

  return (
    <div style={styles.container}>
      {/* Centered Header (aligned to GIF width) */}
      <div style={styles.headerBlock}>
        <p style={styles.subtitle}>
          Bite down on the right jaw and hold for 3 seconds.
        </p>
      </div>

      {/* GIF Placeholder */}
      <div style={styles.gifCard}>
        <div style={styles.gifInner}>
          <div style={styles.gifText}>GIF / Visual Cue Here</div>
        </div>
      </div>

      {/* Centered Start Button OR Status Bar */}
      <div style={styles.buttonWrapper}>
        {!isTraining ? (
          <button
            type="button"
            onClick={() => setIsTraining(true)}
            style={styles.startButton}
          >
            Start
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setIsTraining(false)}
            style={styles.statusBar}
          >
            <motion.span
              style={styles.statusDot}
              animate={{ scale: [1, 1.25, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
            />
            Training…
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

  title: {
    margin: 0,
    fontSize: 20,
    fontWeight: 900,
    color: "var(--text-primary)",
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
  },

  gifText: {
    fontSize: 12,
    fontWeight: 900,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "var(--text-secondary)",
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
  },

  fakeFill: {
    position: "absolute",
    inset: 0,
    width: "60%",
    background: "rgba(255,45,141,0.12)",
    zIndex: 0,
  },
};