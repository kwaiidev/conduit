import React, { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { ArrowLeft, Zap, Waves, Grid3X3 } from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.08 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 280, damping: 26 },
  },
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "22px",
    padding: "40px",
    maxWidth: "1400px",
    margin: "0 auto",
    width: "100%",
    background: "var(--shell-content-bg)",
    minHeight: "100%",
    paddingBottom: "96px",
  },

  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-end",
    gap: "16px",
  },
  headerLeft: {
    display: "flex",
    alignItems: "flex-end",
    gap: "16px",
  },
  headerTitles: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
  },

  backButton: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    background: "var(--bg-secondary)",
    border: "1px solid var(--border)",
    color: "var(--text-primary)",
    padding: "10px 12px",
    borderRadius: "14px",
    cursor: "pointer",
    fontWeight: 700,
  },

  title: {
    fontSize: "40px",
    fontWeight: 800,
    margin: 0,
    color: "var(--text-primary)",
    letterSpacing: "-0.02em",
    lineHeight: 1.1,
  },

  statusBadge: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    color: "#FF2D8D",
    fontWeight: 800,
    letterSpacing: "0.12em",
    fontSize: "12px",
    textTransform: "uppercase",
  },
  statusDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    background: "#FF2D8D",
  },

  /* Dot row */
  sensorDotsRow: {
    display: "flex",
    alignItems: "center",
    gap: "18px",
    flexWrap: "wrap",
  },
  sensorDotWrap: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "10px 12px",
    borderRadius: "999px",
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
  },
  sensorDot: {
    width: "12px",
    height: "12px",
    borderRadius: "50%",
    background: "#FF2D8D",
    boxShadow: "0 0 0 6px rgba(255,45,141,0.10)",
  },
  sensorDotLabel: {
    fontSize: "12px",
    fontWeight: 900,
    color: "var(--text-primary)",
    letterSpacing: "-0.01em",
  },
  jawLabelRow: {
    display: "inline-flex",
    alignItems: "center",
    gap: "6px",
  },
  jawDotIdle: {
    background: "rgba(255,45,141,0.40)",
    boxShadow: "0 0 0 6px rgba(255,45,141,0.06)",
  },
  jawDotActive: {
    background: "#FF2D8D",
    boxShadow: "0 0 0 8px rgba(255,45,141,0.14)",
  },

  /* Panels */
  rowPanels: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "18px",
    alignItems: "stretch",
  },

  cameraPanelSection: {
    display: "flex",
    flexDirection: "column",
  },

  panel: {
    borderRadius: "26px",
    border: "1px solid var(--border)",
    background: "var(--bg-secondary)",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
    minHeight: "520px",
  },
  panelHeader: {
    padding: "16px 18px",
    borderBottom: "1px solid var(--border)",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
    background: "linear-gradient(180deg, rgba(255,45,141,0.06), transparent)",
  },
  panelHeaderLeft: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  panelIcon: {
    width: "38px",
    height: "38px",
    borderRadius: "14px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "var(--bg-tertiary)",
    color: "var(--text-secondary)",
    border: "1px solid var(--border)",
    flexShrink: 0,
  },
  panelTitle: {
    fontWeight: 950,
    color: "var(--text-primary)",
    fontSize: "14px",
    letterSpacing: "-0.01em",
  },
  panelDescription: {
    color: "var(--text-secondary)",
    fontSize: "12px",
    marginTop: "2px",
  },
  pill: {
    fontSize: "12px",
    fontWeight: 900,
    color: "#FF2D8D",
    background: "rgba(255,45,141,0.08)",
    border: "1px solid rgba(255,45,141,0.2)",
    padding: "8px 10px",
    borderRadius: "999px",
    whiteSpace: "nowrap",
  },
  panelBody: {
    padding: "18px",
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    flex: 1,
  },

  placeholderOuter: {
    borderRadius: "18px",
    border: "1px dashed var(--border)",
    background:
      "radial-gradient(circle at 20% 20%, rgba(255,45,141,0.10), transparent 45%), radial-gradient(circle at 80% 20%, rgba(255,45,141,0.07), transparent 42%), linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.00))",
    height: "100%",
    minHeight: "420px",
    display: "flex",
    padding: "14px",
  },
  placeholderInner: {
    flex: 1,
    borderRadius: "14px",
    border: "1px solid var(--border)",
    background: "var(--bg-tertiary)",
    display: "grid",
    placeItems: "center",
  },
  placeholderText: {
    color: "var(--text-secondary)",
    fontSize: "12px",
    fontWeight: 800,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
  },
};

export default function Visualizations() {
  const nav = useNavigate();

  const fakeLevels = useMemo(
    () => ({
      left: { signal: 0.62 },
      middle: { signal: 0.74 },
      right: { signal: 0.58 },
      jawClench: false,
    }),
    []
  );

  return (
    <motion.div
      style={styles.container}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Header */}
      <motion.section style={styles.header} variants={itemVariants}>
        <div style={styles.headerLeft}>
          <div style={styles.headerTitles}>
            <h1 style={styles.title}>Visualizations</h1>
            <p
              style={{
                margin: 0,
                fontSize: "14px",
                fontWeight: 700,
                color: "#FF2D8D",
                letterSpacing: "-0.01em",
              }}
            >
              Ever wonder what goes on inside your brain?
            </p>
          </div>
        </div>

        <div style={styles.headerRight}>
          <StatusBadge label="Streaming Ready" />
        </div>
      </motion.section>

      {/* Sensor dots */}
      <motion.section style={styles.sensorDotsRow} variants={itemVariants}>
        <SensorDot label="Left" value={fakeLevels.left.signal} />
        <SensorDot label="Middle" value={fakeLevels.middle.signal} />
        <SensorDot label="Right" value={fakeLevels.right.signal} />
        <JawDot active={fakeLevels.jawClench} />
      </motion.section>

      {/* Heatmap + Waves */}
      <motion.section style={styles.rowPanels} variants={containerVariants}>
        <motion.div style={styles.panel} variants={itemVariants}>
          <PanelHeader
            icon={<Grid3X3 size={18} />}
            title="EEG Heatmap"
            description="Band power / channel intensity (placeholder)"
            rightSlot={<Pill label="8–16 Hz α" />}
          />
          <div style={styles.panelBody}>
            <PlaceholderCanvas label="Heatmap goes here" />
          </div>
        </motion.div>

        <motion.div style={styles.panel} variants={itemVariants}>
          <PanelHeader
            icon={<Waves size={18} />}
            title="Brain Wave Visualizer"
            description="Waveforms + bands (placeholder)"
            rightSlot={<Pill label="Raw / Filtered" />}
          />
          <div style={styles.panelBody}>
            <PlaceholderCanvas label="Wave visualizer goes here" />
          </div>
        </motion.div>
      </motion.section>

      {/* Camera preview placeholder */}
      <motion.section style={styles.cameraPanelSection} variants={itemVariants}>
        <motion.div style={styles.panel} variants={itemVariants}>
          <PanelHeader
            icon={<Waves size={18} />}
            title="Live Camera Preview"
            description="Eye tracking + ASL tracking (placeholder)"
          />
          <div style={styles.panelBody}>
            <PlaceholderCanvas label="Camera feed / CV overlay goes here" />
          </div>
        </motion.div>
      </motion.section>
    </motion.div>
  );
}

/* ----------------------------- UI Pieces ----------------------------- */

function StatusBadge({ label }: { label: string }) {
  return (
    <motion.div
      style={styles.statusBadge}
      animate={{ opacity: [1, 0.75, 1] }}
      transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
    >
      <motion.div
        style={styles.statusDot}
        animate={{ scale: [1, 1.2, 1], opacity: [1, 0.6, 1] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
      {label}
    </motion.div>
  );
}

function PanelHeader({
  icon,
  title,
  description,
  rightSlot,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  rightSlot?: React.ReactNode;
}) {
  return (
    <div style={styles.panelHeader}>
      <div style={styles.panelHeaderLeft}>
        <div style={styles.panelIcon}>{icon}</div>
        <div>
          <div style={styles.panelTitle}>{title}</div>
          <div style={styles.panelDescription}>{description}</div>
        </div>
      </div>
      <div>{rightSlot}</div>
    </div>
  );
}

function Pill({ label }: { label: string }) {
  return <div style={styles.pill}>{label}</div>;
}

function PlaceholderCanvas({ label }: { label: string }) {
  return (
    <div style={styles.placeholderOuter}>
      <div style={styles.placeholderInner}>
        <div style={styles.placeholderText}>{label}</div>
      </div>
    </div>
  );
}

function SensorDot({ label, value }: { label: string; value: number }) {
  const strength = Math.max(0, Math.min(1, value));
  const opacity = 0.35 + strength * 0.65;

  return (
    <div style={styles.sensorDotWrap}>
      <motion.div
        style={{ ...styles.sensorDot, opacity }}
        animate={{ scale: [1, 1.08, 1] }}
        transition={{ duration: 1.6, repeat: Infinity, ease: "easeInOut" }}
      />
      <div style={styles.sensorDotLabel}>{label}</div>
    </div>
  );
}

function JawDot({ active }: { active: boolean }) {
  return (
    <div style={styles.sensorDotWrap}>
      <motion.div
        style={{
          ...styles.sensorDot,
          ...(active ? styles.jawDotActive : styles.jawDotIdle),
        }}
        animate={active ? { scale: [1, 1.18, 1] } : { scale: [1, 1.05, 1] }}
        transition={{ duration: active ? 0.7 : 1.6, repeat: Infinity }}
      />
      <div style={styles.sensorDotLabel}>
        <span style={styles.jawLabelRow}>
          <Zap size={12} />
          EEG
        </span>
      </div>
    </div>
  );
}