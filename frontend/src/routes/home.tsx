import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { enableASL, disableASL, getASLReady } from "../lib/aslcv";
import { enableVoice, disableVoice, getVoiceReady } from "../lib/voicetts";
import { enableEEG, disableEEG } from "../lib/eeg";
import { motion } from "motion/react";
import { 
  MousePointer2, 
  Zap, 
  Type, 
  Activity, 
  Eye, 
  Mic, 
  ArrowRight 
} from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.06, delayChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { type: "spring" as const, stiffness: 300, damping: 24 },
  },
};

export default function Home() {
  const nav = useNavigate();
  const [activeModes, setActiveModes] = useState(['cv-pointer', 'eeg-select']);

  // Sync voice and sign toggles with real API state on mount
  useEffect(() => {
    const sync = (id: string, ready: boolean) => {
      setActiveModes((prev) =>
        ready
          ? prev.includes(id) ? prev : [...prev, id]
          : prev.filter((m) => m !== id)
      );
    };
    getVoiceReady().then((ready) => sync('voice-text', ready));
    getASLReady().then((ready) => sync('sign-text', ready));
  }, []);

  const featureGroups = [
    {
      title: "Pointer Control",
      features: [
        { id: 'cv-pointer', icon: <MousePointer2 size={20} />, name: "CV Cursor", description: "Head/eye movement control" },
      ]
    },
    {
      title: "Selection",
      features: [
        { id: 'eeg-select', icon: <Zap size={20} />, name: "EEG Jaw Clench", description: "Neural signal selection", training: true },
      ]
    },
    {
      title: "Text Input",
      features: [
        { id: 'voice-text', icon: <Mic size={20} />, name: "Voice-to-Text", description: "Natural dictation engine" },
        { id: 'sign-text', icon: <Mic size={20} />, name: "Sign-to-Text", description: "Sign to text conversion"}
      ]
    }
  ];

  const handleRetrainEEG = () => {
    nav("/onboarding", { state: { startStep: 1 } });
  };

  const toggleFeature = async (featureId: string) => {
    const isCurrentlyActive = activeModes.includes(featureId);
    const next = !isCurrentlyActive;

    // Optimistically update UI
    setActiveModes((prev) =>
      next ? [...prev, featureId] : prev.filter((id) => id !== featureId)
    );

    if (featureId === 'sign-text' || featureId === 'voice-text') {
      const enable = featureId === 'sign-text' ? enableASL : enableVoice;
      const disable = featureId === 'sign-text' ? disableASL : disableVoice;
      try {
        if (next) {
          await enable();
        } else {
          await disable();
        }
      } catch (e) {
        console.error(`${featureId} toggle failed:`, e);
        setActiveModes((prev) =>
          isCurrentlyActive ? [...prev, featureId] : prev.filter((id) => id !== featureId)
        );
      }
    }
    if (featureId === 'eeg-select') {
      try {
        if (next) {
          await enableEEG();
        } else {
          await disableEEG();
        }
      } catch (e) {
        console.error("ASL toggle failed:", e);
        // Revert on failure
        setActiveModes((prev) =>
          isCurrentlyActive ? [...prev, featureId] : prev.filter((id) => id !== featureId)
        );
      }
    }
  };

  return (
    <motion.div
      style={styles.container}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* Welcome Header */}
      <motion.section
        style={styles.header}
        className="home-header"
        variants={itemVariants}
      >
        <div style={styles.headerContent}>
          <motion.div
            style={styles.statusBadge}
            animate={{ opacity: [1, 0.7, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <motion.div
              style={styles.statusDot}
              animate={{ scale: [1, 1.2, 1], opacity: [1, 0.6, 1] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
            System Operational
          </motion.div>
          <h1 style={styles.title} className="home-title">
            Welcome back, <span style={styles.titleAccent}>User</span>
          </h1>
        </div>
        
        <motion.button 
          onClick={handleRetrainEEG}
          style={styles.retrainButton}
          className="home-retrain-button"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.98 }}
        >
          <Zap size={18} />
          Train EEG Signals
          <ArrowRight size={16} />
        </motion.button>
      </motion.section>

      {/* Control Grid */}
      <motion.div style={styles.grid} variants={containerVariants}>
        {featureGroups.map((group, groupIndex) => (
          <motion.div
            key={group.title}
            style={styles.group}
            variants={itemVariants}
          >
            <h2 style={styles.groupTitle}>{group.title}</h2>
            <div style={styles.featuresList}>
              {group.features.map((feature, featureIndex) => {
                const isActive = activeModes.includes(feature.id);
                return (
                  <motion.div
                    key={feature.id}
                    style={{
                      ...styles.featureCard,
                      ...(isActive ? styles.featureCardActive : {}),
                    }}
                    className={isActive ? "home-feature-card home-feature-card-active" : "home-feature-card"}
                    onClick={() => toggleFeature(feature.id)}
                    variants={itemVariants}
                    whileHover={{ y: -4, transition: { duration: 0.2 } }}
                    whileTap={{ scale: 0.99 }}
                  >
                    <div style={styles.featureContent}>
                      <motion.div
                        style={{
                          ...styles.featureIcon,
                          ...(isActive ? styles.featureIconActive : {}),
                        }}
                        animate={isActive ? { scale: [1, 1.05, 1] } : {}}
                        transition={{ duration: 0.3 }}
                      >
                        {feature.icon}
                      </motion.div>
                      <div style={styles.featureText}>
                        <span style={styles.featureName}>{feature.name}</span>
                        <p style={styles.featureDescription}>{feature.description}</p>
                      </div>
                    </div>
                    <Toggle active={isActive} />
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}

const Toggle: React.FC<{ active: boolean }> = ({ active }) => {
  return (
    <div style={{
      ...styles.toggle,
      ...(active ? styles.toggleActive : {}),
    }}>
      <motion.div
        style={{
          ...styles.toggleThumb,
          ...(active ? styles.toggleThumbActive : {}),
        }}
        layout
        transition={{ type: "spring", stiffness: 400, damping: 30 }}
      />
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '48px',
    paddingBottom: '96px',
    padding: '40px',
    maxWidth: '1400px',
    margin: '0 auto',
    width: '100%',
    background: 'var(--shell-content-bg)',
    minHeight: '100%',
  },
  header: {
    display: 'flex',
    flexDirection: 'column',
    gap: '32px', // gap-8
    alignItems: 'flex-start',
  },
  headerContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px', // gap-2
  },
  statusBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px', // gap-2
    color: '#FF2D8D',
    fontWeight: 700,
    letterSpacing: '0.1em',
    fontSize: '12px', // text-xs
    textTransform: 'uppercase',
  },
  statusDot: {
    width: '6px', // w-1.5
    height: '6px', // h-1.5
    borderRadius: '50%',
    background: '#FF2D8D',
    animation: 'pulse 2s ease-in-out infinite',
  },
  title: {
    fontSize: '48px',
    fontWeight: 700,
    letterSpacing: '-0.02em',
    color: 'var(--text-primary)',
    margin: 0,
    lineHeight: 1.2,
  },
  titleAccent: {
    color: '#FF2D8D',
  },
  retrainButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px', // gap-2
    background: '#FF2D8D',
    color: 'white',
    padding: '14px 24px', // px-6 py-3.5
    borderRadius: '16px', // rounded-2xl
    fontWeight: 700,
    fontSize: '16px',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 10px 25px rgba(255, 45, 141, 0.2)', // shadow-lg shadow-[#FF2D8D]/20
    transition: 'all 0.2s ease',
    alignSelf: 'flex-start',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', // grid-cols-1 lg:grid-cols-3
    gap: '40px', // gap-10
  },
  group: {
    display: 'flex',
    flexDirection: 'column',
    gap: '24px', // gap-6
  },
  groupTitle: {
    fontSize: '14px',
    fontWeight: 700,
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'var(--text-secondary)',
    margin: 0,
  },
  featuresList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px', // gap-4
  },
  featureCard: {
    padding: '20px',
    borderRadius: '24px',
    border: '1px solid var(--border)',
    background: 'var(--bg-secondary)',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  featureCardActive: {
    background: 'rgba(255, 45, 141, 0.05)', // bg-[#FF2D8D]/5
    borderColor: 'rgba(255, 45, 141, 0.2)', // border-[#FF2D8D]/20
  },
  featureContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px', // gap-4
    flex: 1,
  },
  featureIcon: {
    width: '48px',
    height: '48px',
    borderRadius: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'var(--bg-tertiary)',
    color: 'var(--text-secondary)',
    transition: 'all 0.2s ease',
    flexShrink: 0,
  },
  featureIconActive: {
    background: '#FF2D8D',
    color: 'white',
  },
  featureText: {
    flex: 1,
  },
  featureName: {
    display: 'block',
    fontWeight: 700,
    color: 'var(--text-primary)',
    fontSize: '16px',
    marginBottom: '4px',
  },
  featureDescription: {
    fontSize: '12px',
    color: 'var(--text-secondary)',
    margin: 0,
  },
  toggle: {
    width: '44px',
    height: '24px',
    borderRadius: '12px',
    background: 'var(--bg-tertiary)',
    position: 'relative',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    flexShrink: 0,
  },
  toggleActive: {
    background: '#FF2D8D',
  },
  toggleThumb: {
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    background: 'white',
    position: 'absolute',
    top: '2px',
    left: '2px',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  toggleThumbActive: {
    left: '22px',
  },
};

// Add pulse animation to CSS
if (typeof document !== 'undefined' && !document.getElementById('home-pulse-animation')) {
  const styleSheet = document.createElement('style');
  styleSheet.id = 'home-pulse-animation';
  styleSheet.textContent = `
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }
  `;
  document.head.appendChild(styleSheet);
}
