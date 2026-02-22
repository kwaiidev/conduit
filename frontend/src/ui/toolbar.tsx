import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Home, Settings, BarChart3, Minimize2, Maximize2, Minus, Square, X, Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";

export default function CompactToolbar() {
  const navigate = useNavigate();
  const { isDark, toggleTheme } = useTheme();
  const [isOverlay, setIsOverlay] = useState(false);
  const [isTogglingOverlay, setIsTogglingOverlay] = useState(false);

  useEffect(() => {
    // Get initial overlay mode
    window.electron?.getOverlayMode().then(setIsOverlay);

    // Listen for mode changes
    window.electron?.onOverlayModeChanged?.((mode) => {
      setIsOverlay(mode);
    });
  }, []);

  const toggleOverlay = async () => {
    if (isTogglingOverlay) {
      return;
    }
    if (typeof window.electron?.toggleOverlay !== "function") {
      console.error("Overlay toggle is unavailable in this runtime.");
      return;
    }

    setIsTogglingOverlay(true);
    try {
      const newMode = await window.electron.toggleOverlay();
      if (typeof newMode === "boolean") {
        setIsOverlay(newMode);
      }
    } catch (error) {
      console.error("Overlay toggle failed:", error);
    } finally {
      setIsTogglingOverlay(false);
    }
  };

  return (
    <div
      className="app-title-bar"
      style={{
        ...styles.toolbar,
        background: "var(--toolbar-bg)",
        borderBottomColor: "var(--toolbar-border)",
      }}
    >
      <div style={styles.left} className="app-drag-region">
        <span style={{ ...styles.logo, color: "var(--toolbar-text)" }}>Conduit</span>
      </div>

      <div style={styles.center} className="app-drag-region">
        <ToolbarButton
          active={false}
          onClick={() => navigate("/home")}
          label="Home"
          icon={Home}
        />
        <ToolbarButton
          active={false}
          onClick={() => navigate("/visuals")}
          label="Visuals"
          icon={BarChart3}
        />
      </div>

      <div style={styles.right} className="app-no-drag">
        {/* Window controls */}
        <button
          type="button"
          onClick={() => window.electron?.minimize()}
          style={styles.windowButton}
          title="Minimize"
        >
          <Minus size={14} />
        </button>
        <button
          type="button"
          onClick={() => window.electron?.maximize()}
          style={styles.windowButton}
          title="Maximize"
        >
          <Square size={14} />
        </button>
        <button
          type="button"
          onClick={() => window.electron?.close()}
          style={styles.closeButton}
          title="Close"
        >
          <X size={14} />
        </button>

        {/* Theme toggle */}
        <button
          type="button"
          onClick={toggleTheme}
          style={styles.windowButton}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* Overlay toggle */}
        <button
          type="button"
          onClick={toggleOverlay}
          style={{
            ...styles.overlayButton,
            ...(isTogglingOverlay ? styles.overlayButtonDisabled : {}),
          }}
          title={isTogglingOverlay ? "Switching overlay mode..." : isOverlay ? "Exit overlay mode" : "Enter overlay mode"}
          disabled={isTogglingOverlay}
        >
          {isOverlay ? <Maximize2 size={16} /> : <Minimize2 size={16} />}
          <span>{isTogglingOverlay ? "Switching..." : isOverlay ? "Expand" : "Overlay"}</span>
        </button>
      </div>
    </div>
  );
}

const ToolbarButton: React.FC<{
  active: boolean;
  onClick: () => void;
  label: string;
  icon: any;
  color?: string;
}> = ({ active, onClick, label, icon: Icon, color }) => (
  <button
    type="button"
    onClick={onClick}
    style={{
      ...styles.button,
      ...(active ? styles.buttonActive : {}),
      ...(color ? { color } : {}),
    }}
  >
    <Icon size={16} />
    <span>{label}</span>
  </button>
);

const styles: Record<string, React.CSSProperties> = {
  toolbar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 16px",
    height: 48,
    borderBottom: "1px solid var(--toolbar-border)",
  },
  left: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  logo: {
    fontSize: 14,
    fontWeight: 600,
  },
  center: {
    display: "flex",
    gap: 8,
    flex: 1,
    justifyContent: "center",
    // WebkitAppRegion: "no-drag" as any, // Removed, use attribute if needed
  },
  right: {
    display: "flex",
    gap: 8,
    // WebkitAppRegion: "no-drag" as any, // Removed, use attribute if needed
  },
  button: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 12px",
    background: "transparent",
    border: "1px solid transparent",
    borderRadius: 6,
    color: "var(--toolbar-muted)",
    cursor: "pointer",
    fontSize: 13,
    transition: "all 0.2s",
  },
  buttonActive: {
    background: "var(--bg-hover)",
    border: "1px solid var(--border)",
    color: "var(--toolbar-text)",
  },
  overlayButton: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 12px",
    background: "#3b82f6",
    border: "none",
    borderRadius: 6,
    color: "#ffffff",
    cursor: "pointer",
    fontSize: 13,
    fontWeight: 600,
    transition: "all 0.2s",
  },
  overlayButtonDisabled: {
    opacity: 0.7,
    cursor: "wait",
  },
  windowButton: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 32,
    height: 32,
    background: "transparent",
    border: "none",
    borderRadius: 6,
    color: "var(--toolbar-muted)",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  closeButton: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    width: 32,
    height: 32,
    background: "transparent",
    border: "none",
    borderRadius: 6,
    color: "var(--toolbar-muted)",
    cursor: "pointer",
    transition: "all 0.2s",
    marginRight: 8,
  },
};
