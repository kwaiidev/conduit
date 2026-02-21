import React, { useState, useEffect } from "react";
import Toolbar from "./toolbar";
import OverlayBar from "./overlay";

export default function Shell({ children }: { children: React.ReactNode }) {
  const [isOverlay, setIsOverlay] = useState(false);

  useEffect(() => {
    console.log("Shell mounted");
    window.electron?.getOverlayMode().then((mode) => {
      console.log("Initial overlay mode in Shell:", mode);
      setIsOverlay(mode);
    });
    window.electron?.onOverlayModeChanged((mode) => {
      console.log("Overlay mode changed in Shell:", mode);
      setIsOverlay(mode);
    });
  }, []);

  console.log("Shell rendering, isOverlay:", isOverlay);

  if (isOverlay) {
    return (
      <div style={styles.overlayRoot}>
        <OverlayBar />
      </div>
    );
  }

  return (
    <div style={styles.root}>
      <Toolbar />
      <div style={styles.content}>{children}</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    fontFamily: "'Poppins', system-ui, sans-serif",
    background: "var(--shell-bg)",
    color: "var(--text-primary)",
  },
  overlayRoot: {
    minHeight: "88px",
    display: "flex",
    flexDirection: "column",
    background: "transparent",
  },
  content: {
    flex: 1,
    padding: 20,
    overflow: "auto",
    background: "var(--shell-content-bg)",
  },
};