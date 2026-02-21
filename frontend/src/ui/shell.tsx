import React from "react";
import Toolbar from "./toolbar";

export default function Shell({ children }: { children: React.ReactNode }) {
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
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
    background: "#0b0b0b",
    color: "#eaeaea",
  },
  content: { flex: 1, padding: 20, overflow: "auto" },
};