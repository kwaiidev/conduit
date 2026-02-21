import React from "react";
import { resetOnboarding } from "../state/onboarding";

export default function Home() {
  return (
    <div style={{ maxWidth: 700 }}>
      <h1>Home</h1>
      <p>You're in the main app.</p>

      <button
        onClick={() => {
          resetOnboarding();
          location.href = "/onboarding";
        }}
        style={styles.secondary}
      >
        Reset onboarding
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  secondary: {
    marginTop: 16,
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    background: "#1b1b1b",
    color: "#eaeaea",
    cursor: "pointer",
  },
};