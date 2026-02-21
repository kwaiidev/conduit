import React from "react";
import { useNavigate } from "react-router-dom";
import { setCompletedOnboarding } from "../state/onboarding";

export default function Onboarding() {
  const nav = useNavigate();

  return (
    <div style={{ maxWidth: 700 }}>
      <h1>Welcome 👋</h1>
      <p>This is the onboarding screen. Put your setup steps here.</p>

      <button
        onClick={() => {
          setCompletedOnboarding();
          nav("/home", { replace: true });
        }}
        style={styles.primary}
      >
        Finish onboarding
      </button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  primary: {
    marginTop: 16,
    padding: "10px 14px",
    borderRadius: 12,
    border: "1px solid #2a2a2a",
    background: "#1b1b1b",
    color: "#eaeaea",
    cursor: "pointer",
  },
};