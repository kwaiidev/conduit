import React from "react";
import { resetOnboarding } from "../state/onboarding";

export default function Home() {
  return (
    <div className="page-container">
      <h1>Home</h1>
      <p>You're in the main app.</p>

      <button
        onClick={() => {
          resetOnboarding();
          location.href = "/onboarding";
        }}
        className="btn btn-secondary"
      >
        Reset onboarding
      </button>
    </div>
  );
}