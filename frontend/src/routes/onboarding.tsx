import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { setCompletedOnboarding } from "../state/onboarding";

export default function Onboarding() {
  const nav = useNavigate();
  const [step, setStep] = useState(0);
  const [animationKey, setAnimationKey] = useState(0);

  const handleNext = () => {
    if (step < steps.length - 1) {
      setStep(step + 1);
      setAnimationKey(prev => prev + 1);
    } else {
      setCompletedOnboarding();
      nav("/home", { replace: true });
    }
  };

  const handleBack = () => {
    if (step > 0) {
      setStep(step - 1);
      setAnimationKey(prev => prev + 1);
    }
  };

  const currentStep = steps[step];

  return (
    <div className="onboarding-container">
      <div className="onboarding-content">
        <div key={animationKey} className="onboarding-inner">
          <div className="step-indicator">
            Step {step + 1} of {steps.length}
          </div>
          
          <h1>{currentStep.title}</h1>
          <p>{currentStep.description}</p>

          <div className="step-content">
            {currentStep.content}
          </div>

          <div className="button-group">
            {step > 0 && (
              <button className="btn btn-secondary" onClick={handleBack}>
                Back
              </button>
            )}
            <button className="btn btn-primary" onClick={handleNext}>
              {step < steps.length - 1 ? "Next" : "Get Started"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const steps = [
  {
    title: "Welcome to Conduit",
    description: "Control your device using brainwave patterns detected through EEG.",
    content: <div className="placeholder-content">🧠 Welcome screen</div>,
  },
  {
    title: "Connect Your EEG Device",
    description: "Make sure your EEG headset is connected and powered on.",
    content: <div className="placeholder-content">📡 Device connection UI goes here</div>,
  },
  {
    title: "Calibrate Baseline",
    description: "Relax and clear your mind while we establish your baseline brainwave patterns.",
    content: <div className="placeholder-content">📊 Baseline calibration UI goes here</div>,
  },
  {
    title: "Train Control Patterns",
    description: "Follow the prompts to train different brainwave patterns for control.",
    content: <div className="placeholder-content">🎯 Pattern training UI goes here</div>,
  },
  {
    title: "All Set!",
    description: "You're ready to control Conduit with your brainwaves.",
    content: <div className="placeholder-content">✅ Setup complete</div>,
  },
];