import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "motion/react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import { setCompletedOnboarding } from "../state/onboarding";
import { JawClenchTrainingMiddle } from "./onboarding/jawmiddle";
import { JawClenchTrainingRight } from "./onboarding/jawright";
import { JawClenchTrainingLeft } from "./onboarding/jawleft";
import { JawClenchTrainingSlack } from "./onboarding/jawslack";

export default function Onboarding() {
  const nav = useNavigate();
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();
  const startStep = (location.state as { startStep?: number })?.startStep ?? 0;
  const [step, setStep] = useState(startStep);
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
  const isWelcomeStep = step === 0;

  return (
    <div className="onboarding-container">
      {/* Theme toggle - top right */}
      <button
        type="button"
        onClick={toggleTheme}
        className="onboarding-theme-toggle"
        title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        aria-label="Toggle theme"
      >
        {isDark ? <Sun size={20} /> : <Moon size={20} />}
      </button>

      <div className="onboarding-content">
        <div key={animationKey} className="onboarding-inner">
          

          {isWelcomeStep ? (
            <>
              <motion.h1
                className="onboarding-welcome-title"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              >
                <motion.span
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.4, delay: 0.2 }}
                >
                  Welcome to{" "}
                </motion.span>
                <motion.span
                  className="onboarding-pink-accent"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.45 }}
                >
                  Conduit
                </motion.span>
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 }}
              >
                {currentStep.description}
              </motion.p>
              <motion.div
                className="step-content"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.75 }}
              >
                {currentStep.content}
              </motion.div>
              <motion.div
                className="button-group"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: 1 }}
              >
                <button className="btn btn-primary" onClick={handleNext}>
                  Next
                </button>
              </motion.div>
            </>
          ) : (
            <>
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
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const steps = [
  {
    title: "Welcome to Conduit",
    description: "You can do anything you put your mind to.",
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
    content: <JawClenchTrainingSlack />,
  },
  {
    title: "Train Left Jaw Clench",
    description: "Click or hover over 'Start' for 3 seconds to begin training.",
    content: <JawClenchTrainingLeft />,
  },
  {
    title: "Train Right Jaw Clench",
    description: "Click or hover over 'Start' for 3 seconds to begin training.",
    content: <JawClenchTrainingRight />,

  },
  {
    title: "Train Middle Jaw Clench",
    description: "Click or hover over 'Start' for 3 seconds to begin training.",
    content: <JawClenchTrainingMiddle />,

  },
  {
    title: "All Set!",
    description: "You're ready to control Conduit with your brainwaves.",
    content: <div className="placeholder-content">✅ Setup complete</div>,
  },
];