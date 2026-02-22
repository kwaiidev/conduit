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
import { CVCursorCalibrationCenter } from "./onboarding/cvcalibrate";
import { EEGConnectStep } from "./onboarding/eegconnect";
import { ServiceStartupStep } from "./onboarding/serviceStartup";
import { ASLVisualsCheckStep } from "./onboarding/aslVisualsCheck";

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
    title: "Calibrate CV Cursor",
    description: "You can do anything you put your mind to.",
    content: <CVCursorCalibrationCenter />,
  },
  {
    title: "Connect Your EEG Device",
    description: "Launch the EEG service terminal and verify your headset stream is available.",
    content: <EEGConnectStep />,
  },
  {
    title: "Start Voice Service",
    description: "Launch the voice backend terminal so voice input is available in Home/Overlay.",
    content: <ServiceStartupStep service="voice" />,
  },
  {
    title: "Start ASL + Validate Visuals",
    description: "Start ASL backend, enable ASL detection/session, and verify ASL + eye-gaze CV are both online.",
    content: <ASLVisualsCheckStep />,
  },
  {
    title: "Calibrate Baseline",
    description: "Press Start to launch EEG training and hold a neutral jaw baseline.",
    content: <JawClenchTrainingSlack />,
  },
  {
    title: "Train Left Jaw Clench",
    description: "Press Start to keep EEG active, then hold left jaw clench for 3 seconds.",
    content: <JawClenchTrainingLeft />,
  },
  {
    title: "Train Right Jaw Clench",
    description: "Press Start to keep EEG active, then hold right jaw clench for 3 seconds.",
    content: <JawClenchTrainingRight />,

  },
  {
    title: "Train Middle Jaw Clench",
    description: "Press Start to keep EEG active, then hold middle jaw clench for 3 seconds.",
    content: <JawClenchTrainingMiddle />,

  },
  {
    title: "All Set!",
    description: "You're ready to control Conduit with your brainwaves.",
    content: <div className="placeholder-content">✅ Setup complete</div>,
  },
];
