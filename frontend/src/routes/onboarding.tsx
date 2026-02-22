import React, { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "motion/react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import { setCompletedOnboarding } from "../state/onboarding";
import { JawClenchTrainingRight } from "./onboarding/jawright";
import { JawClenchTrainingLeft } from "./onboarding/jawleft";
import { JawClenchTrainingSlack } from "./onboarding/jawslack";
import { CVCursorCalibrationCenter } from "./onboarding/cvcalibrate";
import { EEGConnectStep } from "./onboarding/eegconnect";
import { ServiceStartupStep } from "./onboarding/serviceStartup";
import { ASLVisualsCheckStep } from "./onboarding/aslVisualsCheck";
import conduitWordmark from "../assets/conduit.svg";

function OnboardingWelcomeHero() {
  return (
    <div className="onboarding-welcome-hero" aria-hidden>
      <img src={conduitWordmark} alt="Conduit" className="onboarding-welcome-logo" />
    </div>
  );
}

function OnboardingCompletionCard({ isTransitioningHome }: { isTransitioningHome: boolean }) {
  return (
    <motion.div
      className="onboarding-all-set-card"
      initial={{ opacity: 0, y: 20, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="onboarding-all-set-ring" aria-hidden />
      <motion.div
        className="onboarding-all-set-label"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.1 }}
      >
        Setup complete
      </motion.div>
      <motion.h3
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.2 }}
      >
        You are ready to drive Conduit
      </motion.h3>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.3 }}
      >
        Core calibration and modality checks are complete. Enter Home to begin live control.
      </motion.p>
      <motion.div
        className="onboarding-all-set-tags"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.4 }}
      >
        <span>Eye Cursor</span>
        <span>EEG</span>
        <span>Voice</span>
        <span>ASL</span>
      </motion.div>
      <motion.div
        className="onboarding-all-set-progress"
        animate={{ scaleX: isTransitioningHome ? 1 : 0 }}
        transition={{ duration: 0.62, ease: [0.16, 1, 0.3, 1] }}
      />
    </motion.div>
  );
}

export default function Onboarding() {
  const nav = useNavigate();
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();
  const startStep = (location.state as { startStep?: number })?.startStep ?? 0;
  const [step, setStep] = useState(startStep);
  const [animationKey, setAnimationKey] = useState(0);
  const [isCvCalibrationActive, setIsCvCalibrationActive] = useState(false);
  const [isTransitioningHome, setIsTransitioningHome] = useState(false);
  const transitionTimerRef = useRef<number | null>(null);

  const isCvCalibrationStep = step === 1;
  const isFinalStep = step === steps.length - 1;
  const isWelcomeStep = step === 0;
  const calibrationLocked = isCvCalibrationStep && isCvCalibrationActive;
  const navigationLocked = calibrationLocked || isTransitioningHome;
  const currentStep = steps[step];
  const stepContent = isCvCalibrationStep ? (
    <CVCursorCalibrationCenter onCalibrationStateChange={setIsCvCalibrationActive} />
  ) : isFinalStep ? (
    <OnboardingCompletionCard isTransitioningHome={isTransitioningHome} />
  ) : (
    currentStep.content
  );
  const stepContentClassName = `step-content${isFinalStep ? " step-content-final" : ""}`;

  useEffect(() => {
    return () => {
      if (transitionTimerRef.current !== null) {
        window.clearTimeout(transitionTimerRef.current);
      }
    };
  }, []);

  const handleNext = () => {
    if (navigationLocked) {
      return;
    }
    if (step < steps.length - 1) {
      setStep(step + 1);
      setAnimationKey(prev => prev + 1);
    } else {
      setIsTransitioningHome(true);
      transitionTimerRef.current = window.setTimeout(() => {
        setCompletedOnboarding();
        nav("/home", { replace: true });
      }, 720);
    }
  };

  const handleBack = () => {
    if (navigationLocked) {
      return;
    }
    if (step > 0) {
      setStep(step - 1);
      setAnimationKey(prev => prev + 1);
    }
  };

  return (
    <div
      className={`onboarding-container${calibrationLocked ? " onboarding-calibration-locked" : ""}${
        isTransitioningHome ? " onboarding-transitioning-home" : ""
      }`}
    >
      {/* Theme toggle - top right */}
      <button
        type="button"
        onClick={toggleTheme}
        className="onboarding-theme-toggle"
        title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        aria-label="Toggle theme"
        disabled={navigationLocked}
      >
        {isDark ? <Sun size={20} /> : <Moon size={20} />}
      </button>

      <div className="onboarding-content">
        <div key={animationKey} className="onboarding-inner">
          {isWelcomeStep ? (
            <>
              <motion.h1
                className="onboarding-welcome-title"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
              >
                Welcome to Conduit
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.15 }}
              >
                {currentStep.description}
              </motion.p>
              <motion.div
                className="onboarding-welcome-hero-wrap"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: 0.22 }}
              >
                {stepContent}
              </motion.div>
              <motion.div
                className="button-group"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2, delay: 0.35 }}
              >
                <button className="btn btn-primary" onClick={handleNext} disabled={navigationLocked}>
                  Next
                </button>
              </motion.div>
            </>
          ) : (
            <>
              <h1>{currentStep.title}</h1>
              <p>{currentStep.description}</p>
              <div className={stepContentClassName}>
                {stepContent}
              </div>
              <div className="button-group">
                {step > 0 && (
                  <button className="btn btn-secondary" onClick={handleBack} disabled={navigationLocked}>
                    Back
                  </button>
                )}
                <button className="btn btn-primary" onClick={handleNext} disabled={navigationLocked}>
                  {isTransitioningHome ? "Entering Home..." : step < steps.length - 1 ? "Next" : "Enter Home"}
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
    description: "A multimodal accessibility tool for hands-free computer control.",
    content: <OnboardingWelcomeHero />,
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
    title: "All Set!",
    description: "You're ready to control Conduit with your brainwaves.",
    content: null,
  },
];
