import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "motion/react";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import { setCompletedOnboarding } from "../state/onboarding";
import { resetAllModalityPreferences } from "../state/modalityPreferences";
import { JawClenchTrainingRight } from "./onboarding/jawright";
import { JawClenchTrainingLeft } from "./onboarding/jawleft";
import { JawClenchTrainingSlack } from "./onboarding/jawslack";
import { CVCursorCalibrationCenter } from "./onboarding/cvcalibrate";
import { EEGConnectStep } from "./onboarding/eegconnect";
import { ServiceStartupStep } from "./onboarding/serviceStartup";
import { ASLVisualsCheckStep } from "./onboarding/aslVisualsCheck";
import conduitWordmark from "../assets/conduit.svg";
function OnboardingWelcomeHero() {
    return (_jsx("div", { className: "onboarding-welcome-hero", "aria-hidden": true, children: _jsx("img", { src: conduitWordmark, alt: "Conduit", className: "onboarding-welcome-logo" }) }));
}
function OnboardingCompletionCard({ isTransitioningHome }) {
    return (_jsxs(motion.div, { className: "onboarding-all-set-card", initial: { opacity: 0, y: 20, scale: 0.97 }, animate: { opacity: 1, y: 0, scale: 1 }, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] }, children: [_jsx("div", { className: "onboarding-all-set-ring", "aria-hidden": true }), _jsx(motion.div, { className: "onboarding-all-set-label", initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.35, delay: 0.1 }, children: "Setup complete" }), _jsx(motion.h3, { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.35, delay: 0.2 }, children: "You are ready to drive Conduit" }), _jsx(motion.p, { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.35, delay: 0.3 }, children: "Core calibration and modality checks are complete. Enter Home to begin live control." }), _jsxs(motion.div, { className: "onboarding-all-set-tags", initial: { opacity: 0, y: 8 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.35, delay: 0.4 }, children: [_jsx("span", { children: "Eye Cursor" }), _jsx("span", { children: "EEG" }), _jsx("span", { children: "Voice" }), _jsx("span", { children: "ASL" })] }), _jsx(motion.div, { className: "onboarding-all-set-progress", animate: { scaleX: isTransitioningHome ? 1 : 0 }, transition: { duration: 0.62, ease: [0.16, 1, 0.3, 1] } })] }));
}
export default function Onboarding() {
    const nav = useNavigate();
    const location = useLocation();
    const { isDark, toggleTheme } = useTheme();
    const startStep = location.state?.startStep ?? 0;
    const [step, setStep] = useState(startStep);
    const [animationKey, setAnimationKey] = useState(0);
    const [isCvCalibrationActive, setIsCvCalibrationActive] = useState(false);
    const [isTransitioningHome, setIsTransitioningHome] = useState(false);
    const transitionTimerRef = useRef(null);
    const isCvCalibrationStep = step === 1;
    const isFinalStep = step === steps.length - 1;
    const isWelcomeStep = step === 0;
    const calibrationLocked = isCvCalibrationStep && isCvCalibrationActive;
    const navigationLocked = calibrationLocked || isTransitioningHome;
    const currentStep = steps[step];
    const stepContent = isCvCalibrationStep ? (_jsx(CVCursorCalibrationCenter, { onCalibrationStateChange: setIsCvCalibrationActive })) : isFinalStep ? (_jsx(OnboardingCompletionCard, { isTransitioningHome: isTransitioningHome })) : (currentStep.content);
    const stepContentClassName = `step-content${isFinalStep ? " step-content-final" : ""}`;
    useEffect(() => {
        if (startStep === 0) {
            // Fresh onboarding should start from a clean modality-selection state.
            resetAllModalityPreferences(false);
        }
    }, [startStep]);
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
        }
        else {
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
    return (_jsxs("div", { className: `onboarding-container${calibrationLocked ? " onboarding-calibration-locked" : ""}${isTransitioningHome ? " onboarding-transitioning-home" : ""}`, children: [_jsx("button", { type: "button", onClick: toggleTheme, className: "onboarding-theme-toggle", title: isDark ? "Switch to light mode" : "Switch to dark mode", "aria-label": "Toggle theme", disabled: navigationLocked, children: isDark ? _jsx(Sun, { size: 20 }) : _jsx(Moon, { size: 20 }) }), _jsx("div", { className: "onboarding-content", children: _jsx("div", { className: "onboarding-inner", children: isWelcomeStep ? (_jsxs(_Fragment, { children: [_jsx(motion.h1, { className: "onboarding-welcome-title", initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] }, children: "Welcome to Conduit" }), _jsx(motion.p, { initial: { opacity: 0, y: 12 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.3, delay: 0.15 }, children: currentStep.description }), _jsx(motion.div, { className: "onboarding-welcome-hero-wrap", initial: { opacity: 0, scale: 0.98 }, animate: { opacity: 1, scale: 1 }, transition: { duration: 0.3, delay: 0.22 }, children: stepContent }), _jsx(motion.div, { className: "button-group", initial: { opacity: 0 }, animate: { opacity: 1 }, transition: { duration: 0.2, delay: 0.35 }, children: _jsx("button", { className: "btn btn-primary", onClick: handleNext, disabled: navigationLocked, children: "Next" }) })] })) : (_jsxs(_Fragment, { children: [_jsx("h1", { children: currentStep.title }), _jsx("p", { children: currentStep.description }), _jsx("div", { className: stepContentClassName, children: stepContent }), _jsxs("div", { className: "button-group", children: [step > 0 && (_jsx("button", { className: "btn btn-secondary", onClick: handleBack, disabled: navigationLocked, children: "Back" })), _jsx("button", { className: "btn btn-primary", onClick: handleNext, disabled: navigationLocked, children: isTransitioningHome ? "Entering Home..." : step < steps.length - 1 ? "Next" : "Enter Home" })] })] })) }, animationKey) })] }));
}
const steps = [
    {
        title: "Welcome to Conduit",
        description: "A multimodal accessibility tool for hands-free computer control.",
        content: _jsx(OnboardingWelcomeHero, {}),
    },
    {
        title: "Calibrate CV Cursor",
        description: "You can do anything you put your mind to.",
        content: _jsx(CVCursorCalibrationCenter, {}),
    },
    {
        title: "Connect Your EEG Device",
        description: "Launch the EEG service terminal and verify your headset stream is available.",
        content: _jsx(EEGConnectStep, {}),
    },
    {
        title: "Start Voice Service",
        description: "Launch the voice backend terminal so voice input is available in Home/Overlay.",
        content: _jsx(ServiceStartupStep, { service: "voice" }),
    },
    {
        title: "Start ASL + Validate Visuals",
        description: "Start ASL backend, enable ASL detection/session, and verify ASL + eye-gaze CV are both online.",
        content: _jsx(ASLVisualsCheckStep, {}),
    },
    {
        title: "Calibrate Baseline",
        description: "Press Start to launch EEG training and hold a neutral jaw baseline.",
        content: _jsx(JawClenchTrainingSlack, {}),
    },
    {
        title: "Train Left Jaw Clench",
        description: "Press Start to keep EEG active, then hold left jaw clench for 3 seconds.",
        content: _jsx(JawClenchTrainingLeft, {}),
    },
    {
        title: "Train Right Jaw Clench",
        description: "Press Start to keep EEG active, then hold right jaw clench for 3 seconds.",
        content: _jsx(JawClenchTrainingRight, {}),
    },
    {
        title: "All Set!",
        description: "You're ready to control Conduit with your brainwaves.",
        content: null,
    },
];
