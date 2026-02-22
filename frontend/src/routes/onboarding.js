import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from "react";
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
    const startStep = location.state?.startStep ?? 0;
    const [step, setStep] = useState(startStep);
    const [animationKey, setAnimationKey] = useState(0);
    const handleNext = () => {
        if (step < steps.length - 1) {
            setStep(step + 1);
            setAnimationKey(prev => prev + 1);
        }
        else {
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
    return (_jsxs("div", { className: "onboarding-container", children: [_jsx("button", { type: "button", onClick: toggleTheme, className: "onboarding-theme-toggle", title: isDark ? "Switch to light mode" : "Switch to dark mode", "aria-label": "Toggle theme", children: isDark ? _jsx(Sun, { size: 20 }) : _jsx(Moon, { size: 20 }) }), _jsx("div", { className: "onboarding-content", children: _jsx("div", { className: "onboarding-inner", children: isWelcomeStep ? (_jsxs(_Fragment, { children: [_jsxs(motion.h1, { className: "onboarding-welcome-title", initial: { opacity: 0, y: 24 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }, children: [_jsxs(motion.span, { initial: { opacity: 0, x: -12 }, animate: { opacity: 1, x: 0 }, transition: { duration: 0.4, delay: 0.2 }, children: ["Welcome to", " "] }), _jsx(motion.span, { className: "onboarding-pink-accent", initial: { opacity: 0, scale: 0.9 }, animate: { opacity: 1, scale: 1 }, transition: { duration: 0.5, delay: 0.45 }, children: "Conduit" })] }), _jsx(motion.p, { initial: { opacity: 0, y: 16 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.5, delay: 0.6 }, children: currentStep.description }), _jsx(motion.div, { className: "step-content", initial: { opacity: 0, scale: 0.96 }, animate: { opacity: 1, scale: 1 }, transition: { duration: 0.5, delay: 0.75 }, children: currentStep.content }), _jsx(motion.div, { className: "button-group", initial: { opacity: 0 }, animate: { opacity: 1 }, transition: { duration: 0.4, delay: 1 }, children: _jsx("button", { className: "btn btn-primary", onClick: handleNext, children: "Next" }) })] })) : (_jsxs(_Fragment, { children: [_jsx("h1", { children: currentStep.title }), _jsx("p", { children: currentStep.description }), _jsx("div", { className: "step-content", children: currentStep.content }), _jsxs("div", { className: "button-group", children: [step > 0 && (_jsx("button", { className: "btn btn-secondary", onClick: handleBack, children: "Back" })), _jsx("button", { className: "btn btn-primary", onClick: handleNext, children: step < steps.length - 1 ? "Next" : "Get Started" })] })] })) }, animationKey) })] }));
}
const steps = [
    {
        title: "Welcome to Conduit",
        description: "You can do anything you put your mind to.",
        content: _jsx("div", { className: "placeholder-content", children: "\uD83E\uDDE0 Welcome screen" }),
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
        title: "Train Middle Jaw Clench",
        description: "Press Start to keep EEG active, then hold middle jaw clench for 3 seconds.",
        content: _jsx(JawClenchTrainingMiddle, {}),
    },
    {
        title: "All Set!",
        description: "You're ready to control Conduit with your brainwaves.",
        content: _jsx("div", { className: "placeholder-content", children: "\u2705 Setup complete" }),
    },
];
