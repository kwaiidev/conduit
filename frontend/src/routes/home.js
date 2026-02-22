import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { enableASL, disableASL, getASLReady } from "../lib/aslcv";
import { enableVoice, disableVoice, getVoiceReady } from "../lib/voicetts";
import { enableEEG, disableEEG } from "../lib/eeg";
import { enableCvCursorControl, disableCvCursorControl, getCvReady } from "../lib/cv";
import { CVCursorCalibrationCenter } from "./onboarding/cvcalibrate";
import { motion } from "motion/react";
import { MousePointer2, Zap, Type, Activity, Eye, Mic, ArrowRight } from "lucide-react";
const CV_POINTER_PREFERENCE_KEY = "conduit-modality-intent-cv-pointer";
const SIGN_TEXT_PREFERENCE_KEY = "conduit-modality-intent-sign-text";
function getModalityPreference(key) {
    if (typeof window === "undefined" || !window.localStorage) {
        return null;
    }
    const value = localStorage.getItem(key);
    if (value === null) {
        return null;
    }
    return value === "1";
}
function setModalityPreference(key, enabled) {
    if (typeof window === "undefined" || !window.localStorage) {
        return;
    }
    localStorage.setItem(key, enabled ? "1" : "0");
}
function shouldActivateByIntent(featureId, backendReady) {
    if (featureId === "cv-pointer") {
        const preference = getModalityPreference(CV_POINTER_PREFERENCE_KEY);
        return preference === null ? false : preference && backendReady;
    }
    if (featureId === "sign-text") {
        const preference = getModalityPreference(SIGN_TEXT_PREFERENCE_KEY);
        return preference === null ? false : preference && backendReady;
    }
    return backendReady;
}
const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.06, delayChildren: 0.1 },
    },
};
const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { type: "spring", stiffness: 300, damping: 24 },
    },
};
export default function Home() {
    const nav = useNavigate();
    const [activeModes, setActiveModes] = useState([]);
    const [showCvCenterCalibration, setShowCvCenterCalibration] = useState(false);
    // Sync voice and sign toggles with real API state on mount
    useEffect(() => {
        const sync = (id, ready) => {
            setActiveModes((prev) => shouldActivateByIntent(id, ready)
                ? prev.includes(id) ? prev : [...prev, id]
                : prev.filter((m) => m !== id));
        };
        getCvReady().then((ready) => sync('cv-pointer', ready));
        getVoiceReady().then((ready) => sync('voice-text', ready));
        getASLReady().then((ready) => sync('sign-text', ready));
    }, []);
    useEffect(() => {
        if (showCvCenterCalibration) {
            document.documentElement.setAttribute("data-snap-modal-open", "true");
        }
        else {
            document.documentElement.removeAttribute("data-snap-modal-open");
        }
        return () => {
            document.documentElement.removeAttribute("data-snap-modal-open");
        };
    }, [showCvCenterCalibration]);
    const closeCvCalibration = React.useCallback(async () => {
        setShowCvCenterCalibration(false);
        try {
            await enableCvCursorControl();
        }
        catch (error) {
            console.error("CV enable after closing calibration modal failed:", error);
            setActiveModes((prev) => prev.filter((id) => id !== 'cv-pointer'));
            setModalityPreference(CV_POINTER_PREFERENCE_KEY, false);
            await disableCvCursorControl();
        }
    }, []);
    const handleCvCalibrationLocked = React.useCallback(() => {
        void (async () => {
            try {
                await enableCvCursorControl();
                setShowCvCenterCalibration(false);
            }
            catch (error) {
                console.error("CV re-enable after calibration failed:", error);
                setShowCvCenterCalibration(false);
                setActiveModes((prev) => prev.filter((id) => id !== 'cv-pointer'));
                setModalityPreference(CV_POINTER_PREFERENCE_KEY, false);
                await disableCvCursorControl();
            }
        })();
    }, []);
    const featureGroups = [
        {
            title: "Pointer Control",
            features: [
                { id: 'cv-pointer', icon: _jsx(MousePointer2, { size: 20 }), name: "CV Cursor", description: "Eye movement control" },
            ]
        },
        {
            title: "Selection",
            features: [
                { id: 'eeg-select', icon: _jsx(Zap, { size: 20 }), name: "EEG Jaw Clench", description: "Neural signal selection", training: true },
            ]
        },
        {
            title: "Text Input",
            features: [
                { id: 'voice-text', icon: _jsx(Mic, { size: 20 }), name: "Voice-to-Text", description: "Natural dictation engine" },
                { id: 'sign-text', icon: _jsx(Mic, { size: 20 }), name: "Sign-to-Text", description: "Sign to text conversion" }
            ]
        }
    ];
    const handleRetrainEEG = () => {
        nav("/onboarding", { state: { startStep: 1 } });
    };
    const quickStartSteps = [
        "Use the toggles above to enable the modalities you want to control right now.",
        "Keep your webcam view clear and your EEG headset steady for stable confidence scores.",
        "Use voice for explicit commands, gaze for continuous cursor movement, EEG for selection, and ASL for text input.",
        "If control quality drops, rerun training with Train EEG Signals and recalibrate from onboarding.",
        "Open Visuals to monitor latency, stability, and active modality health in real time.",
    ];
    const modalityGuide = [
        {
            icon: _jsx(Eye, { size: 16 }),
            name: "Eye Tracking CV",
            hint: "Look where you want the pointer to move. The One Euro filter suppresses jitter while keeping fast movement responsive.",
        },
        {
            icon: _jsx(Activity, { size: 16 }),
            name: "EEG Jaw Clench",
            hint: "EEG windows are converted into intents. Use this for selection and key actions when physical input is limited.",
        },
        {
            icon: _jsx(Mic, { size: 16 }),
            name: "Voice Command + Speech",
            hint: "Short push-to-talk utterances are parsed into canonical actions. Use voice for fast explicit commands and dictation.",
        },
        {
            icon: _jsx(Type, { size: 16 }),
            name: "ASL Sign CV",
            hint: "Hand-sign predictions are smoothed across frames to reduce false letters before text events are emitted.",
        },
    ];
    const architectureGuidance = [
        "Each modality runs as an independent service and publishes canonical events to a shared event bus.",
        "Fusion arbitrates by confidence, timestamp recency, and intent priority to pick one executable event per frame.",
        "Stale events older than 150 ms are dropped to keep interaction responsive and predictable.",
        "When confidence or latency is unsafe, the pipeline emits noop fallback events instead of blocking the system.",
    ];
    const toggleFeature = async (featureId) => {
        const isCurrentlyActive = activeModes.includes(featureId);
        const next = !isCurrentlyActive;
        // Optimistically update UI
        setActiveModes((prev) => next ? [...prev, featureId] : prev.filter((id) => id !== featureId));
        if (featureId === 'cv-pointer') {
            setModalityPreference(CV_POINTER_PREFERENCE_KEY, next);
        }
        if (featureId === 'sign-text') {
            setModalityPreference(SIGN_TEXT_PREFERENCE_KEY, next);
        }
        if (featureId === 'cv-pointer') {
            if (next) {
                setShowCvCenterCalibration(true);
            }
            else {
                setShowCvCenterCalibration(false);
                try {
                    await disableCvCursorControl();
                }
                catch (error) {
                    console.error("CV disable failed:", error);
                    setActiveModes((prev) => isCurrentlyActive ? [...prev, featureId] : prev.filter((id) => id !== featureId));
                    setModalityPreference(CV_POINTER_PREFERENCE_KEY, isCurrentlyActive);
                }
            }
            return;
        }
        if (featureId === 'eeg-select') {
            try {
                if (next) {
                    await enableEEG();
                }
                else {
                    await disableEEG();
                }
            }
            catch (e) {
                console.error("EEG toggle failed:", e);
                // Revert on failure
                setActiveModes((prev) => isCurrentlyActive ? [...prev, featureId] : prev.filter((id) => id !== featureId));
            }
        }
        if (featureId === 'sign-text' || featureId === 'voice-text') {
            const enable = featureId === 'sign-text' ? enableASL : enableVoice;
            const disable = featureId === 'sign-text' ? disableASL : disableVoice;
            try {
                if (next) {
                    await enable();
                }
                else {
                    await disable();
                    if (featureId === 'sign-text') {
                        setModalityPreference(SIGN_TEXT_PREFERENCE_KEY, false);
                    }
                }
            }
            catch (e) {
                console.error(`${featureId} toggle failed:`, e);
                setActiveModes((prev) => isCurrentlyActive ? [...prev, featureId] : prev.filter((id) => id !== featureId));
                if (featureId === 'sign-text') {
                    setModalityPreference(SIGN_TEXT_PREFERENCE_KEY, isCurrentlyActive);
                }
            }
        }
    };
    return (_jsxs(motion.div, { style: styles.container, variants: containerVariants, initial: "hidden", animate: "visible", children: [_jsxs(motion.section, { style: {
                    ...styles.header,
                    ...(showCvCenterCalibration ? styles.blockedInteractiveLayer : {}),
                }, className: "home-header", variants: itemVariants, "data-snap-ignore": showCvCenterCalibration ? "true" : undefined, "aria-hidden": showCvCenterCalibration ? true : undefined, children: [_jsxs("div", { style: styles.headerContent, children: [_jsxs(motion.div, { style: styles.statusBadge, animate: { opacity: [1, 0.7, 1] }, transition: { duration: 2, repeat: Infinity, ease: "easeInOut" }, children: [_jsx(motion.div, { style: styles.statusDot, animate: { scale: [1, 1.2, 1], opacity: [1, 0.6, 1] }, transition: { duration: 1.5, repeat: Infinity, ease: "easeInOut" } }), "System Operational"] }), _jsxs("h1", { style: styles.title, className: "home-title", children: ["Welcome back, ", _jsx("span", { style: styles.titleAccent, children: "User" })] })] }), _jsxs(motion.button, { onClick: handleRetrainEEG, style: styles.retrainButton, className: "home-retrain-button", whileHover: { scale: 1.05 }, whileTap: { scale: 0.98 }, children: [_jsx(Zap, { size: 18 }), "Train EEG Signals", _jsx(ArrowRight, { size: 16 })] })] }), _jsx(motion.div, { style: {
                    ...styles.grid,
                    ...(showCvCenterCalibration ? styles.blockedInteractiveLayer : {}),
                }, variants: containerVariants, "data-snap-ignore": showCvCenterCalibration ? "true" : undefined, "aria-hidden": showCvCenterCalibration ? true : undefined, children: featureGroups.map((group) => (_jsxs(motion.div, { style: styles.group, variants: itemVariants, children: [_jsx("h2", { style: styles.groupTitle, children: group.title }), _jsx("div", { style: styles.featuresList, children: group.features.map((feature) => {
                                const isActive = activeModes.includes(feature.id);
                                return (_jsxs(motion.div, { style: {
                                        ...styles.featureCard,
                                        ...(isActive ? styles.featureCardActive : {}),
                                    }, className: isActive ? "home-feature-card home-feature-card-active" : "home-feature-card", onClick: () => toggleFeature(feature.id), variants: itemVariants, whileHover: { y: -4, transition: { duration: 0.2 } }, whileTap: { scale: 0.99 }, children: [_jsxs("div", { style: styles.featureContent, children: [_jsx(motion.div, { style: {
                                                        ...styles.featureIcon,
                                                        ...(isActive ? styles.featureIconActive : {}),
                                                    }, animate: isActive ? { scale: [1, 1.05, 1] } : {}, transition: { duration: 0.3 }, children: feature.icon }), _jsxs("div", { style: styles.featureText, children: [_jsx("span", { style: styles.featureName, children: feature.name }), _jsx("p", { style: styles.featureDescription, children: feature.description })] })] }), _jsx(Toggle, { active: isActive })] }, feature.id));
                            }) })] }, group.title))) }), _jsxs(motion.section, { style: {
                    ...styles.instructionsPanel,
                    ...(showCvCenterCalibration ? styles.blockedInteractiveLayer : {}),
                }, variants: itemVariants, "data-snap-ignore": showCvCenterCalibration ? "true" : undefined, "aria-hidden": showCvCenterCalibration ? true : undefined, children: [_jsxs("div", { style: styles.instructionsHeader, children: [_jsx("span", { style: styles.instructionsEyebrow, children: "Start Here" }), _jsx("h2", { style: styles.instructionsTitle, children: "Why Conduit exists and how to use it" }), _jsx("p", { style: styles.instructionsLead, children: "Conduit is built to give full computer access to people who cannot rely on conventional mouse and keyboard input. It translates gaze, EEG, voice, and sign signals into a shared canonical control event format so the system can safely fuse them in real time." })] }), _jsxs("div", { style: styles.instructionsGrid, children: [_jsxs("article", { style: styles.instructionsCard, children: [_jsx("h3", { style: styles.instructionsCardTitle, children: "Quick Start Flow" }), _jsx("ol", { style: styles.instructionsList, children: quickStartSteps.map((step) => (_jsx("li", { style: styles.instructionsListItem, children: step }, step))) })] }), _jsxs("article", { style: styles.instructionsCard, children: [_jsx("h3", { style: styles.instructionsCardTitle, children: "Modality Guide" }), _jsx("div", { style: styles.modalityList, children: modalityGuide.map((item) => (_jsxs("div", { style: styles.modalityItem, children: [_jsx("div", { style: styles.modalityIcon, children: item.icon }), _jsxs("div", { style: styles.modalityContent, children: [_jsx("span", { style: styles.modalityName, children: item.name }), _jsx("p", { style: styles.modalityHint, children: item.hint })] })] }, item.name))) })] }), _jsxs("article", { style: styles.instructionsCard, children: [_jsx("h3", { style: styles.instructionsCardTitle, children: "Data + Safety Contract" }), _jsx("ul", { style: styles.instructionsList, children: architectureGuidance.map((point) => (_jsx("li", { style: styles.instructionsListItem, children: point }, point))) })] })] })] }), showCvCenterCalibration ? (_jsx("div", { style: styles.cvModalBackdrop, role: "dialog", "aria-modal": "true", "data-snap-modal-root": "true", children: _jsxs("div", { style: styles.cvModalCard, children: [_jsxs("div", { style: styles.cvModalHeader, children: [_jsxs("div", { style: styles.cvModalHeaderText, children: [_jsx("h3", { style: styles.cvModalTitle, children: "Center Eye Alignment" }), _jsx("p", { style: styles.cvModalLead, children: "Recalibrate center gaze before re-enabling live cursor control." })] }), _jsx("button", { type: "button", onClick: () => void closeCvCalibration(), style: styles.cvModalCloseButton, children: "Close" })] }), _jsx(CVCursorCalibrationCenter, { autoStart: true, onCenterLocked: handleCvCalibrationLocked })] }) })) : null] }));
}
const Toggle = ({ active }) => {
    return (_jsx("div", { style: {
            ...styles.toggle,
            ...(active ? styles.toggleActive : {}),
        }, children: _jsx(motion.div, { style: {
                ...styles.toggleThumb,
                ...(active ? styles.toggleThumbActive : {}),
            }, layout: true, transition: { type: "spring", stiffness: 400, damping: 30 } }) }));
};
const styles = {
    container: {
        display: 'flex',
        flexDirection: 'column',
        gap: '48px',
        paddingBottom: '96px',
        padding: 'clamp(20px, 4vw, 40px)',
        maxWidth: '1400px',
        margin: '0 auto',
        width: '100%',
        background: 'var(--shell-content-bg)',
        minHeight: '100%',
    },
    header: {
        display: 'flex',
        flexDirection: 'column',
        gap: '32px', // gap-8
        alignItems: 'flex-start',
    },
    headerContent: {
        display: 'flex',
        flexDirection: 'column',
        gap: '8px', // gap-2
    },
    statusBadge: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px', // gap-2
        color: '#FF2D8D',
        fontWeight: 700,
        letterSpacing: '0.1em',
        fontSize: '12px', // text-xs
        textTransform: 'uppercase',
    },
    statusDot: {
        width: '6px', // w-1.5
        height: '6px', // h-1.5
        borderRadius: '50%',
        background: '#FF2D8D',
        animation: 'pulse 2s ease-in-out infinite',
    },
    title: {
        fontSize: '48px',
        fontWeight: 700,
        letterSpacing: '-0.02em',
        color: 'var(--text-primary)',
        margin: 0,
        lineHeight: 1.2,
    },
    titleAccent: {
        color: '#FF2D8D',
    },
    retrainButton: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px', // gap-2
        background: '#FF2D8D',
        color: 'white',
        padding: '14px 24px', // px-6 py-3.5
        borderRadius: '16px', // rounded-2xl
        fontWeight: 700,
        fontSize: '16px',
        border: 'none',
        cursor: 'pointer',
        boxShadow: '0 10px 25px rgba(255, 45, 141, 0.2)', // shadow-lg shadow-[#FF2D8D]/20
        transition: 'all 0.2s ease',
        alignSelf: 'flex-start',
    },
    instructionsPanel: {
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        padding: 'clamp(12px, 2.2vw, 18px)',
        borderRadius: '22px',
        border: '1px solid var(--border)',
        background: 'linear-gradient(145deg, var(--bg-secondary), var(--bg-primary))',
        maxWidth: '1160px',
        width: '100%',
        alignSelf: 'center',
    },
    instructionsHeader: {
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        maxWidth: '720px',
    },
    instructionsEyebrow: {
        fontSize: '11px',
        fontWeight: 700,
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        color: '#FF2D8D',
    },
    instructionsTitle: {
        margin: 0,
        fontSize: 'clamp(18px, 2.3vw, 24px)',
        fontWeight: 700,
        color: 'var(--text-primary)',
        lineHeight: 1.2,
    },
    instructionsLead: {
        margin: 0,
        color: 'var(--text-secondary)',
        fontSize: '13px',
        lineHeight: 1.55,
        maxWidth: '700px',
    },
    instructionsGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
        gap: '12px',
    },
    instructionsCard: {
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        padding: '14px',
        borderRadius: '16px',
        border: '1px solid var(--border)',
        background: 'var(--bg-primary)',
        minHeight: '100%',
    },
    instructionsCardTitle: {
        margin: 0,
        fontSize: '13px',
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        color: 'var(--text-primary)',
        fontWeight: 700,
    },
    instructionsList: {
        margin: 0,
        paddingLeft: '16px',
        display: 'flex',
        flexDirection: 'column',
        gap: '7px',
    },
    instructionsListItem: {
        color: 'var(--text-secondary)',
        fontSize: '12px',
        lineHeight: 1.45,
    },
    modalityList: {
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
    },
    modalityItem: {
        display: 'flex',
        alignItems: 'flex-start',
        gap: '8px',
    },
    modalityIcon: {
        width: '24px',
        height: '24px',
        borderRadius: '8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-secondary)',
        color: '#FF2D8D',
        flexShrink: 0,
    },
    modalityContent: {
        display: 'flex',
        flexDirection: 'column',
        gap: '2px',
    },
    modalityName: {
        fontSize: '13px',
        fontWeight: 700,
        color: 'var(--text-primary)',
        lineHeight: 1.3,
    },
    modalityHint: {
        margin: 0,
        color: 'var(--text-secondary)',
        fontSize: '12px',
        lineHeight: 1.4,
    },
    grid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '40px', // gap-10
    },
    group: {
        display: 'flex',
        flexDirection: 'column',
        gap: '24px', // gap-6
    },
    groupTitle: {
        fontSize: '14px',
        fontWeight: 700,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: 'var(--text-secondary)',
        margin: 0,
    },
    featuresList: {
        display: 'flex',
        flexDirection: 'column',
        gap: '16px', // gap-4
    },
    featureCard: {
        padding: '20px',
        borderRadius: '24px',
        border: '1px solid var(--border)',
        background: 'var(--bg-secondary)',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    featureCardActive: {
        background: 'rgba(255, 45, 141, 0.05)', // bg-[#FF2D8D]/5
        borderColor: 'rgba(255, 45, 141, 0.2)', // border-[#FF2D8D]/20
    },
    featureContent: {
        display: 'flex',
        alignItems: 'center',
        gap: '16px', // gap-4
        flex: 1,
    },
    featureIcon: {
        width: '48px',
        height: '48px',
        borderRadius: '16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'var(--bg-tertiary)',
        color: 'var(--text-secondary)',
        transition: 'all 0.2s ease',
        flexShrink: 0,
    },
    featureIconActive: {
        background: '#FF2D8D',
        color: 'white',
    },
    featureText: {
        flex: 1,
    },
    featureName: {
        display: 'block',
        fontWeight: 700,
        color: 'var(--text-primary)',
        fontSize: '16px',
        marginBottom: '4px',
    },
    featureDescription: {
        fontSize: '12px',
        color: 'var(--text-secondary)',
        margin: 0,
    },
    toggle: {
        width: '44px',
        height: '24px',
        borderRadius: '12px',
        background: 'var(--bg-tertiary)',
        position: 'relative',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        flexShrink: 0,
    },
    toggleActive: {
        background: '#FF2D8D',
    },
    toggleThumb: {
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        background: 'white',
        position: 'absolute',
        top: '2px',
        left: '2px',
        transition: 'all 0.2s ease',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    },
    toggleThumbActive: {
        left: '22px',
    },
    cvModalBackdrop: {
        position: 'fixed',
        inset: 0,
        background: 'rgba(8, 12, 18, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        zIndex: 1200,
        backdropFilter: 'blur(4px)',
        WebkitBackdropFilter: 'blur(4px)',
    },
    cvModalCard: {
        width: 'min(760px, 100%)',
        maxHeight: '92vh',
        overflowY: 'auto',
        borderRadius: '20px',
        border: '1px solid var(--border)',
        background: 'var(--bg-primary)',
        boxShadow: '0 30px 80px rgba(0, 0, 0, 0.34)',
        padding: '18px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
    },
    cvModalHeader: {
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: '12px',
    },
    cvModalHeaderText: {
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
    },
    cvModalTitle: {
        margin: 0,
        fontSize: '18px',
        fontWeight: 700,
        color: 'var(--text-primary)',
    },
    cvModalLead: {
        margin: 0,
        fontSize: '13px',
        color: 'var(--text-secondary)',
        lineHeight: 1.4,
    },
    cvModalCloseButton: {
        height: '36px',
        padding: '0 12px',
        borderRadius: '10px',
        border: '1px solid var(--border)',
        background: 'transparent',
        color: 'var(--text-secondary)',
        fontWeight: 700,
        cursor: 'pointer',
        flexShrink: 0,
    },
    blockedInteractiveLayer: {
        pointerEvents: 'none',
        userSelect: 'none',
    },
};
// Add pulse animation to CSS
if (typeof document !== 'undefined' && !document.getElementById('home-pulse-animation')) {
    const styleSheet = document.createElement('style');
    styleSheet.id = 'home-pulse-animation';
    styleSheet.textContent = `
    @keyframes pulse {
      0%, 100% {
        opacity: 1;
      }
      50% {
        opacity: 0.5;
      }
    }
  `;
    document.head.appendChild(styleSheet);
}
