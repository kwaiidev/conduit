import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { Zap, Waves, Grid3X3, Camera } from "lucide-react";
const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.06, delayChildren: 0.08 },
    },
};
const itemVariants = {
    hidden: { opacity: 0, y: 18 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { type: "spring", stiffness: 280, damping: 26 },
    },
};
const styles = {
    container: {
        display: "flex",
        flexDirection: "column",
        gap: "22px",
        padding: "40px",
        maxWidth: "1400px",
        margin: "0 auto",
        width: "100%",
        background: "var(--shell-content-bg)",
        minHeight: "100%",
        paddingBottom: "96px",
    },
    header: {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
        gap: "16px",
    },
    headerLeft: {
        display: "flex",
        alignItems: "flex-end",
        gap: "16px",
    },
    headerTitles: {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
    },
    headerRight: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
    },
    backButton: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        color: "var(--text-primary)",
        padding: "10px 12px",
        borderRadius: "14px",
        cursor: "pointer",
        fontWeight: 700,
    },
    title: {
        fontSize: "40px",
        fontWeight: 800,
        margin: 0,
        color: "var(--text-primary)",
        letterSpacing: "-0.02em",
        lineHeight: 1.1,
    },
    statusBadge: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        color: "#FF2D8D",
        fontWeight: 800,
        letterSpacing: "0.12em",
        fontSize: "12px",
        textTransform: "uppercase",
    },
    statusDot: {
        width: "7px",
        height: "7px",
        borderRadius: "50%",
        background: "#FF2D8D",
    },
    /* Dot row */
    sensorDotsRow: {
        display: "flex",
        alignItems: "center",
        gap: "18px",
        flexWrap: "wrap",
    },
    sensorDotWrap: {
        display: "flex",
        alignItems: "center",
        gap: "10px",
        padding: "10px 12px",
        borderRadius: "999px",
        border: "1px solid var(--border)",
        background: "var(--bg-secondary)",
    },
    sensorDot: {
        width: "12px",
        height: "12px",
        borderRadius: "50%",
        background: "#FF2D8D",
        boxShadow: "0 0 0 6px rgba(255,45,141,0.10)",
    },
    sensorDotLabel: {
        fontSize: "12px",
        fontWeight: 900,
        color: "var(--text-primary)",
        letterSpacing: "-0.01em",
    },
    jawLabelRow: {
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
    },
    jawDotIdle: {
        background: "rgba(255,45,141,0.40)",
        boxShadow: "0 0 0 6px rgba(255,45,141,0.06)",
    },
    jawDotActive: {
        background: "#FF2D8D",
        boxShadow: "0 0 0 8px rgba(255,45,141,0.14)",
    },
    /* Panels */
    rowPanels: {
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "18px",
        alignItems: "stretch",
    },
    cameraPanelSection: {
        display: "flex",
        flexDirection: "column",
    },
    panel: {
        borderRadius: "26px",
        border: "1px solid var(--border)",
        background: "var(--bg-secondary)",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        minHeight: "520px",
    },
    panelHeader: {
        padding: "16px 18px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "12px",
        background: "linear-gradient(180deg, rgba(255,45,141,0.06), transparent)",
    },
    panelHeaderLeft: {
        display: "flex",
        alignItems: "center",
        gap: "12px",
    },
    panelIcon: {
        width: "38px",
        height: "38px",
        borderRadius: "14px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg-tertiary)",
        color: "var(--text-secondary)",
        border: "1px solid var(--border)",
        flexShrink: 0,
    },
    panelTitle: {
        fontWeight: 950,
        color: "var(--text-primary)",
        fontSize: "14px",
        letterSpacing: "-0.01em",
    },
    panelDescription: {
        color: "var(--text-secondary)",
        fontSize: "12px",
        marginTop: "2px",
    },
    pill: {
        fontSize: "12px",
        fontWeight: 900,
        color: "#FF2D8D",
        background: "rgba(255,45,141,0.08)",
        border: "1px solid rgba(255,45,141,0.2)",
        padding: "8px 10px",
        borderRadius: "999px",
        whiteSpace: "nowrap",
    },
    panelBody: {
        padding: "18px",
        display: "flex",
        flexDirection: "column",
        gap: "14px",
        flex: 1,
    },
    placeholderOuter: {
        borderRadius: "18px",
        border: "1px dashed var(--border)",
        background: "radial-gradient(circle at 20% 20%, rgba(255,45,141,0.10), transparent 45%), radial-gradient(circle at 80% 20%, rgba(255,45,141,0.07), transparent 42%), linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.00))",
        height: "100%",
        minHeight: "420px",
        display: "flex",
        padding: "14px",
    },
    placeholderInner: {
        flex: 1,
        borderRadius: "14px",
        border: "1px solid var(--border)",
        background: "var(--bg-tertiary)",
        display: "grid",
        placeItems: "center",
    },
    placeholderText: {
        color: "var(--text-secondary)",
        fontSize: "12px",
        fontWeight: 800,
        letterSpacing: "0.12em",
        textTransform: "uppercase",
    },
};
export default function Visualizations() {
    const nav = useNavigate();
    const fakeLevels = useMemo(() => ({
        left: { signal: 0.62 },
        middle: { signal: 0.74 },
        right: { signal: 0.58 },
        jawClench: false,
    }), []);
    return (_jsxs(motion.div, { style: styles.container, variants: containerVariants, initial: "hidden", animate: "visible", children: [_jsxs(motion.section, { style: styles.header, variants: itemVariants, children: [_jsx("div", { style: styles.headerLeft, children: _jsxs("div", { style: styles.headerTitles, children: [_jsx("h1", { style: styles.title, children: "Visualizations" }), _jsx("p", { style: {
                                        margin: 0,
                                        fontSize: "14px",
                                        fontWeight: 700,
                                        color: "#FF2D8D",
                                        letterSpacing: "-0.01em",
                                    }, children: "Ever wonder what goes on inside your brain?" })] }) }), _jsx("div", { style: styles.headerRight, children: _jsx(StatusBadge, { label: "Streaming Ready" }) })] }), _jsxs(motion.section, { style: styles.sensorDotsRow, variants: itemVariants, children: [_jsx(SensorDot, { label: "Left", value: fakeLevels.left.signal }), _jsx(SensorDot, { label: "Middle", value: fakeLevels.middle.signal }), _jsx(SensorDot, { label: "Right", value: fakeLevels.right.signal }), _jsx(JawDot, { active: fakeLevels.jawClench })] }), _jsxs(motion.section, { style: styles.rowPanels, variants: containerVariants, children: [_jsxs(motion.div, { style: styles.panel, variants: itemVariants, children: [_jsx(PanelHeader, { icon: _jsx(Grid3X3, { size: 18 }), title: "EEG Heatmap", description: "Band power / channel intensity (placeholder)", rightSlot: _jsx(Pill, { label: "8\u201316 Hz \u03B1" }) }), _jsx("div", { style: styles.panelBody, children: _jsx("img", { src: "http://127.0.0.1:8770/topo" }) })] }), _jsxs(motion.div, { style: styles.panel, variants: itemVariants, children: [_jsx(PanelHeader, { icon: _jsx(Waves, { size: 18 }), title: "Brain Wave Visualizer", description: "Waveforms + bands (placeholder)", rightSlot: _jsx(Pill, { label: "Raw / Filtered" }) }), _jsx("div", { style: styles.panelBody, children: _jsx("img", { src: "http://127.0.0.1:8770/waves" }) })] })] }), _jsx(motion.section, { style: styles.cameraPanelSection, variants: itemVariants, children: _jsxs(motion.div, { style: styles.panel, variants: itemVariants, children: [_jsx(PanelHeader, { icon: _jsx(Camera, { size: 18 }), title: "Live Camera Preview", description: "ASL hand sign detection feed", rightSlot: _jsx(ASLStatusPill, {}) }), _jsx("div", { style: styles.panelBody, children: _jsx(ASLFeed, {}) })] }) })] }));
}
/* ----------------------------- UI Pieces ----------------------------- */
function StatusBadge({ label }) {
    return (_jsxs(motion.div, { style: styles.statusBadge, animate: { opacity: [1, 0.75, 1] }, transition: { duration: 2, repeat: Infinity, ease: "easeInOut" }, children: [_jsx(motion.div, { style: styles.statusDot, animate: { scale: [1, 1.2, 1], opacity: [1, 0.6, 1] }, transition: { duration: 1.4, repeat: Infinity, ease: "easeInOut" } }), label] }));
}
function PanelHeader({ icon, title, description, rightSlot, }) {
    return (_jsxs("div", { style: styles.panelHeader, children: [_jsxs("div", { style: styles.panelHeaderLeft, children: [_jsx("div", { style: styles.panelIcon, children: icon }), _jsxs("div", { children: [_jsx("div", { style: styles.panelTitle, children: title }), _jsx("div", { style: styles.panelDescription, children: description })] })] }), _jsx("div", { children: rightSlot })] }));
}
function Pill({ label }) {
    return _jsx("div", { style: styles.pill, children: label });
}
function PlaceholderCanvas({ label }) {
    return (_jsx("div", { style: styles.placeholderOuter, children: _jsx("div", { style: styles.placeholderInner, children: _jsx("div", { style: styles.placeholderText, children: label }) }) }));
}
const ASL_VIDEO_URL = "http://localhost:8765/video";
const ASL_STATUS_URL = "http://localhost:8765/";
function ASLFeed() {
    const [error, setError] = useState(false);
    return (_jsx("div", { style: styles.placeholderOuter, children: error ? (_jsx("div", { style: styles.placeholderInner, children: _jsx("div", { style: styles.placeholderText, children: "ASLCV server offline \u2014 start python -m app.api" }) })) : (_jsx("img", { src: ASL_VIDEO_URL, alt: "ASL live feed", onError: () => setError(true), style: {
                width: "100%",
                height: "100%",
                minHeight: 420,
                objectFit: "cover",
                borderRadius: 14,
                display: "block",
            } })) }));
}
function ASLStatusPill() {
    const [ready, setReady] = useState(null);
    useEffect(() => {
        let cancelled = false;
        const check = async () => {
            try {
                const res = await fetch(ASL_STATUS_URL, { signal: AbortSignal.timeout(1500) });
                const data = await res.json();
                if (!cancelled)
                    setReady(Boolean(data.detection_ready));
            }
            catch {
                if (!cancelled)
                    setReady(false);
            }
        };
        check();
        const interval = setInterval(check, 3000);
        return () => {
            cancelled = true;
            clearInterval(interval);
        };
    }, []);
    if (ready === null)
        return null;
    return (_jsx("div", { style: {
            fontSize: 12,
            fontWeight: 900,
            padding: "8px 10px",
            borderRadius: 999,
            whiteSpace: "nowrap",
            color: ready ? "#22c55e" : "#9ca3af",
            background: ready ? "rgba(34,197,94,0.1)" : "rgba(156,163,175,0.1)",
            border: `1px solid ${ready ? "rgba(34,197,94,0.25)" : "rgba(156,163,175,0.2)"}`,
        }, children: ready ? "● LIVE" : "○ OFFLINE" }));
}
function SensorDot({ label, value }) {
    const strength = Math.max(0, Math.min(1, value));
    const opacity = 0.35 + strength * 0.65;
    return (_jsxs("div", { style: styles.sensorDotWrap, children: [_jsx(motion.div, { style: { ...styles.sensorDot, opacity }, animate: { scale: [1, 1.08, 1] }, transition: { duration: 1.6, repeat: Infinity, ease: "easeInOut" } }), _jsx("div", { style: styles.sensorDotLabel, children: label })] }));
}
function JawDot({ active }) {
    return (_jsxs("div", { style: styles.sensorDotWrap, children: [_jsx(motion.div, { style: {
                    ...styles.sensorDot,
                    ...(active ? styles.jawDotActive : styles.jawDotIdle),
                }, animate: active ? { scale: [1, 1.18, 1] } : { scale: [1, 1.05, 1] }, transition: { duration: active ? 0.7 : 1.6, repeat: Infinity } }), _jsx("div", { style: styles.sensorDotLabel, children: _jsxs("span", { style: styles.jawLabelRow, children: [_jsx(Zap, { size: 12 }), "EEG"] }) })] }));
}
