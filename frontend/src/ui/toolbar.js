import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Home, BarChart3, Minimize2, Maximize2, Minus, Square, X, Sun, Moon } from "lucide-react";
import { useTheme } from "../context/ThemeContext";
export default function CompactToolbar() {
    const navigate = useNavigate();
    const { isDark, toggleTheme } = useTheme();
    const [isOverlay, setIsOverlay] = useState(false);
    useEffect(() => {
        // Get initial overlay mode
        window.electron?.getOverlayMode().then(setIsOverlay);
        // Listen for mode changes
        window.electron?.onOverlayModeChanged?.((mode) => {
            setIsOverlay(mode);
        });
    }, []);
    const toggleOverlay = async () => {
        const newMode = await window.electron?.toggleOverlay();
        if (newMode !== undefined) {
            setIsOverlay(newMode);
        }
    };
    return (_jsxs("div", { className: "app-title-bar", style: {
            ...styles.toolbar,
            background: "var(--toolbar-bg)",
            borderBottomColor: "var(--toolbar-border)",
        }, children: [_jsx("div", { style: styles.left, className: "app-drag-region", children: _jsx("span", { style: { ...styles.logo, color: "var(--toolbar-text)" }, children: "Conduit" }) }), _jsxs("div", { style: styles.center, className: "app-drag-region", children: [_jsx(ToolbarButton, { active: false, onClick: () => navigate("/home"), label: "Home", icon: Home }), _jsx(ToolbarButton, { active: false, onClick: () => navigate("/visuals"), label: "Visuals", icon: BarChart3 })] }), _jsxs("div", { style: styles.right, className: "app-no-drag", children: [_jsx("button", { onClick: () => window.electron?.minimize(), style: styles.windowButton, title: "Minimize", children: _jsx(Minus, { size: 14 }) }), _jsx("button", { onClick: () => window.electron?.maximize(), style: styles.windowButton, title: "Maximize", children: _jsx(Square, { size: 14 }) }), _jsx("button", { onClick: () => window.electron?.close(), style: styles.closeButton, title: "Close", children: _jsx(X, { size: 14 }) }), _jsx("button", { onClick: toggleTheme, style: styles.windowButton, title: isDark ? "Switch to light mode" : "Switch to dark mode", children: isDark ? _jsx(Sun, { size: 16 }) : _jsx(Moon, { size: 16 }) }), _jsxs("button", { onClick: toggleOverlay, style: styles.overlayButton, title: isOverlay ? "Exit overlay mode" : "Enter overlay mode", children: [isOverlay ? _jsx(Maximize2, { size: 16 }) : _jsx(Minimize2, { size: 16 }), _jsx("span", { children: isOverlay ? "Expand" : "Overlay" })] })] })] }));
}
const ToolbarButton = ({ active, onClick, label, icon: Icon, color }) => (_jsxs("button", { onClick: onClick, style: {
        ...styles.button,
        ...(active ? styles.buttonActive : {}),
        ...(color ? { color } : {}),
    }, children: [_jsx(Icon, { size: 16 }), _jsx("span", { children: label })] }));
const styles = {
    toolbar: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 16px",
        height: 48,
        borderBottom: "1px solid var(--toolbar-border)",
    },
    left: {
        display: "flex",
        alignItems: "center",
        gap: 12,
    },
    logo: {
        fontSize: 14,
        fontWeight: 600,
    },
    center: {
        display: "flex",
        gap: 8,
        flex: 1,
        justifyContent: "center",
        // WebkitAppRegion: "no-drag" as any, // Removed, use attribute if needed
    },
    right: {
        display: "flex",
        gap: 8,
        // WebkitAppRegion: "no-drag" as any, // Removed, use attribute if needed
    },
    button: {
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        background: "transparent",
        border: "1px solid transparent",
        borderRadius: 6,
        color: "var(--toolbar-muted)",
        cursor: "pointer",
        fontSize: 13,
        transition: "all 0.2s",
    },
    buttonActive: {
        background: "var(--bg-hover)",
        border: "1px solid var(--border)",
        color: "var(--toolbar-text)",
    },
    overlayButton: {
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 12px",
        background: "#3b82f6",
        border: "none",
        borderRadius: 6,
        color: "#ffffff",
        cursor: "pointer",
        fontSize: 13,
        fontWeight: 600,
        transition: "all 0.2s",
    },
    windowButton: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        background: "transparent",
        border: "none",
        borderRadius: 6,
        color: "var(--toolbar-muted)",
        cursor: "pointer",
        transition: "all 0.2s",
    },
    closeButton: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 32,
        height: 32,
        background: "transparent",
        border: "none",
        borderRadius: 6,
        color: "var(--toolbar-muted)",
        cursor: "pointer",
        transition: "all 0.2s",
        marginRight: 8,
    },
};
