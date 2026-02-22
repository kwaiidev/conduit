import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useEffect, useState } from "react";
const STORAGE_KEY = "conduit_theme";
function getSystemTheme() {
    if (typeof window === "undefined")
        return "light";
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function getStoredTheme() {
    if (typeof window === "undefined")
        return null;
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark")
        return v;
    return null;
}
function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
}
const ThemeContext = createContext(null);
export function ThemeProvider({ children }) {
    const [theme, setThemeState] = useState(() => {
        const stored = getStoredTheme();
        const initial = stored ?? getSystemTheme();
        if (typeof document !== "undefined")
            applyTheme(initial);
        return initial;
    });
    useEffect(() => {
        applyTheme(theme);
    }, [theme]);
    useEffect(() => {
        const stored = getStoredTheme();
        if (stored) {
            setThemeState(stored);
            applyTheme(stored);
            return;
        }
        const system = getSystemTheme();
        setThemeState(system);
        applyTheme(system);
        const mq = window.matchMedia("(prefers-color-scheme: dark)");
        const handle = () => {
            if (getStoredTheme() !== null)
                return;
            const next = mq.matches ? "dark" : "light";
            setThemeState(next);
            applyTheme(next);
        };
        mq.addEventListener("change", handle);
        return () => mq.removeEventListener("change", handle);
    }, []);
    const setTheme = (next) => {
        setThemeState(next);
        localStorage.setItem(STORAGE_KEY, next);
        applyTheme(next);
    };
    const toggleTheme = () => {
        const next = theme === "light" ? "dark" : "light";
        setTheme(next);
    };
    return (_jsx(ThemeContext.Provider, { value: {
            theme,
            isDark: theme === "dark",
            setTheme,
            toggleTheme,
        }, children: children }));
}
export function useTheme() {
    const ctx = useContext(ThemeContext);
    if (!ctx)
        throw new Error("useTheme must be used within ThemeProvider");
    return ctx;
}
