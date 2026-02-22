import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Shell from './ui/shell';
import OverlayBar from './ui/overlay';
import SnapCursorLayer from './ui/snapCursorLayer';
import Onboarding from './routes/onboarding';
import Home from './routes/home';
import Visuals from './routes/visuals';
import { hasCompletedOnboarding } from './state/onboarding';
import '../index.css';
const isSnapOverlayQuery = typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("snapOverlay") === "1";
const launchPathQuery = typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("launchPath")
    : null;
if (!isSnapOverlayQuery &&
    typeof window !== "undefined" &&
    typeof launchPathQuery === "string" &&
    launchPathQuery.startsWith("/") &&
    launchPathQuery.length > 1) {
    const requiresOnboarding = !hasCompletedOnboarding() && launchPathQuery !== "/onboarding";
    const safeLaunchPath = requiresOnboarding ? "/onboarding" : launchPathQuery;
    if (window.location.pathname !== safeLaunchPath || window.location.search) {
        window.history.replaceState(null, "", safeLaunchPath);
    }
}
if (isSnapOverlayQuery && typeof document !== "undefined") {
    document.documentElement.style.background = "transparent";
    document.body.style.background = "transparent";
    document.body.style.margin = "0";
    const root = document.getElementById("root");
    if (root) {
        root.style.background = "transparent";
    }
}
// When we're in overlay mode, the window only shows the overlay bar (no router).
// This runs before any route so the overlay window shows the bar instead of onboarding.
const OverlayOnly = () => (_jsx("div", { style: { minHeight: '100%', background: 'transparent' }, children: _jsx(OverlayBar, {}) }));
// Protected route wrapper - redirects to onboarding if not completed
const ProtectedRoute = ({ children }) => {
    if (!hasCompletedOnboarding()) {
        return _jsx(Navigate, { to: "/onboarding", replace: true });
    }
    return _jsx(_Fragment, { children: children });
};
// Root component that handles routing (full window only)
const SnapCursorRoot = () => {
    const location = useLocation();
    if (location.pathname.startsWith('/onboarding')) {
        return null;
    }
    return _jsx(SnapCursorLayer, {});
};
const PendingRouteRedirect = () => {
    const navigate = useNavigate();
    useEffect(() => {
        let mounted = true;
        if (typeof window.electron?.consumePendingRoute !== "function") {
            return () => {
                mounted = false;
            };
        }
        window.electron.consumePendingRoute().then((targetPath) => {
            if (!mounted) {
                return;
            }
            if (typeof targetPath === "string" && targetPath.startsWith("/")) {
                const requiresOnboarding = !hasCompletedOnboarding() && targetPath !== "/onboarding";
                const safeTargetPath = requiresOnboarding ? "/onboarding" : targetPath;
                navigate(safeTargetPath, { replace: true });
            }
        });
        return () => {
            mounted = false;
        };
    }, [navigate]);
    return null;
};
const App = () => {
    return (_jsx(ThemeProvider, { children: _jsxs(BrowserRouter, { children: [_jsx(PendingRouteRedirect, {}), _jsxs(Routes, { children: [_jsx(Route, { path: "/onboarding", element: _jsx(Onboarding, {}) }), _jsx(Route, { path: "/home", element: _jsx(ProtectedRoute, { children: _jsx(Shell, { children: _jsx(Home, {}) }) }) }), _jsx(Route, { path: "/visuals", element: _jsx(ProtectedRoute, { children: _jsx(Shell, { children: _jsx(Visuals, {}) }) }) }), _jsx(Route, { path: "/", element: _jsx(Navigate, { to: hasCompletedOnboarding() ? "/home" : "/onboarding", replace: true }) }), _jsx(Route, { path: "*", element: _jsx(Navigate, { to: hasCompletedOnboarding() ? "/home" : "/onboarding", replace: true }) })] }), _jsx(SnapCursorRoot, {})] }) }));
};
// Decides whether to show overlay bar only (overlay window) or full app (router).
const AppBoot = () => {
    const isSnapOverlayWindow = isSnapOverlayQuery;
    const [overlayMode, setOverlayMode] = useState(null);
    useEffect(() => {
        if (isSnapOverlayWindow) {
            return;
        }
        if (typeof window.electron?.getOverlayMode !== 'function') {
            setOverlayMode(false);
            return;
        }
        window.electron.getOverlayMode().then(setOverlayMode);
        window.electron?.onOverlayModeChanged?.(setOverlayMode);
    }, [isSnapOverlayWindow]);
    if (isSnapOverlayWindow) {
        return _jsx(SnapCursorLayer, {});
    }
    if (overlayMode === true) {
        return _jsx(OverlayOnly, {});
    }
    if (overlayMode === false) {
        return _jsx(App, {});
    }
    return null;
};
ReactDOM.createRoot(document.getElementById('root')).render(_jsx(React.StrictMode, { children: _jsx(AppBoot, {}) }));
