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

const isSnapOverlayQuery =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("snapOverlay") === "1";
const launchPathQuery =
  typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("launchPath")
    : null;

if (
  !isSnapOverlayQuery &&
  typeof window !== "undefined" &&
  typeof launchPathQuery === "string" &&
  launchPathQuery.startsWith("/") &&
  launchPathQuery.length > 1
) {
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
const OverlayOnly: React.FC = () => (
  <div style={{ minHeight: '100%', background: 'transparent' }}>
    <OverlayBar />
  </div>
);

// Protected route wrapper - redirects to onboarding if not completed
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  if (!hasCompletedOnboarding()) {
    return <Navigate to="/onboarding" replace />;
  }
  return <>{children}</>;
};

// Root component that handles routing (full window only)
const SnapCursorRoot: React.FC = () => {
  const location = useLocation();
  if (location.pathname.startsWith('/onboarding')) {
    return null;
  }
  return <SnapCursorLayer />;
};

const PendingRouteRedirect: React.FC = () => {
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

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <PendingRouteRedirect />
        <Routes>
          <Route path="/onboarding" element={<Onboarding />} />
          <Route
            path="/home"
            element={
              <ProtectedRoute>
                <Shell>
                  <Home />
                </Shell>
              </ProtectedRoute>
            }
          />
          <Route
            path="/visuals"
            element={
              <ProtectedRoute>
                <Shell>
                  <Visuals />
                </Shell>
              </ProtectedRoute>
            }
          />
          <Route
            path="/"
            element={<Navigate to="/onboarding" replace />}
          />
          <Route
            path="*"
            element={<Navigate to="/onboarding" replace />}
          />
        </Routes>
        <SnapCursorRoot />
      </BrowserRouter>
    </ThemeProvider>
  );
};



// Decides whether to show overlay bar only (overlay window) or full app (router).
const AppBoot: React.FC = () => {
  const isSnapOverlayWindow = isSnapOverlayQuery;
  const [overlayMode, setOverlayMode] = useState<boolean | null>(null);

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
    return <SnapCursorLayer />;
  }

  if (overlayMode === true) {
    return <OverlayOnly />;
  }
  if (overlayMode === false) {
    return <App />;
  }
  return null;
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppBoot />
  </React.StrictMode>
);
