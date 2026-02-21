import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext';
import Shell from './ui/shell';
import OverlayBar from './ui/overlay';
import Onboarding from './routes/onboarding';
import Home from './routes/home';
import { hasCompletedOnboarding } from './state/onboarding';
import '../index.css';

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
const App: React.FC = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
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
          <Route path="/" element={<Navigate to="/onboarding" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
};

// Decides whether to show overlay bar only (overlay window) or full app (router).
const AppBoot: React.FC = () => {
  const [overlayMode, setOverlayMode] = useState<boolean | null>(null);

  useEffect(() => {
    if (typeof window.electron?.getOverlayMode !== 'function') {
      setOverlayMode(false);
      return;
    }
    window.electron.getOverlayMode().then(setOverlayMode);
    window.electron?.onOverlayModeChanged?.(setOverlayMode);
  }, []);

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