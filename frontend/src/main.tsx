import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Shell from './ui/shell';
import Onboarding from './routes/onboarding';
import Home from './routes/home';
import { hasCompletedOnboarding } from './state/onboarding';
import '../index.css';

// Protected route wrapper - redirects to onboarding if not completed
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  if (!hasCompletedOnboarding()) {
    return <Navigate to="/onboarding" replace />;
  }
  return <>{children}</>;
};

// Root component that handles routing
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        {/* Onboarding route - no shell wrapper */}
        <Route path="/onboarding" element={<Onboarding />} />
        
        {/* Protected routes with Shell wrapper */}
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
        
        {/* Root redirect - send to onboarding or home based on completion */}
        <Route
          path="/"
          element={
            hasCompletedOnboarding() ? (
              <Navigate to="/home" replace />
            ) : (
              <Navigate to="/onboarding" replace />
            )
          }
        />
        
        {/* Catch all - redirect to root */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
};

// Mount the app
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);