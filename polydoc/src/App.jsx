import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/contexts/AuthContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import ErrorBoundary from '@/components/ErrorBoundary';
import LandingPage from '@/pages/LandingPage';
import Dashboard from '@/pages/Dashboard';
import ProtectedRoute from '@/components/ProtectedRoute';

function App() {
  return (
    <ErrorBoundary fallbackMessage="Something went wrong with the application. Please refresh the page.">
      <ThemeProvider>
        <AuthProvider>
          <Router>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route 
                path="/dashboard" 
                element={
                  <ProtectedRoute>
                    <ErrorBoundary fallbackMessage="There was an issue loading the dashboard. Please try again.">
                      <Dashboard />
                    </ErrorBoundary>
                  </ProtectedRoute>
                } 
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
