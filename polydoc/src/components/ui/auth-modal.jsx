import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Chrome, Loader2 } from 'lucide-react';
import { Button } from './button';
import { useAuth } from '@/contexts/AuthContext';

export const AuthModal = ({ isOpen, onClose, mode = 'signin' }) => {
  const { signInWithGoogle } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError('');
    
    const result = await signInWithGoogle();
    
    if (result.success) {
      onClose();
    } else {
      setError(result.error);
    }
    
    setLoading(false);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-black/50 backdrop-blur-sm"
          onClick={onClose}
        />
        
        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="relative w-full max-w-md bg-background border border-muted rounded-3xl p-6 shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 hover:bg-muted rounded-full transition-colors"
          >
            <X className="h-4 w-4" />
          </button>

          {/* Content */}
          <div className="space-y-6">
            {/* Header */}
            <div className="text-center space-y-2">
              <h2 className="text-2xl font-bold">
                {mode === 'signin' ? 'Welcome Back' : 'Get Started'}
              </h2>
              <p className="text-muted-foreground">
                {mode === 'signin' 
                  ? 'Sign in to access your documents and continue your conversations' 
                  : 'Create your account to start processing documents with AI'
                }
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-3 bg-destructive/10 border border-destructive/20 rounded-xl text-destructive text-sm"
              >
                {error}
              </motion.div>
            )}

            {/* Google Sign In Button */}
            <Button
              onClick={handleGoogleSignIn}
              disabled={loading}
              className="w-full h-12 rounded-xl bg-white hover:bg-gray-50 text-gray-700 border border-gray-200 shadow-sm"
              variant="outline"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <>
                  <Chrome className="h-5 w-5 mr-2" />
                  Continue with Google
                </>
              )}
            </Button>

            {/* Features List */}
            <div className="space-y-3 pt-4">
              <p className="text-sm font-medium text-muted-foreground">What you'll get:</p>
              <div className="space-y-2">
                {[
                  'Process multiple document formats',
                  'Multi-lingual document support',
                  'AI-powered document chat',
                  'Secure cloud storage',
                  'Document history & management'
                ].map((feature, index) => (
                  <div key={index} className="flex items-center space-x-2">
                    <div className="h-1.5 w-1.5 bg-primary rounded-full" />
                    <span className="text-sm text-muted-foreground">{feature}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <p className="text-xs text-muted-foreground text-center">
              By continuing, you agree to our Terms of Service and Privacy Policy
            </p>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
