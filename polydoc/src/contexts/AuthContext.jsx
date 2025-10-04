import React, { createContext, useContext, useEffect, useState } from 'react';
import { 
  onAuthStateChanged, 
  signInWithPopup, 
  signInWithRedirect,
  getRedirectResult,
  signOut as firebaseSignOut
} from 'firebase/auth';
import { doc, setDoc, getDoc } from 'firebase/firestore';
import { auth, googleProvider, db } from '@/config/firebase';
import useNetworkStatus from '@/hooks/useNetworkStatus';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [authError, setAuthError] = useState(null);
  const { isOnline } = useNetworkStatus();

  // Helper function to handle user document creation/update
  const handleUserDocument = async (user) => {
    try {
      const userDocRef = doc(db, 'users', user.uid);
      const userDoc = await getDoc(userDocRef);
      
      if (!userDoc.exists()) {
        await setDoc(userDocRef, {
          uid: user.uid,
          email: user.email,
          displayName: user.displayName,
          photoURL: user.photoURL,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        });
      } else {
        // Update last login
        await setDoc(userDocRef, {
          updatedAt: new Date().toISOString(),
        }, { merge: true });
      }
    } catch (firestoreError) {
      console.warn('Failed to update user document in Firestore:', firestoreError);
      // Don't fail authentication if Firestore is unavailable
    }
  };

  // Sign in with Google (popup method with redirect fallback)
  const signInWithGoogle = async (useRedirect = false) => {
    if (!isOnline) {
      const errorMsg = 'Cannot sign in while offline. Please check your internet connection.';
      setAuthError(errorMsg);
      return { success: false, error: errorMsg };
    }

    setAuthError(null);
    
    // If useRedirect is true, use redirect method
    if (useRedirect) {
      try {
        await signInWithRedirect(auth, googleProvider);
        // The result will be handled in the redirect result check in useEffect
        return { success: true, redirect: true };
      } catch (error) {
        console.error('Error with redirect sign in:', error);
        const errorMessage = handleAuthError(error);
        setAuthError(errorMessage);
        return { success: false, error: errorMessage };
      }
    }
    
    // Try popup method first
    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      
      await handleUserDocument(user);
      
      return { success: true, user };
    } catch (error) {
      console.error('Error signing in with Google popup:', error);
      
      // If popup fails due to CORS or popup blocking, try redirect as fallback
      if (error.code === 'auth/popup-blocked' || 
          error.code === 'auth/popup-closed-by-user' ||
          error.message.includes('CORS') ||
          error.message.includes('cross-origin')) {
        console.log('Popup failed, trying redirect method...');
        return await signInWithGoogle(true); // Recursive call with redirect
      }
      
      const errorMessage = handleAuthError(error);
      setAuthError(errorMessage);
      return { success: false, error: errorMessage };
    }
  };
  
  // Enhanced error handling function
  const handleAuthError = (error) => {
    console.error('Firebase Auth Error Details:', {
      code: error.code,
      message: error.message,
      customData: error.customData
    });
    
    // Handle specific Firebase auth errors with user-friendly messages
    switch (error.code) {
      case 'auth/network-request-failed':
        return 'Network error. Please check your internet connection and try again.';
      case 'auth/unauthorized-domain':
        return 'This domain is not authorized for Firebase authentication. Please contact the administrator.';
      case 'auth/popup-blocked':
        return 'Popup was blocked by your browser. Please allow popups and try again.';
      case 'auth/popup-closed-by-user':
        return 'Sign-in was cancelled. Please try again.';
      case 'auth/cancelled-popup-request':
        return 'Another sign-in popup is already open. Please close it and try again.';
      case 'auth/too-many-requests':
        return 'Too many failed attempts. Please wait a moment and try again.';
      case 'auth/user-disabled':
        return 'Your account has been disabled. Please contact support.';
      case 'auth/operation-not-allowed':
        return 'Google sign-in is not enabled for this app. Please contact the administrator.';
      case 'auth/internal-error':
        return 'An internal error occurred. Please try again later.';
      case 'auth/web-storage-unsupported':
        return 'Your browser does not support web storage. Please try a different browser.';
      default:
        // For unknown errors, check if it's a network/connectivity issue
        if (error.message.includes('CORS') || error.message.includes('cross-origin')) {
          return 'Browser security settings are blocking sign-in. Please try using the redirect method or check your browser settings.';
        }
        if (error.message.includes('Failed to fetch') || error.message.includes('Network Error')) {
          return 'Unable to connect to authentication service. Please check your internet connection.';
        }
        return `Authentication failed: ${error.message}`;
    }
  };

  // Sign in with redirect (alternative method)
  const signInWithGoogleRedirect = async () => {
    if (!isOnline) {
      const errorMsg = 'Cannot sign in while offline. Please check your internet connection.';
      setAuthError(errorMsg);
      return { success: false, error: errorMsg };
    }

    setAuthError(null);
    try {
      await signInWithRedirect(auth, googleProvider);
      return { success: true, redirect: true };
    } catch (error) {
      console.error('Error with redirect sign in:', error);
      setAuthError(error.message);
      return { success: false, error: error.message };
    }
  };

  // Sign out
  const signOut = async () => {
    try {
      await firebaseSignOut(auth);
      return { success: true };
    } catch (error) {
      console.error('Error signing out:', error);
      return { success: false, error: error.message };
    }
  };

  // Listen for auth state changes and handle redirect results
  useEffect(() => {
    if (!auth) {
      console.warn('Firebase auth not initialized');
      setLoading(false);
      return;
    }

    // Check for redirect result on component mount
    const checkRedirectResult = async () => {
      try {
        const result = await getRedirectResult(auth);
        if (result && result.user) {
          console.log('Successfully signed in via redirect:', result.user.email);
          await handleUserDocument(result.user);
        }
      } catch (error) {
        console.error('Error handling redirect result:', error);
        setAuthError(error.message);
      }
    };

    checkRedirectResult();

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    }, (error) => {
      console.error('Auth state change error:', error);
      setLoading(false);
    });

    return unsubscribe;
  }, []);

  const value = {
    user,
    loading,
    authError,
    isOnline,
    signInWithGoogle,
    signInWithGoogleRedirect,
    signOut,
    clearAuthError: () => setAuthError(null),
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
