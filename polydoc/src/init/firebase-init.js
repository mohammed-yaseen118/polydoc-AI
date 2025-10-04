/**
 * Firebase Initialization and Connection Monitoring
 * This module handles Firebase initialization, connection monitoring, and error recovery
 */

import { initializeConnectionMonitoring } from '@/config/firebase';

// Firebase initialization flag
let isInitialized = false;
let initializationPromise = null;

/**
 * Initialize Firebase with proper error handling and monitoring
 */
export const initializeFirebase = async () => {
  if (isInitialized) {
    return true;
  }

  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {
    try {
      console.log('Initializing Firebase services...');
      
      // Start connection monitoring
      initializeConnectionMonitoring();
      
      // Wait for initial connection to be established
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Firebase initialization timeout'));
        }, 15000); // 15 second timeout
        
        // Check Firebase connection
        const checkConnection = async () => {
          try {
            // Try to access Firebase services
            const { auth, db } = await import('@/config/firebase');
            
            if (auth && db) {
              clearTimeout(timeout);
              resolve(true);
            } else {
              setTimeout(checkConnection, 1000);
            }
          } catch (error) {
            clearTimeout(timeout);
            reject(error);
          }
        };
        
        checkConnection();
      });
      
      isInitialized = true;
      console.log('Firebase initialized successfully');
      return true;
      
    } catch (error) {
      console.error('Firebase initialization failed:', error);
      
      // Reset for retry
      initializationPromise = null;
      
      // Don't throw error - allow graceful degradation
      console.warn('Continuing without Firebase - some features may be limited');
      return false;
    }
  })();

  return initializationPromise;
};

/**
 * Check if Firebase is properly initialized
 */
export const isFirebaseReady = () => {
  return isInitialized;
};

/**
 * Reset Firebase initialization (for testing or recovery)
 */
export const resetFirebaseInitialization = () => {
  isInitialized = false;
  initializationPromise = null;
};

/**
 * Get Firebase connection status
 */
export const getFirebaseStatus = async () => {
  try {
    const { testFirebaseConnection } = await import('@/config/firebase');
    const isConnected = await testFirebaseConnection();
    
    return {
      initialized: isInitialized,
      connected: isConnected,
      status: isInitialized && isConnected ? 'ready' : 'degraded'
    };
  } catch (error) {
    return {
      initialized: false,
      connected: false,
      status: 'error',
      error: error.message
    };
  }
};

// Initialize Firebase when module is imported
if (typeof window !== 'undefined') {
  // Initialize Firebase after a short delay to allow other modules to load
  setTimeout(() => {
    initializeFirebase().catch(error => {
      console.warn('Initial Firebase setup failed:', error);
    });
  }, 1000);
}
