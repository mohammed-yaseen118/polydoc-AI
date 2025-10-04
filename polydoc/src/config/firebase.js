import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, connectAuthEmulator } from 'firebase/auth';
import { getFirestore, connectFirestoreEmulator, enableNetwork, disableNetwork, doc, getDoc } from 'firebase/firestore';
import { getStorage, connectStorageEmulator } from 'firebase/storage';

// Debug environment variables
console.log('Environment check:');
console.log('NODE_ENV:', import.meta.env.MODE);
console.log('VITE_FIREBASE_API_KEY exists:', !!import.meta.env.VITE_FIREBASE_API_KEY);
console.log('VITE_FIREBASE_AUTH_DOMAIN exists:', !!import.meta.env.VITE_FIREBASE_AUTH_DOMAIN);
console.log('VITE_FIREBASE_PROJECT_ID exists:', !!import.meta.env.VITE_FIREBASE_PROJECT_ID);

// Firebase configuration with fallback values from your .env
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY || "AIzaSyCmAMUuacg2EPx6CXJ9-wB0M0HBojVhz2Y",
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN || "polydoc-b2c8f.firebaseapp.com",
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID || "polydoc-b2c8f",
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET || "polydoc-b2c8f.firebasestorage.app",
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID || "415564421653",
  appId: import.meta.env.VITE_FIREBASE_APP_ID || "1:415564421653:web:db431fa12ac81b3e550c1c"
};

// Debug final configuration
console.log('Firebase config loaded:', {
  apiKey: firebaseConfig.apiKey ? 'Present' : 'Missing',
  authDomain: firebaseConfig.authDomain,
  projectId: firebaseConfig.projectId,
  storageBucket: firebaseConfig.storageBucket,
  messagingSenderId: firebaseConfig.messagingSenderId ? 'Present' : 'Missing',
  appId: firebaseConfig.appId ? 'Present' : 'Missing'
});

// Validate configuration
if (!firebaseConfig.apiKey || !firebaseConfig.authDomain || !firebaseConfig.projectId) {
  console.error('Firebase configuration is incomplete. Please check your .env file.');
  console.error('Required variables: VITE_FIREBASE_API_KEY, VITE_FIREBASE_AUTH_DOMAIN, VITE_FIREBASE_PROJECT_ID');
  throw new Error('Firebase configuration is incomplete');
}

// Initialize Firebase with error suppression
let app;
try {
  app = initializeApp(firebaseConfig);
  console.log('Firebase app initialized successfully');
} catch (error) {
  console.warn('Firebase initialization warning:', error.message);
  // Continue with limited functionality
  app = initializeApp(firebaseConfig); // Try again
}

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app);

// Initialize Google Auth Provider
export const googleProvider = new GoogleAuthProvider();

// Add settings to avoid popup issues and handle CORS better
googleProvider.setCustomParameters({
  prompt: 'select_account',
  // Force account selection to avoid cached login states that might cause CORS issues
  login_hint: null
});

// Add OAuth scopes for proper authentication
googleProvider.addScope('email');
googleProvider.addScope('profile');

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);

// Check if we're in development and should use emulators
const useEmulators = import.meta.env.MODE === 'development' && import.meta.env.VITE_USE_FIREBASE_EMULATORS === 'true';

// Add retry mechanism for network failures
let isInitializing = false;
const MAX_RETRY_ATTEMPTS = 3;
const RETRY_DELAY = 2000; // 2 seconds

// Add connection retry helper
const withRetry = async (operation, maxAttempts = MAX_RETRY_ATTEMPTS) => {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      console.warn(`Attempt ${attempt}/${maxAttempts} failed:`, error.message);
      if (attempt === maxAttempts) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
    }
  }
};

// Initialize Firebase Storage
export const storage = getStorage(app);

// Global flag to track emulator connections
let emulatorsAlreadyConnected = false;

// Connect to emulators if enabled (only in development)
if (useEmulators && !emulatorsAlreadyConnected) {
  let emulatorsConnected = false;
  
  try {
    // Connect to Auth emulator first (most important for CORS issues)
    connectAuthEmulator(auth, `http://localhost:${import.meta.env.VITE_AUTH_EMULATOR_PORT || 9099}`, { disableWarnings: true });
    console.log('Connected to Auth emulator on port', import.meta.env.VITE_AUTH_EMULATOR_PORT || 9099);
    emulatorsConnected = true;
  } catch (error) {
    console.warn('Failed to connect to Auth emulator:', error.message);
  }
  
  try {
    // Connect to Firestore emulator
    connectFirestoreEmulator(db, 'localhost', 8080);
    console.log('Connected to Firestore emulator on port 8080');
    emulatorsConnected = true;
  } catch (error) {
    console.warn('Failed to connect to Firestore emulator:', error.message);
  }
  
  try {
    // Connect to Storage emulator
    connectStorageEmulator(storage, 'localhost', 9199);
    console.log('Connected to Storage emulator on port 9199');
    emulatorsConnected = true;
  } catch (error) {
    console.warn('Failed to connect to Storage emulator:', error.message);
  }
  
  emulatorsAlreadyConnected = true;
  
  if (emulatorsConnected) {
    console.log('Firebase emulators are active - using local development environment');
  } else {
    console.log('Firebase emulators not available - using production Firebase');
  }
}

// Connection state management
let isConnectionHealthy = true;
let connectionRetryCount = 0;
const MAX_CONNECTION_RETRIES = 5;
const CONNECTION_RETRY_DELAY = 1000; // 1 second

// Enhanced network connectivity handler
export const handleFirestoreOffline = async () => {
  try {
    isConnectionHealthy = false;
    await disableNetwork(db);
    console.log('Firestore offline mode enabled');
  } catch (error) {
    console.warn('Failed to enable Firestore offline mode:', error);
  }
};

export const handleFirestoreOnline = async () => {
  try {
    if (!isConnectionHealthy) {
      await withRetry(async () => {
        await enableNetwork(db);
        isConnectionHealthy = true;
      });
      console.log('Firestore online mode enabled');
    }
  } catch (error) {
    console.warn('Failed to enable Firestore online mode:', error);
  }
};

// Test Firebase connection health
export const testFirebaseConnection = async () => {
  try {
    // Try a simple Firestore operation
    const testDoc = doc(db, '_connection_test', 'test');
    await getDoc(testDoc);
    return true;
  } catch (error) {
    console.warn('Firebase connection test failed:', error);
    return false;
  }
};

// Initialize connection monitoring
export const initializeConnectionMonitoring = () => {
  // Monitor connection state changes
  if (typeof window !== 'undefined') {
    // Test connection periodically
    setInterval(async () => {
      if (navigator.onLine) {
        const isHealthy = await testFirebaseConnection();
        if (!isHealthy && isConnectionHealthy) {
          console.warn('Firebase connection lost, enabling offline mode');
          await handleFirestoreOffline();
        } else if (isHealthy && !isConnectionHealthy) {
          console.log('Firebase connection restored, enabling online mode');
          await handleFirestoreOnline();
        }
      }
    }, 30000); // Check every 30 seconds
  }
};

// Listen for online/offline events
if (typeof window !== 'undefined') {
  window.addEventListener('online', handleFirestoreOnline);
  window.addEventListener('offline', handleFirestoreOffline);
}

export default app;
