// view-firestore-data.js
// Node.js script to view Firebase Firestore data

const { initializeApp } = require('firebase/app');
const { getFirestore, collection, getDocs, query, where } = require('firebase/firestore');

// Your Firebase config (copy from src/config/firebase.js)
const firebaseConfig = {
  apiKey: "your-api-key-here",
  authDomain: "your-project-id.firebaseapp.com",
  projectId: "your-project-id",
  storageBucket: "your-project-id.appspot.com",
  messagingSenderId: "your-sender-id",
  appId: "your-app-id"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

async function viewAllUsers() {
  console.log('🔥 FIREBASE USERS:');
  console.log('================');
  
  try {
    const usersCollection = collection(db, 'users');
    const userSnapshot = await getDocs(usersCollection);
    
    if (userSnapshot.empty) {
      console.log('No users found in Firestore');
      return;
    }
    
    userSnapshot.forEach((doc) => {
      const userData = doc.data();
      console.log(`👤 User ID: ${doc.id}`);
      console.log(`   Email: ${userData.email}`);
      console.log(`   Name: ${userData.displayName}`);
      console.log(`   Created: ${userData.createdAt}`);
      console.log(`   Last Updated: ${userData.updatedAt}`);
      console.log('');
    });
  } catch (error) {
    console.error('Error fetching users:', error);
  }
}

async function viewUserPreferences() {
  console.log('⚙️ USER PREFERENCES:');
  console.log('===================');
  
  try {
    const prefsCollection = collection(db, 'user_preferences');
    const prefsSnapshot = await getDocs(prefsCollection);
    
    if (prefsSnapshot.empty) {
      console.log('No user preferences found');
      return;
    }
    
    prefsSnapshot.forEach((doc) => {
      const prefs = doc.data();
      console.log(`🎨 User: ${doc.id}`);
      console.log(`   Theme: ${prefs.theme || 'default'}`);
      console.log(`   Language: ${prefs.language || 'en'}`);
      console.log(`   Notifications: ${prefs.notifications || false}`);
      console.log('');
    });
  } catch (error) {
    console.error('Error fetching preferences:', error);
  }
}

async function viewChatMetadata() {
  console.log('💬 CHAT METADATA:');
  console.log('================');
  
  try {
    const chatsCollection = collection(db, 'chat_metadata');
    const chatsSnapshot = await getDocs(chatsCollection);
    
    if (chatsSnapshot.empty) {
      console.log('No chat metadata found');
      return;
    }
    
    chatsSnapshot.forEach((doc) => {
      const chat = doc.data();
      console.log(`💭 Chat: ${doc.id}`);
      console.log(`   User: ${chat.user_id}`);
      console.log(`   Document: ${chat.document_id}`);
      console.log(`   Title: ${chat.title}`);
      console.log(`   Last Message: ${chat.last_message || 'N/A'}`);
      console.log(`   Updated: ${chat.updated_at}`);
      console.log('');
    });
  } catch (error) {
    console.error('Error fetching chat metadata:', error);
  }
}

async function viewSpecificUser(email) {
  console.log(`🔍 SEARCHING FOR USER: ${email}`);
  console.log('===============================');
  
  try {
    const usersCollection = collection(db, 'users');
    const q = query(usersCollection, where('email', '==', email));
    const querySnapshot = await getDocs(q);
    
    if (querySnapshot.empty) {
      console.log(`No user found with email: ${email}`);
      return;
    }
    
    querySnapshot.forEach((doc) => {
      const userData = doc.data();
      console.log(`✅ Found User:`);
      console.log(`   Firebase UID: ${doc.id}`);
      console.log(`   Email: ${userData.email}`);
      console.log(`   Display Name: ${userData.displayName}`);
      console.log(`   Photo URL: ${userData.photoURL}`);
      console.log(`   Account Created: ${userData.createdAt}`);
      console.log(`   Last Sign-in: ${userData.updatedAt}`);
    });
  } catch (error) {
    console.error('Error searching for user:', error);
  }
}

// Main execution
async function main() {
  console.log('🔥 PolyDoc AI - Firebase Firestore Data Viewer');
  console.log('===============================================\n');
  
  await viewAllUsers();
  await viewUserPreferences();
  await viewChatMetadata();
  
  // Example: Search for specific user
  // await viewSpecificUser('your-email@example.com');
  
  console.log('✅ Data viewing complete!');
  process.exit(0);
}

// Handle command line arguments
const args = process.argv.slice(2);
if (args.length > 0) {
  const command = args[0];
  const param = args[1];
  
  switch (command) {
    case 'users':
      viewAllUsers().then(() => process.exit(0));
      break;
    case 'preferences':
      viewUserPreferences().then(() => process.exit(0));
      break;
    case 'chats':
      viewChatMetadata().then(() => process.exit(0));
      break;
    case 'search':
      if (param) {
        viewSpecificUser(param).then(() => process.exit(0));
      } else {
        console.log('Usage: node view-firestore-data.js search email@example.com');
        process.exit(1);
      }
      break;
    default:
      console.log('Usage:');
      console.log('  node view-firestore-data.js users');
      console.log('  node view-firestore-data.js preferences');  
      console.log('  node view-firestore-data.js chats');
      console.log('  node view-firestore-data.js search email@example.com');
      process.exit(1);
  }
} else {
  main();
}
