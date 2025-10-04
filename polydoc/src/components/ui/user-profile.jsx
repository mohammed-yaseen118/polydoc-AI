import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { LogOut, User, FileText, Settings } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from './button';

export const UserProfile = ({ className = '' }) => {
  const { user, signOut } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSignOut = async () => {
    await signOut();
    setIsOpen(false);
  };

  if (!user) return null;

  return (
    <div className={`relative ${className}`} ref={dropdownRef}>
      {/* Profile Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 p-2 rounded-3xl hover:bg-muted transition-colors"
      >
        <img
          src={user.photoURL || '/default-avatar.svg'}
          alt={user.displayName || 'User'}
          className="w-8 h-8 rounded-full ring-2 ring-primary/20 transition-all duration-200 hover:ring-primary/40"
          onError={(e) => {
            e.target.src = '/default-avatar.svg';
          }}
        />
        <span className="hidden sm:block text-sm font-medium truncate max-w-24">
          {user.displayName?.split(' ')[0] || 'User'}
        </span>
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="absolute right-0 top-full mt-2 w-64 bg-background border border-muted rounded-2xl shadow-lg overflow-hidden z-50"
          >
            {/* User Info */}
            <div className="p-4 border-b border-muted">
              <div className="flex items-center space-x-3">
                <img
                  src={user.photoURL || '/default-avatar.svg'}
                  alt={user.displayName || 'User'}
                  className="w-10 h-10 rounded-full ring-2 ring-primary/20"
                  onError={(e) => {
                    e.target.src = '/default-avatar.svg';
                  }}
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{user.displayName}</p>
                  <p className="text-sm text-muted-foreground truncate">{user.email}</p>
                </div>
              </div>
            </div>

            {/* Menu Items */}
            <div className="p-2">
              <button className="w-full flex items-center space-x-3 p-3 rounded-xl hover:bg-muted transition-colors text-left">
                <User className="h-4 w-4" />
                <span className="text-sm">Profile</span>
              </button>
              
              <button className="w-full flex items-center space-x-3 p-3 rounded-xl hover:bg-muted transition-colors text-left" onClick={() => {
                window.location.href = '/dashboard';
              }}>
                <FileText className="h-4 w-4" />
                <span className="text-sm">My Documents</span>
              </button>
              
              <button className="w-full flex items-center space-x-3 p-3 rounded-xl hover:bg-muted transition-colors text-left">
                <Settings className="h-4 w-4" />
                <span className="text-sm">Settings</span>
              </button>

              <div className="border-t border-muted my-2" />
              
              <button
                onClick={handleSignOut}
                className="w-full flex items-center space-x-3 p-3 rounded-xl hover:bg-destructive/10 hover:text-destructive transition-colors text-left"
              >
                <LogOut className="h-4 w-4" />
                <span className="text-sm">Sign Out</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
