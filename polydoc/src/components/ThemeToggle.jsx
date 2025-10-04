import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { motion } from 'framer-motion';
import { useTheme } from '@/contexts/ThemeContext';
import { Button } from '@/components/ui/button';

export const ThemeToggle = ({ className = "" }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={toggleTheme}
      className={`relative overflow-hidden ${className}`}
      aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
    >
      <motion.div
        initial={false}
        animate={{
          scale: theme === 'light' ? 1 : 0,
          opacity: theme === 'light' ? 1 : 0,
          rotate: theme === 'light' ? 0 : 180,
        }}
        transition={{ 
          duration: 0.3,
          ease: "easeInOut"
        }}
        className="absolute"
      >
        <Sun className="h-4 w-4" />
      </motion.div>
      
      <motion.div
        initial={false}
        animate={{
          scale: theme === 'dark' ? 1 : 0,
          opacity: theme === 'dark' ? 1 : 0,
          rotate: theme === 'dark' ? 0 : -180,
        }}
        transition={{ 
          duration: 0.3,
          ease: "easeInOut"
        }}
        className="absolute"
      >
        <Moon className="h-4 w-4" />
      </motion.div>
    </Button>
  );
};
