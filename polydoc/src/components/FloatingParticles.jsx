import React from 'react';
import { motion } from 'framer-motion';

const FloatingParticles = ({ count = 12, className = "" }) => {
  const particles = Array.from({ length: count }, (_, i) => i);

  return (
    <div className={`absolute inset-0 overflow-hidden pointer-events-none ${className}`}>
      {particles.map((particle) => (
        <motion.div
          key={particle}
          className="absolute w-2 h-2 bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-full"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            scale: Math.random() * 0.5 + 0.5,
            opacity: 0
          }}
          animate={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            opacity: [0, 1, 0],
            scale: [0.5, 1, 0.5]
          }}
          transition={{
            duration: Math.random() * 20 + 15,
            repeat: Infinity,
            ease: "linear",
            delay: particle * 2
          }}
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
        />
      ))}
      
      {/* Larger floating elements */}
      {Array.from({ length: 6 }, (_, i) => (
        <motion.div
          key={`large-${i}`}
          className="absolute w-4 h-4 bg-gradient-to-r from-primary/10 to-blue-500/10 rounded-full blur-sm"
          animate={{
            x: [0, Math.random() * 200 - 100],
            y: [0, Math.random() * 200 - 100],
            scale: [1, 1.5, 1],
            opacity: [0.3, 0.7, 0.3]
          }}
          transition={{
            duration: Math.random() * 15 + 10,
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 3
          }}
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
        />
      ))}
    </div>
  );
};

export default FloatingParticles;
