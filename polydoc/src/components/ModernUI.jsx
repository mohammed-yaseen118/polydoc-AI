import React from 'react';
import { motion } from 'framer-motion';
import { useMagnetic, useScrollProgress, useTextReveal } from '@/hooks/useScrollAnimations';

// Magnetic Button Component
export const MagneticButton = ({ children, className = "", ...props }) => {
  const [ref, position] = useMagnetic(0.2);

  return (
    <motion.button
      ref={ref}
      animate={{ x: position.x, y: position.y }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className={`relative overflow-hidden ${className}`}
      {...props}
    >
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-full"
        initial={{ scale: 0, opacity: 0 }}
        whileHover={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      />
      <span className="relative z-10">{children}</span>
    </motion.button>
  );
};

// Scroll Progress Indicator
export const ScrollProgress = () => {
  const progress = useScrollProgress();

  return (
    <motion.div
      className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-primary to-purple-500 origin-left z-50"
      initial={{ scaleX: 0 }}
      animate={{ scaleX: progress }}
      transition={{ duration: 0.1 }}
    />
  );
};

// Parallax Card Component
export const ParallaxCard = ({ children, speed = 0.5, className = "" }) => {
  const [ref, offset] = useParallax(speed);

  return (
    <motion.div
      ref={ref}
      style={{ y: offset }}
      className={`relative ${className}`}
    >
      {children}
    </motion.div>
  );
};

// Text Reveal Component
export const TextReveal = ({ children, className = "", delay = 0 }) => {
  const [ref, isVisible] = useTextReveal();

  return (
    <div ref={ref} className={`overflow-hidden ${className}`}>
      <motion.div
        initial={{ y: 100, opacity: 0 }}
        animate={isVisible ? { y: 0, opacity: 1 } : { y: 100, opacity: 0 }}
        transition={{ 
          duration: 0.8, 
          delay,
          ease: [0.25, 0.46, 0.45, 0.94]
        }}
      >
        {children}
      </motion.div>
    </div>
  );
};

// Floating Orb Component
export const FloatingOrb = ({ size = "w-96 h-96", color = "from-primary/10 to-purple-500/10", ...props }) => {
  return (
    <motion.div
      animate={{
        x: [0, 100, -50, 0],
        y: [0, -80, 120, 0],
        scale: [1, 1.2, 0.8, 1],
        rotate: [0, 180, 360]
      }}
      transition={{
        duration: 20,
        repeat: Infinity,
        ease: "linear"
      }}
      className={`absolute ${size} bg-gradient-to-r ${color} rounded-full blur-3xl pointer-events-none opacity-30`}
      {...props}
    />
  );
};

// Modern Card with Hover Effects
export const ModernCard = ({ children, className = "", ...props }) => {
  return (
    <motion.div
      whileHover={{ 
        y: -10,
        transition: { type: "spring", stiffness: 300 }
      }}
      className={`
        relative group cursor-pointer
        bg-gradient-to-br from-background/80 to-background/40 
        backdrop-blur-xl border border-white/10 
        rounded-3xl overflow-hidden
        shadow-2xl hover:shadow-4xl
        ${className}
      `}
      {...props}
    >
      {/* Animated border gradient */}
      <motion.div
        className="absolute inset-0 rounded-3xl p-[1px] bg-gradient-to-r from-primary/50 to-purple-500/50 opacity-0 group-hover:opacity-100"
        initial={{ rotate: 0 }}
        whileHover={{ rotate: 180 }}
        transition={{ duration: 0.8 }}
      >
        <div className="w-full h-full rounded-3xl bg-background/90 backdrop-blur-xl" />
      </motion.div>
      
      {/* Content */}
      <div className="relative z-10 p-6">
        {children}
      </div>
      
      {/* Hover glow effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-primary/5 to-purple-500/5 rounded-3xl opacity-0 group-hover:opacity-100"
        transition={{ duration: 0.3 }}
      />
    </motion.div>
  );
};

// Animated Number Counter
export const AnimatedCounter = ({ from = 0, to, duration = 2, suffix = "" }) => {
  const [count, setCount] = React.useState(from);

  React.useEffect(() => {
    const startTime = Date.now();
    const animate = () => {
      const now = Date.now();
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / (duration * 1000), 1);
      
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(from + (to - from) * easeOut);
      
      setCount(current);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    animate();
  }, [from, to, duration]);

  return <span>{count}{suffix}</span>;
};

// Morphing Shape Component
export const MorphingShape = ({ className = "" }) => {
  return (
    <motion.div
      className={`absolute ${className}`}
      animate={{
        borderRadius: [
          "60% 40% 30% 70% / 60% 30% 70% 40%",
          "30% 60% 70% 40% / 50% 60% 30% 60%",
          "50% 60% 30% 60% / 60% 40% 30% 70%",
          "60% 40% 30% 70% / 60% 30% 70% 40%"
        ]
      }}
      transition={{
        duration: 8,
        repeat: Infinity,
        ease: "easeInOut"
      }}
    />
  );
};

// Liquid Button Component
export const LiquidButton = ({ children, className = "", ...props }) => {
  return (
    <motion.button
      className={`
        relative px-8 py-4 rounded-full border border-primary/20 
        bg-gradient-to-r from-primary/10 to-purple-500/10
        overflow-hidden group
        ${className}
      `}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      {...props}
    >
      {/* Liquid background effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-primary to-purple-500"
        initial={{ x: "-100%" }}
        whileHover={{ x: "0%" }}
        transition={{ type: "spring", stiffness: 100 }}
      />
      
      {/* Text */}
      <motion.span
        className="relative z-10 font-medium group-hover:text-white transition-colors duration-300"
        initial={{ color: "currentColor" }}
      >
        {children}
      </motion.span>
    </motion.button>
  );
};
