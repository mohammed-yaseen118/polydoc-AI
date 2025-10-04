import React from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { FolderOpen, FileText, Sparkles } from 'lucide-react';
import { useParallax, useScrollReveal } from '@/hooks/useScrollAnimations';

const AnimatedLogo3D = () => {
  const [ref, isVisible] = useScrollReveal(0.3);
  const { scrollY } = useScroll();
  
  // Scroll-based transformations
  const rotateX = useTransform(scrollY, [0, 500], [0, 360]);
  const scale = useTransform(scrollY, [0, 300], [1, 0.8]);
  const opacity = useTransform(scrollY, [0, 200], [1, 0.7]);

  return (
    <div ref={ref} className="relative w-full h-full flex items-center justify-center perspective-1000">
      {/* Morphing background shape */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-primary/5 via-purple-500/5 to-blue-500/5 rounded-3xl"
        animate={{
          borderRadius: [
            "60% 40% 30% 70% / 60% 30% 70% 40%",
            "30% 60% 70% 40% / 50% 60% 30% 60%",
            "50% 60% 30% 60% / 60% 40% 30% 70%",
            "60% 40% 30% 70% / 60% 30% 70% 40%"
          ]
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      />
      
      {/* 3D Container with scroll effects */}
      <motion.div
        initial={{ rotateY: -30, rotateX: 15, scale: 0.8, opacity: 0 }}
        animate={isVisible ? { 
          rotateY: 0,
          rotateX: 0,
          scale: 1,
          opacity: 1
        } : {
          rotateY: -30,
          rotateX: 15,
          scale: 0.8,
          opacity: 0
        }}
        style={{
          rotateX,
          scale,
          opacity
        }}
        transition={{
          duration: 1.5,
          ease: [0.25, 0.46, 0.45, 0.94]
        }}
        className="transform-gpu preserve-3d"
      >
        {/* Main Logo Container */}
        <div className="relative group">
          {/* Background Glow Effect */}
          <motion.div
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.7, 0.3]
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute inset-0 bg-gradient-to-r from-primary/30 via-purple-500/30 to-blue-500/30 rounded-3xl blur-xl"
          />
          
          {/* Main Content */}
          <div className="relative bg-gradient-to-br from-background/80 to-muted/60 backdrop-blur-sm rounded-3xl p-8 border border-border/50 shadow-2xl">
            
            {/* Animated Folder Icon */}
            <motion.div
              initial={{ scale: 0, rotateZ: -180 }}
              animate={{ scale: 1, rotateZ: 0 }}
              transition={{ 
                duration: 1.5, 
                type: "spring", 
                stiffness: 200,
                delay: 0.2
              }}
              className="mb-6 flex justify-center"
            >
              <motion.div
                animate={{
                  rotateZ: [0, 10, 0, -10, 0],
                  y: [0, -5, 0, -3, 0]
                }}
                transition={{
                  duration: 6,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="relative"
              >
                <FolderOpen className="w-20 h-20 text-primary drop-shadow-2xl" />
                
                {/* Floating particles around folder */}
                {[...Array(6)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-2 h-2 bg-gradient-to-r from-primary to-purple-500 rounded-full"
                    animate={{
                      x: [0, Math.cos(i * 60 * Math.PI / 180) * 30],
                      y: [0, Math.sin(i * 60 * Math.PI / 180) * 30],
                      opacity: [0, 1, 0],
                      scale: [0, 1, 0]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      delay: i * 0.5,
                      ease: "easeInOut"
                    }}
                    style={{
                      left: '50%',
                      top: '50%',
                      transform: 'translate(-50%, -50%)'
                    }}
                  />
                ))}
              </motion.div>
            </motion.div>
            
            {/* POLYDOC Text with 3D Effect */}
            <div className="text-center">
              <motion.h1
                initial={{ opacity: 0, y: 20, scale: 0.5 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ 
                  duration: 1.2, 
                  type: "spring", 
                  stiffness: 150,
                  delay: 0.5
                }}
                className="relative text-4xl md:text-5xl lg:text-6xl font-bold tracking-wide"
              >
                {/* 3D Text Shadow Layers */}
                <span 
                  className="absolute inset-0 text-primary/20"
                  style={{ transform: 'translate(4px, 4px)' }}
                >
                  POLYDOC
                </span>
                <span 
                  className="absolute inset-0 text-primary/40"
                  style={{ transform: 'translate(2px, 2px)' }}
                >
                  POLYDOC
                </span>
                
                {/* Main Text with Gradient */}
                <motion.span
                  animate={{
                    backgroundPosition: ['0%', '100%', '0%']
                  }}
                  transition={{
                    duration: 8,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  className="relative bg-gradient-to-r from-primary via-purple-500 via-blue-500 to-primary bg-clip-text text-transparent"
                  style={{
                    backgroundSize: '200% 100%'
                  }}
                >
                  POLYDOC
                </motion.span>
              </motion.h1>
              
              {/* Animated Underline */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ 
                  duration: 1,
                  delay: 1.2,
                  ease: "easeOut"
                }}
                className="h-1 bg-gradient-to-r from-primary to-purple-500 rounded-full mt-4 mx-auto"
                style={{ width: '80%' }}
              />
              
              {/* Subtitle */}
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ 
                  duration: 0.8,
                  delay: 1.5
                }}
                className="text-muted-foreground text-lg md:text-xl mt-4 font-medium tracking-wide"
              >
                AI Document Intelligence
              </motion.p>
            </div>
            
            {/* Corner Decorations */}
            {[...Array(4)].map((_, i) => (
              <motion.div
                key={i}
                className={`absolute w-4 h-4 border-2 border-primary/50 ${
                  i === 0 ? 'top-4 left-4 border-b-0 border-r-0' :
                  i === 1 ? 'top-4 right-4 border-b-0 border-l-0' :
                  i === 2 ? 'bottom-4 left-4 border-t-0 border-r-0' :
                  'bottom-4 right-4 border-t-0 border-l-0'
                }`}
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 1, 0.5]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  delay: i * 0.5,
                  ease: "easeInOut"
                }}
              />
            ))}
          </div>
        </div>
      </motion.div>
      
      {/* CSS Styles for 3D effect */}
      <style jsx>{`
        .perspective-1000 {
          perspective: 1000px;
        }
        .preserve-3d {
          transform-style: preserve-3d;
        }
        .transform-gpu {
          transform: translateZ(0);
          backface-visibility: hidden;
        }
      `}</style>
    </div>
  );
};

export default AnimatedLogo3D;
