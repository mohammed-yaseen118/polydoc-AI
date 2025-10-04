import { useEffect, useState, useRef } from 'react';
import { useMotionValue, useSpring, useTransform } from 'framer-motion';

// Hook for scroll progress
export const useScrollProgress = () => {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const updateScrollProgress = () => {
      const scrollTop = window.pageYOffset;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = scrollTop / docHeight;
      setScrollProgress(progress);
    };

    window.addEventListener('scroll', updateScrollProgress);
    return () => window.removeEventListener('scroll', updateScrollProgress);
  }, []);

  return scrollProgress;
};

// Hook for parallax effects
export const useParallax = (speed = 0.5) => {
  const ref = useRef(null);
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      if (!ref.current) return;
      
      const rect = ref.current.getBoundingClientRect();
      const scrolled = window.pageYOffset;
      const rate = scrolled * speed;
      
      setOffset(rate);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [speed]);

  return [ref, offset];
};

// Hook for scroll-triggered reveals
export const useScrollReveal = (threshold = 0.1) => {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [threshold]);

  return [ref, isVisible];
};

// Hook for magnetic effect
export const useMagnetic = (strength = 0.3) => {
  const ref = useRef(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const handleMouseMove = (e) => {
      const rect = element.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      
      const deltaX = (e.clientX - centerX) * strength;
      const deltaY = (e.clientY - centerY) * strength;
      
      setPosition({ x: deltaX, y: deltaY });
    };

    const handleMouseLeave = () => {
      setPosition({ x: 0, y: 0 });
    };

    element.addEventListener('mousemove', handleMouseMove);
    element.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      element.removeEventListener('mousemove', handleMouseMove);
      element.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [strength]);

  return [ref, position];
};

// Hook for smooth scroll animations
export const useSmoothScroll = () => {
  const scrollTo = (target, offset = 0) => {
    const element = document.querySelector(target);
    if (element) {
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;
      
      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  return { scrollTo };
};

// Hook for text reveal animations
export const useTextReveal = () => {
  const ref = useRef(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, []);

  return [ref, isVisible];
};

// Hook for scroll-based scale
export const useScrollScale = () => {
  const scrollY = useMotionValue(0);
  const scale = useTransform(scrollY, [0, 1000], [1, 0.8]);

  useEffect(() => {
    const unsubscribe = scrollY.onChange((latest) => {
      // Handle scroll changes if needed
    });

    const handleScroll = () => {
      scrollY.set(window.scrollY);
    };

    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
      unsubscribe();
    };
  }, [scrollY]);

  return scale;
};
