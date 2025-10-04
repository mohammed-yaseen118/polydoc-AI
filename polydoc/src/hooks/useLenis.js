import { useEffect } from 'react';
import Lenis from 'lenis';

export const useLenis = (options = {}) => {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // Custom easing
      direction: 'vertical', // 'vertical', 'horizontal'
      gestureDirection: 'vertical', // 'vertical', 'horizontal', 'both'
      smooth: true,
      mouseMultiplier: 1,
      smoothTouch: false,
      touchMultiplier: 2,
      infinite: false,
      ...options
    });

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    // Handle anchor links with smooth scrolling
    const handleAnchorClick = (e) => {
      const target = e.target;
      if (target.tagName === 'A' && target.getAttribute('href')?.startsWith('#')) {
        e.preventDefault();
        const id = target.getAttribute('href').substring(1);
        const element = document.getElementById(id);
        if (element) {
          lenis.scrollTo(element, {
            duration: 2,
            easing: (t) => 1 - Math.pow(1 - t, 3) // Smooth easing for anchor scrolling
          });
        }
      }
    };

    document.addEventListener('click', handleAnchorClick);

    return () => {
      lenis.destroy();
      document.removeEventListener('click', handleAnchorClick);
    };
  }, []);
};

export default useLenis;
