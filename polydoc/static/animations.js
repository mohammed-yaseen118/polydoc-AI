/**
 * PolyDoc AI - Custom Animation Library
 * Vanilla JavaScript text animations inspired by modern UI libraries
 */

class TextAnimations {
    constructor() {
        this.animations = new Map();
        this.observers = new Map();
        this.setupIntersectionObserver();
    }

    setupIntersectionObserver() {
        this.intersectionObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const animationId = entry.target.dataset.animationId;
                    if (animationId && this.animations.has(animationId)) {
                        const animation = this.animations.get(animationId);
                        if (!animation.triggered) {
                            this.triggerAnimation(animationId);
                        }
                    }
                }
            });
        }, { threshold: 0.3 });
    }

    // Split text animation - letters appear one by one
    splitText(element, options = {}) {
        const defaults = {
            delay: 50,
            duration: 300,
            easing: 'ease-out',
            direction: 'left',
            stagger: true,
            autoPlay: true,
            trigger: 'visible'
        };
        const config = { ...defaults, ...options };

        const text = element.textContent;
        const letters = text.split('');
        element.innerHTML = '';

        // Create span for each letter
        letters.forEach((letter, index) => {
            const span = document.createElement('span');
            span.textContent = letter === ' ' ? '\u00A0' : letter;
            span.style.display = 'inline-block';
            span.style.opacity = '0';
            span.style.transform = this.getInitialTransform(config.direction);
            span.style.transition = `all ${config.duration}ms ${config.easing}`;
            element.appendChild(span);
        });

        const animationId = this.generateId();
        const animation = {
            element,
            letters: element.children,
            config,
            triggered: false
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        if (config.trigger === 'visible') {
            this.intersectionObserver.observe(element);
        } else if (config.autoPlay) {
            this.triggerAnimation(animationId);
        }

        return animationId;
    }

    // Wave text animation - letters move in a wave pattern
    waveText(element, options = {}) {
        const defaults = {
            delay: 100,
            amplitude: 10,
            frequency: 0.5,
            duration: 2000,
            autoPlay: true,
            repeat: true
        };
        const config = { ...defaults, ...options };

        const text = element.textContent;
        const letters = text.split('');
        element.innerHTML = '';

        letters.forEach((letter, index) => {
            const span = document.createElement('span');
            span.textContent = letter === ' ' ? '\u00A0' : letter;
            span.style.display = 'inline-block';
            span.style.transition = 'transform 0.3s ease';
            element.appendChild(span);
        });

        const animationId = this.generateId();
        const animation = {
            element,
            letters: element.children,
            config,
            triggered: false,
            animationFrame: null
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        if (config.autoPlay) {
            this.triggerWaveAnimation(animationId);
        }

        return animationId;
    }

    // Typewriter animation - text appears character by character
    typewriter(element, options = {}) {
        const defaults = {
            delay: 50,
            cursor: true,
            cursorChar: '|',
            autoPlay: true,
            loop: false
        };
        const config = { ...defaults, ...options };

        const text = element.textContent;
        element.textContent = '';

        if (config.cursor) {
            const cursor = document.createElement('span');
            cursor.textContent = config.cursorChar;
            cursor.style.animation = 'blink 1s infinite';
            cursor.className = 'typewriter-cursor';
            element.appendChild(cursor);
        }

        const animationId = this.generateId();
        const animation = {
            element,
            text,
            config,
            triggered: false,
            currentIndex: 0
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        // Add cursor blink animation CSS
        this.addCursorStyles();

        if (config.autoPlay) {
            this.triggerTypewriterAnimation(animationId);
        }

        return animationId;
    }

    // Fade in with slide animation
    fadeInSlide(element, options = {}) {
        const defaults = {
            direction: 'up',
            distance: 30,
            duration: 600,
            delay: 0,
            easing: 'ease-out',
            autoPlay: true,
            trigger: 'visible'
        };
        const config = { ...defaults, ...options };

        element.style.opacity = '0';
        element.style.transform = this.getInitialTransform(config.direction, config.distance);
        element.style.transition = `all ${config.duration}ms ${config.easing} ${config.delay}ms`;

        const animationId = this.generateId();
        const animation = {
            element,
            config,
            triggered: false
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        if (config.trigger === 'visible') {
            this.intersectionObserver.observe(element);
        } else if (config.autoPlay) {
            this.triggerAnimation(animationId);
        }

        return animationId;
    }

    // Scale animation
    scaleIn(element, options = {}) {
        const defaults = {
            scale: 0.8,
            duration: 400,
            delay: 0,
            easing: 'ease-out',
            autoPlay: true,
            trigger: 'visible'
        };
        const config = { ...defaults, ...options };

        element.style.opacity = '0';
        element.style.transform = `scale(${config.scale})`;
        element.style.transition = `all ${config.duration}ms ${config.easing} ${config.delay}ms`;

        const animationId = this.generateId();
        const animation = {
            element,
            config,
            triggered: false
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        if (config.trigger === 'visible') {
            this.intersectionObserver.observe(element);
        } else if (config.autoPlay) {
            this.triggerAnimation(animationId);
        }

        return animationId;
    }

    // Counter animation
    countUp(element, options = {}) {
        const defaults = {
            start: 0,
            end: parseInt(element.textContent) || 100,
            duration: 2000,
            easing: 'ease-out',
            autoPlay: true,
            trigger: 'visible'
        };
        const config = { ...defaults, ...options };

        element.textContent = config.start;

        const animationId = this.generateId();
        const animation = {
            element,
            config,
            triggered: false
        };

        this.animations.set(animationId, animation);
        element.dataset.animationId = animationId;

        if (config.trigger === 'visible') {
            this.intersectionObserver.observe(element);
        } else if (config.autoPlay) {
            this.triggerCountAnimation(animationId);
        }

        return animationId;
    }

    // Trigger animation methods
    triggerAnimation(animationId) {
        const animation = this.animations.get(animationId);
        if (!animation || animation.triggered) return;

        animation.triggered = true;

        if (animation.letters) {
            // Split text or wave animation
            Array.from(animation.letters).forEach((letter, index) => {
                setTimeout(() => {
                    letter.style.opacity = '1';
                    letter.style.transform = 'none';
                }, index * animation.config.delay);
            });
        } else if (animation.text !== undefined) {
            // Typewriter animation
            this.triggerTypewriterAnimation(animationId);
        } else {
            // Fade/scale animations
            setTimeout(() => {
                animation.element.style.opacity = '1';
                animation.element.style.transform = 'none';
            }, 50);
        }
    }

    triggerWaveAnimation(animationId) {
        const animation = this.animations.get(animationId);
        if (!animation) return;

        let startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = elapsed / animation.config.duration;

            Array.from(animation.letters).forEach((letter, index) => {
                const offset = Math.sin((progress * Math.PI * 2 * animation.config.frequency) + (index * 0.5)) * animation.config.amplitude;
                letter.style.transform = `translateY(${offset}px)`;
            });

            if (animation.config.repeat && progress < 1) {
                animation.animationFrame = requestAnimationFrame(animate);
            } else if (animation.config.repeat) {
                startTime = currentTime;
                animation.animationFrame = requestAnimationFrame(animate);
            }
        };

        animation.animationFrame = requestAnimationFrame(animate);
    }

    triggerTypewriterAnimation(animationId) {
        const animation = this.animations.get(animationId);
        if (!animation) return;

        const typeNextChar = () => {
            if (animation.currentIndex < animation.text.length) {
                const currentText = animation.text.substring(0, animation.currentIndex + 1);
                const cursor = animation.element.querySelector('.typewriter-cursor');
                
                if (cursor) {
                    animation.element.textContent = currentText;
                    animation.element.appendChild(cursor);
                } else {
                    animation.element.textContent = currentText;
                }

                animation.currentIndex++;
                setTimeout(typeNextChar, animation.config.delay);
            } else if (animation.config.loop) {
                setTimeout(() => {
                    animation.currentIndex = 0;
                    animation.element.textContent = '';
                    if (animation.config.cursor) {
                        const cursor = document.createElement('span');
                        cursor.textContent = animation.config.cursorChar;
                        cursor.className = 'typewriter-cursor';
                        animation.element.appendChild(cursor);
                    }
                    setTimeout(typeNextChar, 500);
                }, 1000);
            }
        };

        typeNextChar();
    }

    triggerCountAnimation(animationId) {
        const animation = this.animations.get(animationId);
        if (!animation) return;

        const startTime = performance.now();
        const startValue = animation.config.start;
        const endValue = animation.config.end;
        const duration = animation.config.duration;

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const currentValue = Math.round(startValue + (endValue - startValue) * easeProgress);

            animation.element.textContent = currentValue;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Utility methods
    getInitialTransform(direction, distance = 30) {
        switch (direction) {
            case 'up': return `translateY(${distance}px)`;
            case 'down': return `translateY(-${distance}px)`;
            case 'left': return `translateX(${distance}px)`;
            case 'right': return `translateX(-${distance}px)`;
            default: return `translateY(${distance}px)`;
        }
    }

    generateId() {
        return 'anim-' + Math.random().toString(36).substring(2, 15);
    }

    addCursorStyles() {
        if (!document.getElementById('typewriter-styles')) {
            const style = document.createElement('style');
            style.id = 'typewriter-styles';
            style.textContent = `
                @keyframes blink {
                    0%, 50% { opacity: 1; }
                    51%, 100% { opacity: 0; }
                }
                .typewriter-cursor {
                    animation: blink 1s infinite;
                    font-weight: 100;
                }
            `;
            document.head.appendChild(style);
        }
    }

    // Public API methods
    play(animationId) {
        this.triggerAnimation(animationId);
    }

    reset(animationId) {
        const animation = this.animations.get(animationId);
        if (!animation) return;

        animation.triggered = false;
        
        if (animation.letters) {
            Array.from(animation.letters).forEach(letter => {
                letter.style.opacity = '0';
                letter.style.transform = this.getInitialTransform(animation.config.direction);
            });
        } else if (animation.text !== undefined) {
            animation.currentIndex = 0;
            animation.element.textContent = '';
        } else {
            animation.element.style.opacity = '0';
            animation.element.style.transform = this.getInitialTransform(animation.config.direction, animation.config.distance);
        }
    }

    destroy(animationId) {
        const animation = this.animations.get(animationId);
        if (animation) {
            if (animation.animationFrame) {
                cancelAnimationFrame(animation.animationFrame);
            }
            this.intersectionObserver.unobserve(animation.element);
            this.animations.delete(animationId);
        }
    }
}

// Global instance
window.TextAnimations = new TextAnimations();

// Convenience functions
window.animateText = {
    splitText: (element, options) => window.TextAnimations.splitText(element, options),
    waveText: (element, options) => window.TextAnimations.waveText(element, options),
    typewriter: (element, options) => window.TextAnimations.typewriter(element, options),
    fadeInSlide: (element, options) => window.TextAnimations.fadeInSlide(element, options),
    scaleIn: (element, options) => window.TextAnimations.scaleIn(element, options),
    countUp: (element, options) => window.TextAnimations.countUp(element, options),
};
