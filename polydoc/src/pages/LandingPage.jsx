import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from '@/contexts/ThemeContext';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import {
  ArrowRight,
  FileText,
  Menu,
  X,
  ChevronRight,
  Users,
  Globe,
  Shield,
  Zap,
  Download,
  Mail,
  Github,
  Twitter,
  Linkedin,
  Info,
  Target,
  Layers,
  Workflow
} from "lucide-react";
// Mock AI Chat Interface for Demo (non-authenticated users)
function DemoAIChatInterface() {
  return (
    <div className="w-full max-w-4xl mx-auto h-[600px] bg-gradient-to-br from-slate-900 to-indigo-950 rounded-xl overflow-hidden shadow-2xl border border-indigo-500/20">
      <div className="bg-indigo-600/30 backdrop-blur-sm p-4 border-b border-indigo-500/30 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <FileText className="text-indigo-300 h-5 w-5" />
          <h2 className="text-white font-medium">PolyDoc Demo</h2>
        </div>
      </div>
      
      <div className="p-4 h-[calc(100%-200px)] overflow-y-auto bg-slate-900/50">
        <div className="flex flex-col items-center justify-center h-full text-center">
          <FileText className="h-12 w-12 text-indigo-400 mb-4" />
          <h3 className="text-indigo-200 text-xl mb-2">Welcome to PolyDoc</h3>
          <p className="text-slate-400 text-sm max-w-xs mb-6">
            Upload documents and ask questions about them. I support multiple languages and preserve document layouts!
          </p>
          <div className="space-y-2">
            <Button 
              size="lg" 
              className="bg-indigo-600 hover:bg-indigo-500 text-white rounded-3xl"
              onClick={() => {
                // This will trigger Google Sign-In
                document.dispatchEvent(new CustomEvent('showSignIn'));
              }}
            >
              Sign In to Get Started
            </Button>
            <p className="text-xs text-slate-400">
              Sign in with Google to try the full demo
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Google Sign-In Modal
function GoogleSignInModal({ isOpen, onClose }) {
  const { signInWithGoogle } = useAuth();
  const navigate = useNavigate();
  const { theme } = useTheme();

  const handleGoogleSignIn = async () => {
    try {
      const result = await signInWithGoogle();
      if (result.success) {
        onClose();
        navigate('/dashboard');
      } else {
        console.error('Sign in failed:', result.error);
        alert('Sign in failed: ' + result.error);
      }
    } catch (error) {
      console.error('Sign in failed:', error);
      alert('Sign in failed: ' + error.message);
    }
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="bg-background dark:bg-slate-900 rounded-3xl shadow-2xl border border-muted p-8 max-w-md w-full"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-center space-y-6">
          <div className="flex items-center justify-center space-x-3 mb-6">
                <div className="h-12 w-12 rounded-3xl overflow-hidden ring-2 ring-primary/20">
                  <img 
                    src="/default-avatar.svg" 
                    alt="PolyDoc Logo"
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      // Fallback to icon if image fails to load
                      e.target.style.display = 'none';
                      e.target.nextElementSibling.style.display = 'flex';
                    }}
                  />
                  <div className="hidden w-full h-full bg-primary items-center justify-center">
                    <FileText className="h-6 w-6 text-primary-foreground" />
                  </div>
                </div>
            <h2 className="text-2xl font-bold">PolyDoc</h2>
          </div>
          
          <div className="space-y-4">
            <h3 className="text-xl font-semibold">Welcome Back!</h3>
            <p className="text-muted-foreground">
              Sign in to access your documents and start processing them.
            </p>
          </div>

          <div className="space-y-3">
            <Button 
              onClick={handleGoogleSignIn}
              size="lg" 
              className="w-full rounded-3xl bg-white hover:bg-gray-50 text-black border border-gray-200 flex items-center justify-center gap-3"
            >
              <svg className="h-5 w-5" viewBox="0 0 24 24">
                <path fill="#4285f4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34a853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#fbbc05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#ea4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </Button>

            <div className="text-xs text-muted-foreground space-y-1">
              <p>By signing in, you agree to our Terms of Service and Privacy Policy.</p>
            </div>
          </div>
        </div>

        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 hover:bg-muted rounded-full transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      </motion.div>
    </motion.div>
  );
}

export default function LandingPage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [scrollY, setScrollY] = useState(0);
  const [showSignInModal, setShowSignInModal] = useState(false);
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Scroll handling

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };

    const handleShowSignIn = () => {
      setShowSignInModal(true);
    };

    window.addEventListener("scroll", handleScroll);
    document.addEventListener('showSignIn', handleShowSignIn);
    
    return () => {
      window.removeEventListener("scroll", handleScroll);
      document.removeEventListener('showSignIn', handleShowSignIn);
    };
  }, []);

  // Redirect authenticated users to dashboard
  useEffect(() => {
    if (user) {
      navigate('/dashboard');
    }
  }, [user, navigate]);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const fadeIn = {
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { 
        duration: 0.8,
        ease: [0.25, 0.46, 0.45, 0.94]
      },
    },
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15,
        delayChildren: 0.1
      },
    },
  };

  const itemFadeIn = {
    hidden: { opacity: 0, y: 30, scale: 0.9 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { 
        duration: 0.7,
        ease: [0.25, 0.46, 0.45, 0.94]
      },
    },
  };

  const slideInLeft = {
    hidden: { opacity: 0, x: -60, scale: 0.9 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { 
        duration: 0.8,
        ease: [0.25, 0.46, 0.45, 0.94]
      },
    },
  };

  const slideInRight = {
    hidden: { opacity: 0, x: 60, scale: 0.9 },
    visible: {
      opacity: 1,
      x: 0,
      scale: 1,
      transition: { 
        duration: 0.8,
        ease: [0.25, 0.46, 0.45, 0.94]
      },
    },
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 transition-colors duration-300">
      
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-30 dark:opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
                           radial-gradient(circle at 75% 75%, rgba(168, 85, 247, 0.1) 0%, transparent 50%)`
        }} />
      </div>
      {/* Clean Modern Header */}
      <motion.header
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrollY > 50 
            ? "bg-white/95 dark:bg-gray-900/95 backdrop-blur-md border-b border-gray-200 dark:border-gray-700 shadow-sm" 
            : "bg-transparent"
        }`}
      >
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <motion.div 
              className="flex items-center space-x-3"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 400, damping: 17 }}
            >
              <motion.div
                whileHover={{ rotate: 360 }}
                transition={{ duration: 0.6 }}
                className="w-9 h-9 rounded-xl bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center"
              >
                <FileText className="w-5 h-5 text-white" />
              </motion.div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                PolyDoc
              </span>
            </motion.div>

            {/* Navigation */}
            <nav className="hidden lg:flex items-center space-x-6">
              {["Features", "About"].map((item, index) => (
                <motion.a
                  key={item}
                  href={`#${item.toLowerCase()}`}
                  className="px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                  whileHover={{ y: -1 }}
                >
                  {item}
                </motion.a>
              ))}
            </nav>

            {/* CTA Buttons */}
            <div className="flex items-center space-x-3">
              <ThemeToggle />
              {!user ? (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400"
                    onClick={() => setShowSignInModal(true)}
                  >
                    Sign In
                  </Button>
                  <Button
                    size="sm"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
                    onClick={() => setShowSignInModal(true)}
                  >
                    Get Started
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
                  onClick={() => navigate('/dashboard')}
                >
                  Dashboard
                </Button>
              )}
            </div>

            {/* Mobile menu button */}
            <motion.button 
              className="lg:hidden p-2 text-gray-700 dark:text-gray-300"
              whileTap={{ scale: 0.95 }}
              onClick={toggleMenu}
            >
              <Menu className="w-6 h-6" />
            </motion.button>
          </div>
        </div>
      </motion.header>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-background md:hidden"
        >
          <div className="container flex h-16 items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center space-x-3">
                <div className="h-10 w-10 rounded-3xl bg-primary flex items-center justify-center">
                  <FileText className="h-5 w-5 text-primary-foreground" />
                </div>
                <span className="font-bold text-xl">PolyDoc</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <ThemeToggle className="rounded-3xl" />
              <button onClick={toggleMenu}>
                <X className="h-6 w-6" />
                <span className="sr-only">Close menu</span>
              </button>
            </div>
          </div>
          <motion.nav
            variants={staggerContainer}
            initial="hidden"
            animate="visible"
            className="container grid gap-3 pb-8 pt-6"
          >
            {["Features", "About"].map((item, index) => (
              <motion.div key={index} variants={itemFadeIn}>
                <a
                  href={`#${item.toLowerCase()}`}
                  className="flex items-center justify-between rounded-3xl px-3 py-2 text-lg font-medium hover:bg-accent"
                  onClick={toggleMenu}
                >
                  {item}
                  <ChevronRight className="h-4 w-4" />
                </a>
              </motion.div>
            ))}
            <motion.div variants={itemFadeIn} className="flex flex-col gap-3 pt-4">
              <Button 
                variant="outline" 
                className="w-full rounded-3xl"
                onClick={() => {
                  setShowSignInModal(true);
                  toggleMenu();
                }}
              >
                Sign In
              </Button>
              <Button 
                className="w-full rounded-3xl"
                onClick={() => {
                  setShowSignInModal(true);
                  toggleMenu();
                }}
              >
                Get Started
              </Button>
            </motion.div>
          </motion.nav>
        </motion.div>
      )}

      {/* Main Content */}
      <main className="relative z-10">
        {/* Hero Section - Full Screen */}
        <section className="min-h-screen flex items-center justify-center relative overflow-hidden">
          <div className="container mx-auto px-6 text-center relative z-10">
            
            {/* Hero Content */}
            <div className="max-w-6xl mx-auto">
              
              {/* Badge */}
              <motion.div 
                className="mb-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <motion.div
                  className="inline-flex items-center px-4 py-2 rounded-full bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 text-sm font-medium"
                  whileHover={{ scale: 1.05 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Document Intelligence Platform
                </motion.div>
              </motion.div>

              {/* Main Heading */}
              <div className="mb-8 space-y-4">
                <motion.h1 
                  className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight leading-tight"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, duration: 0.8 }}
                  style={{ lineHeight: '0.9' }}
                >
                  <span className="block bg-gradient-to-r from-gray-900 via-blue-600 to-purple-600 dark:from-white dark:via-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
                    PolyDoc
                  </span>
                </motion.h1>
                
                <motion.h2 
                  className="text-xl md:text-3xl font-light text-gray-600 dark:text-gray-300 tracking-wide"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.8 }}
                >
                  Intelligent Document Processing
                </motion.h2>

                <motion.div 
                  className="w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 mx-auto rounded-full" 
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.7, duration: 0.8 }}
                />
              </div>

              {/* Description */}
              <motion.p 
                className="text-lg md:text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed mb-12"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.8 }}
              >
                Advanced document processing platform with 
                <span className="text-blue-600 dark:text-blue-400 font-medium"> multi-format support</span>, 
                <span className="text-purple-600 dark:text-purple-400 font-medium"> intelligent extraction</span>, and 
                <span className="text-green-600 dark:text-green-400 font-medium"> seamless workflows</span>.
              </motion.p>

              {/* CTA Buttons */}
              <motion.div 
                className="flex flex-col sm:flex-row items-center justify-center gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.8 }}
              >
                <Button
                  size="lg"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg shadow-lg hover:shadow-xl transition-all"
                  onClick={() => setShowSignInModal(true)}
                >
                  Get Started
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
                
                <Button
                  variant="outline"
                  size="lg"
                  className="px-8 py-3 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800"
                  onClick={() => document.querySelector('#features')?.scrollIntoView({ behavior: 'smooth' })}
                >
                  Learn More
                </Button>
              </motion.div>

              {/* Stats */}
              <motion.div 
                className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-16 max-w-4xl mx-auto"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2, duration: 0.8 }}
              >
                <div className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-blue-600 dark:text-blue-400 mb-2">
                    50+
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 text-sm">Languages</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-purple-600 dark:text-purple-400 mb-2">
                    99%
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 text-sm">Accuracy</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-green-600 dark:text-green-400 mb-2">
                    1000+
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 text-sm">Documents</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl md:text-4xl font-bold text-orange-600 dark:text-orange-400 mb-2">
                    24/7
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 text-sm">Available</div>
                </div>
              </motion.div>
            </div>
          </div>

          {/* Scroll Indicator */}
          <motion.div 
            className="absolute bottom-10 left-1/2 -translate-x-1/2"
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <div className="w-6 h-10 border-2 border-gray-400 dark:border-gray-600 rounded-full flex justify-center">
              <motion.div 
                className="w-1 h-3 bg-gray-500 dark:bg-gray-400 rounded-full mt-2"
                animate={{ scaleY: [1, 0.3, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>
          </motion.div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-24 bg-gray-50 dark:bg-gray-800/50">
          <div className="container mx-auto px-6">
            
            {/* Section Header */}
            <div className="text-center mb-16">
              <motion.div
                className="inline-flex items-center px-4 py-2 rounded-full bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 text-blue-600 dark:text-blue-400 text-sm font-medium mb-6"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <FileText className="w-4 h-4 mr-2" />
                Key Features
              </motion.div>
              
              <motion.h2 
                className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                Everything You Need for
                <span className="block bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Document Processing
                </span>
              </motion.h2>
              
              <motion.p 
                className="text-lg text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                Advanced document processing platform with intelligent extraction,
                multi-format support, and seamless workflow integration.
              </motion.p>
            </div>

            {/* Features Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
              {[
                {
                  icon: <FileText className="h-6 w-6" />,
                  title: "Multi-Format Support",
                  description: "Process PDFs, Word docs, PowerPoint, and images with high accuracy.",
                  color: "blue"
                },
                {
                  icon: <Users className="h-6 w-6" />,
                  title: "Collaborative Workflows",
                  description: "Share documents and collaborate with your team in real-time.",
                  color: "purple"
                },
                {
                  icon: <Globe className="h-6 w-6" />,
                  title: "Multi-Language Support",
                  description: "Process documents in 50+ languages with intelligent recognition.",
                  color: "green"
                },
                {
                  icon: <Shield className="h-6 w-6" />,
                  title: "Secure Processing",
                  description: "Enterprise-grade security with encrypted document handling.",
                  color: "red"
                },
                {
                  icon: <Zap className="h-6 w-6" />,
                  title: "Fast Processing",
                  description: "Lightning-fast document analysis with real-time results.",
                  color: "yellow"
                },
                {
                  icon: <Download className="h-6 w-6" />,
                  title: "Easy Export",
                  description: "Export processed documents in multiple formats instantly.",
                  color: "indigo"
                }
              ].map((feature, index) => (
                <motion.div 
                  key={index}
                  className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all duration-300 group"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  whileHover={{ y: -5 }}
                >
                  {/* Feature Icon */}
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300 ${
                    feature.color === 'blue' ? 'bg-blue-100 dark:bg-blue-900/20' :
                    feature.color === 'purple' ? 'bg-purple-100 dark:bg-purple-900/20' :
                    feature.color === 'green' ? 'bg-green-100 dark:bg-green-900/20' :
                    feature.color === 'red' ? 'bg-red-100 dark:bg-red-900/20' :
                    feature.color === 'yellow' ? 'bg-yellow-100 dark:bg-yellow-900/20' :
                    'bg-indigo-100 dark:bg-indigo-900/20'
                  }`}>
                    <div className={`${
                      feature.color === 'blue' ? 'text-blue-600 dark:text-blue-400' :
                      feature.color === 'purple' ? 'text-purple-600 dark:text-purple-400' :
                      feature.color === 'green' ? 'text-green-600 dark:text-green-400' :
                      feature.color === 'red' ? 'text-red-600 dark:text-red-400' :
                      feature.color === 'yellow' ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-indigo-600 dark:text-indigo-400'
                    }`}>
                      {feature.icon}
                    </div>
                  </div>
                  
                  {/* Feature Content */}
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* Bottom CTA */}
            <motion.div 
              className="text-center mt-16"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <Button
                size="lg"
                className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3"
                onClick={() => setShowSignInModal(true)}
              >
                Try All Features
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </motion.div>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="py-24">
          <div className="container mx-auto px-6">
            <div className="max-w-4xl mx-auto text-center">
              <motion.div
                className="inline-flex items-center px-4 py-2 rounded-full bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 text-sm font-medium mb-6"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <Info className="w-4 h-4 mr-2" />
                About PolyDoc
              </motion.div>
              
              <motion.h2 
                className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                Transforming Document Processing
              </motion.h2>
              
              <motion.p 
                className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed mb-8"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                PolyDoc is a modern document intelligence platform that combines advanced OCR technology
                with intelligent analysis capabilities. Our platform supports multiple document formats,
                preserves original layouts, and provides accurate text extraction with comprehensive
                multilingual support.
              </motion.p>
              
              <motion.div 
                className="grid md:grid-cols-3 gap-8 mt-12"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <div className="text-center">
                  <div className="w-16 h-16 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Target className="w-8 h-8 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Accuracy First</h3>
                  <p className="text-gray-600 dark:text-gray-400">High-precision document processing with 99% accuracy rates</p>
                </div>
                
                <div className="text-center">
                  <div className="w-16 h-16 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Layers className="w-8 h-8 text-green-600 dark:text-green-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Format Support</h3>
                  <p className="text-gray-600 dark:text-gray-400">Process PDFs, images, Word docs, and more seamlessly</p>
                </div>
                
                <div className="text-center">
                  <div className="w-16 h-16 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Workflow className="w-8 h-8 text-purple-600 dark:text-purple-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Easy Integration</h3>
                  <p className="text-gray-600 dark:text-gray-400">Simple API and web interface for seamless workflows</p>
                </div>
              </motion.div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
        <div className="container mx-auto px-6 py-12">
          <div className="grid md:grid-cols-4 gap-8">
            {/* Brand */}
            <div className="col-span-2">
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center">
                  <FileText className="w-4 h-4 text-white" />
                </div>
                <span className="font-bold text-xl text-gray-900 dark:text-white">PolyDoc</span>
              </div>
              <p className="text-gray-600 dark:text-gray-400 mb-4 max-w-md">
                Modern document processing platform with intelligent extraction,
                multi-format support, and seamless workflow integration.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                  <Github className="w-5 h-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                  <Twitter className="w-5 h-5" />
                </a>
                <a href="#" className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                  <Linkedin className="w-5 h-5" />
                </a>
              </div>
            </div>
            
            {/* Product */}
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Product</h3>
              <ul className="space-y-2">
                <li><a href="#features" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400">Features</a></li>
                <li><a href="#about" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400">About</a></li>
                <li><a href="#" className="text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400">Documentation</a></li>
              </ul>
            </div>
            
            {/* Technology */}
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Technology</h3>
              <ul className="space-y-2">
                <li><span className="text-gray-600 dark:text-gray-400">Python & NLP</span></li>
                <li><span className="text-gray-600 dark:text-gray-400">OCR Technology</span></li>
                <li><span className="text-gray-600 dark:text-gray-400">Open Source</span></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-200 dark:border-gray-700 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              &copy; {new Date().getFullYear()} PolyDoc. All rights reserved.
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-4 md:mt-0">
              Built with modern web technologies
            </p>
          </div>
        </div>
      </footer>

      {/* Google Sign-In Modal */}
      <GoogleSignInModal 
        isOpen={showSignInModal} 
        onClose={() => setShowSignInModal(false)} 
      />

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
            transform: translateY(8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fade-in {
          animation: fade-in 0.3s ease-out forwards;
        }
        
        .delay-75 {
          animation-delay: 0.2s;
        }
        
        .delay-150 {
          animation-delay: 0.4s;
        }
      `}</style>
    </div>
  );
}
