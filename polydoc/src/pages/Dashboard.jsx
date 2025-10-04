import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FileUp, 
  Paperclip, 
  X, 
  CornerRightUp, 
  Sparkles,
  ArrowLeft,
  Download,
  Share2,
  User,
  ChevronDown
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ThemeToggle } from '@/components/ThemeToggle';
import { UserProfile } from '@/components/ui/user-profile';
import { cn } from '@/lib/utils';
import { useNavigate } from 'react-router-dom';
import { useLenis } from '@/hooks/useLenis';
import FloatingParticles from '@/components/FloatingParticles';

// File Display Component
function FileDisplay({ fileName, onClear }) {
  return (
    <div className="flex items-center gap-2 bg-black/5 dark:bg-white/5 w-fit px-3 py-1 rounded-lg group border dark:border-white/10">
      <FileUp className="w-4 h-4 dark:text-white" />
      <span className="text-sm dark:text-white">{fileName}</span>
      <button
        type="button"
        onClick={onClear}
        className="ml-1 p-0.5 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
      >
        <X className="w-3 h-3 dark:text-white" />
      </button>
    </div>
  );
}

// Custom hooks (same as before)
function useAutoResizeTextarea({ minHeight, maxHeight }) {
  const textareaRef = useRef(null);

  const adjustHeight = (reset) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    if (reset) {
      textarea.style.height = `${minHeight}px`;
      return;
    }

    textarea.style.height = `${minHeight}px`;
    const newHeight = Math.max(
      minHeight,
      Math.min(textarea.scrollHeight, maxHeight ?? Number.POSITIVE_INFINITY)
    );
    textarea.style.height = `${newHeight}px`;
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = `${minHeight}px`;
    }
  }, [minHeight]);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return { textareaRef, adjustHeight };
}

function useFileInput({ accept, maxSize }) {
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState();

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    validateAndSetFile(file);
  };

  const validateAndSetFile = (file) => {
    setError("");

    if (file) {
      if (maxSize && file.size > maxSize * 1024 * 1024) {
        setError(`File size must be less than ${maxSize}MB`);
        return;
      }

      // Check file extension instead of MIME type for better compatibility
      const fileExtension = file.name.toLowerCase().split('.').pop();
      const allowedExtensions = ['pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'];
      
      if (!allowedExtensions.includes(fileExtension)) {
        setError(`File type must be one of: ${allowedExtensions.join(', ')}`);
        return;
      }

      setSelectedFile(file);
      setFileName(file.name);
    }
  };

  const clearFile = () => {
    setFileName("");
    setError("");
    setSelectedFile(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return {
    fileName,
    error,
    fileInputRef,
    handleFileSelect,
    validateAndSetFile,
    clearFile,
    selectedFile,
  };
}

// Upload Input Component - now supports both upload-only and full modes
function UploadInput({
  id = "upload-input",
  placeholder = "Upload documents and ask questions about them...",
  minHeight = 52,
  maxHeight = 200,
  accept = ".pdf,.doc,.docx,.txt,.png,.jpg,.jpeg",
  maxFileSize = 10,
  onSubmit,
  className,
  uploadOnly = false, // New prop to control upload-only mode
  disabled = false // New prop to disable upload
}) {
  const [inputValue, setInputValue] = useState("");
  const { fileName, fileInputRef, handleFileSelect, clearFile, selectedFile, error } =
    useFileInput({ accept, maxSize: maxFileSize });

  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight,
    maxHeight,
  });

  const handleSubmit = () => {
    if (uploadOnly && selectedFile) {
      onSubmit?.("", selectedFile);
      clearFile();
    } else if (!uploadOnly && (inputValue.trim() || selectedFile)) {
      onSubmit?.(inputValue, selectedFile);
      setInputValue("");
      adjustHeight(true);
      clearFile();
    }
  };

  // Upload-only interface
  if (uploadOnly) {
    return (
      <div className={cn("w-full py-2 sm:py-4 px-2 sm:px-0", className)}>
        <div className="relative max-w-2xl w-full mx-auto flex flex-col gap-4">
          {fileName && <FileDisplay fileName={fileName} onClear={clearFile} />}
          
          {error && (
            <div className="text-red-500 text-sm bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded-lg">
              {error}
            </div>
          )}
          
          <div className="flex flex-col gap-3">
            <Button
              onClick={() => !disabled && fileInputRef.current?.click()}
              className={cn(
                "w-full py-6 border-2 border-dashed border-gray-300 dark:border-gray-600 bg-transparent hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-300 rounded-2xl",
                disabled && "opacity-50 cursor-not-allowed hover:bg-transparent"
              )}
              variant="outline"
              disabled={disabled}
            >
              <FileUp className="w-6 h-6 mr-3" />
              <span className="text-base">
                {disabled ? "System initializing..." : "Click to upload document"}
              </span>
            </Button>
            
            <input
              type="file"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept={accept}
              disabled={disabled}
            />
            
            {selectedFile && (
              <Button
                onClick={handleSubmit}
                className="w-full py-3 bg-primary hover:bg-primary/90 text-primary-foreground rounded-2xl"
                disabled={disabled}
              >
                <FileUp className="w-4 h-4 mr-2" />
                Upload {selectedFile.name}
              </Button>
            )}
          </div>
          
          <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Supports: PDF, DOC, DOCX, TXT, PNG, JPG, JPEG (max {maxFileSize}MB)
          </div>
        </div>
      </div>
    );
  }

  // Original full interface with text input and upload
  return (
    <div className={cn("w-full py-2 sm:py-4 px-2 sm:px-0", className)}>
      <div className="relative max-w-2xl w-full mx-auto flex flex-col gap-2">
        {fileName && <FileDisplay fileName={fileName} onClear={clearFile} />}
        
        {error && (
          <div className="text-red-500 text-sm bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded-lg">
            {error}
          </div>
        )}

        <div className="relative">
          <div
            className="absolute left-2 sm:left-3 top-1/2 -translate-y-1/2 flex items-center justify-center h-7 sm:h-8 w-7 sm:w-8 rounded-lg bg-black/5 dark:bg-white/5 hover:cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
          >
            <Paperclip className="w-3.5 sm:w-4 h-3.5 sm:h-4 transition-opacity transform scale-x-[-1] rotate-45 dark:text-white" />
          </div>

          <input
            type="file"
            className="hidden"
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept={accept}
          />

          <Textarea
            id={id}
            placeholder={placeholder}
            className={cn(
              "max-w-2xl bg-black/5 dark:bg-white/5 w-full rounded-2xl sm:rounded-3xl pl-10 sm:pl-12 pr-12 sm:pr-16",
              "placeholder:text-black/70 dark:placeholder:text-white/70",
              "border-none ring-black/30 dark:ring-white/30",
              "text-black dark:text-white text-wrap py-3 sm:py-4",
              "text-sm sm:text-base",
              "max-h-[200px] overflow-y-auto resize-none leading-[1.2]",
              `min-h-[${minHeight}px]`
            )}
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              adjustHeight();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSubmit();
              }
            }}
          />

          <button
            onClick={handleSubmit}
            className="absolute right-2 sm:right-3 top-1/2 -translate-y-1/2 rounded-xl bg-black/5 dark:bg-white/5 py-1 px-1"
            type="button"
          >
            <CornerRightUp
              className={cn(
                "w-3.5 sm:w-4 h-3.5 sm:h-4 transition-opacity dark:text-white",
                (inputValue || selectedFile) ? "opacity-100" : "opacity-30"
              )}
            />
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Dashboard Component
export default function Dashboard() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [currentDocument, setCurrentDocument] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [documentSummaryComplete, setDocumentSummaryComplete] = useState(false);
  const [systemReady, setSystemReady] = useState(false);
  const [initializationStatus, setInitializationStatus] = useState(null);
  const [recentDocuments, setRecentDocuments] = useState([]);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const messagesEndRef = useRef(null);
  const userMenuRef = useRef(null);
  
  // Initialize Lenis smooth scrolling
  useLenis({
    duration: 1.2,
    easing: (t) => 1 - Math.pow(1 - t, 3),
    smooth: true
  });
  
  // Handle sign out
  const handleSignOut = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };
  
  // Handle click outside to close user menu
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Check backend initialization status
  const checkSystemStatus = async () => {
    try {
      // Use absolute URL for backend health check
      const currentPort = window.location.port || '80';
      const backendUrl = `http://localhost:8000`; // Always use port 8000 for backend
      const healthUrl = `${backendUrl}/health`;
      
      console.log(`Frontend port: ${currentPort}, Backend URL: ${backendUrl}`);
      
      console.log('Checking system status at:', healthUrl);
      
      const response = await fetch(healthUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        // Add timeout
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });
      
      if (!response.ok) {
        throw new Error(`Backend responded with status ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Backend health check result:', result);
      
      // Check both possible locations for models_ready flag
      const modelsReady = result.initialization?.models_ready || result.models_ready || false;
      const isHealthy = result.status === 'healthy';
      const initStatus = result.initialization?.status;
      
      console.log('System status details:', {
        isHealthy,
        modelsReady,
        initStatus,
        initProgress: result.initialization?.progress
      });
      
      // System is ready if healthy and models are ready OR initialization status is 'ready'
      const systemIsReady = isHealthy && (modelsReady || initStatus === 'ready');
      setSystemReady(systemIsReady);
      
      console.log('Final system readiness decision:', {
        isHealthy,
        modelsReady,
        initStatus,
        systemIsReady
      });
      
      if (!isHealthy) {
        setInitializationStatus('Backend is not healthy');
      } else if (!systemIsReady) {
        // If system is healthy but not ready, check initialization details
        const initDetails = result.initialization;
        if (initDetails?.status === 'error') {
          setInitializationStatus(`Initialization failed: ${initDetails.error || 'Unknown error'}`);
        } else if (initDetails?.status === 'ready') {
          setInitializationStatus(null); // System is actually ready
        } else {
          setInitializationStatus(initDetails?.message || 'Models are still loading...');
        }
      } else {
        setInitializationStatus(null); // System is ready
      }
      
      return result;
    } catch (error) {
      console.error('System status check failed:', error);
      setSystemReady(false);
      
      if (error.name === 'AbortError' || error.name === 'TimeoutError') {
        setInitializationStatus('Backend connection timeout - please ensure the backend server is running on port 8000');
      } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        setInitializationStatus('Cannot connect to backend - please ensure the backend server is running on port 8000');
      } else {
        setInitializationStatus(`Backend error: ${error.message}`);
      }
      
      return null;
    }
  };

  // Fetch recent documents
  const fetchRecentDocuments = async () => {
    if (!user?.uid) return;
    
    setLoadingRecent(true);
    try {
      const backendUrl = 'http://localhost:8000';
      const documentsUrl = `${backendUrl}/documents`;
      
      const response = await fetch(documentsUrl, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'user-id': user.uid
        },
      });
      
      if (response.ok) {
        const result = await response.json();
        setRecentDocuments(result.documents || []);
      } else {
        console.error('Failed to fetch recent documents:', response.statusText);
      }
    } catch (error) {
      console.error('Error fetching recent documents:', error);
    } finally {
      setLoadingRecent(false);
    }
  };

  // API Functions
  const uploadDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Always use localhost:8000 for backend
      const backendUrl = 'http://localhost:8000';
      const uploadUrl = `${backendUrl}/upload`;
      
      console.log('Uploading document to:', uploadUrl);
      
      const response = await fetch(uploadUrl, {
        method: 'POST',
        headers: {
          'user-id': user?.uid || 'anonymous'
        },
        body: formData,
      });
      
      if (response.status === 503) {
        throw new Error('System is still initializing. Please wait a moment and try again.');
      }
      
      if (!response.ok) {
        console.log(`HTTP Error: ${response.status} ${response.statusText}`);
        let errorMessage = `Upload failed: ${response.statusText} (${response.status})`;
        
        try {
          const errorData = await response.json();
          console.log('Raw backend error response:', errorData);
          console.log('Error data type:', typeof errorData);
          console.log('Error data stringified:', JSON.stringify(errorData, null, 2));
          
          // Try different ways to extract the error message
          if (typeof errorData === 'string') {
            errorMessage = errorData;
          } else if (errorData && typeof errorData === 'object') {
            if (errorData.detail) {
              errorMessage = Array.isArray(errorData.detail) 
                ? errorData.detail.map(d => typeof d === 'object' ? JSON.stringify(d) : d).join(', ')
                : String(errorData.detail);
            } else if (errorData.message) {
              errorMessage = String(errorData.message);
            } else if (errorData.error) {
              errorMessage = String(errorData.error);
            } else {
              // If it's an object but no standard error fields, stringify the whole thing
              errorMessage = `Upload failed (${response.status}): ` + JSON.stringify(errorData, null, 2);
            }
          } else {
            errorMessage = `Upload failed: ${response.statusText} (${response.status}) - Could not parse error response`;
          }
        } catch (parseError) {
          console.log('Failed to parse error response:', parseError);
          // Try to get response as text if JSON parsing fails
          try {
            const errorText = await response.text();
            console.log('Error response as text:', errorText);
            errorMessage = `Upload failed (${response.status}): ${errorText || response.statusText}`;
          } catch (textError) {
            errorMessage = `Upload failed: ${response.statusText} (${response.status}) - Could not read error response`;
          }
        }
        
        console.log('Final error message:', errorMessage);
        throw new Error(errorMessage);
      }
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Upload error:', error);
      throw error;
    }
  };
  
  const sendChatMessage = async (message, documentId = null) => {
    try {
      // Always use localhost:8000 for backend
      const backendUrl = 'http://localhost:8000';
      const chatUrl = `${backendUrl}/chat`;
      
      console.log('Sending chat message to:', chatUrl);
      
      const response = await fetch(chatUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'user-id': user?.uid || 'anonymous'
        },
        body: JSON.stringify({
          message,
          user_id: user?.uid || 'anonymous',
          document_id: documentId,
          language: 'en'
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Chat failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Chat error:', error);
      throw error;
    }
  };

  const handleSubmit = async (message, file) => {
    if (message.trim() === "" && !file) return;
    
    // BYPASS MODE: Allow upload even when backend is not responding for testing
    // This will show proper error messages but won't block the upload attempt
    console.log('Upload attempt with systemReady:', systemReady);
    
    // Handle file upload
    if (file) {
      setIsUploading(true);
      try {
        // Check system status right before upload
        await checkSystemStatus();
        
        // Add upload message with progress
        const uploadMessage = `Uploading document: ${file.name}`;
        setMessages((prev) => [...prev, { 
          text: uploadMessage, 
          isUser: true, 
          timestamp: new Date() 
        }]);
        
        // Simulate upload progress
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + Math.random() * 20;
          });
        }, 500);
        
        // Upload the document
        const uploadResult = await uploadDocument(file);
        
        // Complete progress
        clearInterval(progressInterval);
        setUploadProgress(100);
        setCurrentDocument({
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified,
          document_id: uploadResult.document_id,
          summary: uploadResult.summary,
          statistics: uploadResult.statistics,
          filename: uploadResult.filename
        });
        
        // Add success message with better formatting
        const successMessage = `✅ Document uploaded successfully!`;
        const summaryMessage = uploadResult.summary ? `📄 Summary: ${uploadResult.summary}` : 'Document processed successfully.';
        
        setMessages((prev) => [...prev, 
          { 
            text: successMessage, 
            isUser: false, 
            timestamp: new Date(),
            type: 'success'
          },
          {
            text: summaryMessage,
            isUser: false,
            timestamp: new Date(),
            type: 'summary'
          }
        ]);
        
        // If there's a message, send it as chat
        if (message.trim()) {
          await handleChatMessage(message, uploadResult.document_id);
        }
        
        // Refresh recent documents after successful upload
        await fetchRecentDocuments();
      } catch (error) {
        let errorMessage = `Γ¥î Upload failed: ${error.message}`;
        
        // Check for specific backend issues and provide helpful messages
        if (error.message.includes('database must be an instance of Database')) {
          errorMessage = `Γ¥î Upload failed: Backend database configuration error. The backend server needs to be restarted or reconfigured.\n\nTechnical details: ${error.message}`;
        } else if (error.message.includes('500') && error.message.includes('Internal Server Error')) {
          errorMessage = `Γ¥î Upload failed: Internal server error. Please check if the backend server is properly configured and running.\n\nTechnical details: ${error.message}`;
        }
        
        setMessages((prev) => [...prev, { 
          text: errorMessage, 
          isUser: false, 
          timestamp: new Date() 
        }]);
      } finally {
        setIsUploading(false);
        setUploadProgress(0); // Reset progress
      }
    } else if (message.trim()) {
      // Handle chat message only
      const documentId = currentDocument?.document_id || null;
      await handleChatMessage(message, documentId);
    }
  };
  
  const handleChatMessage = async (message, documentId) => {
    // Add user message
    setMessages((prev) => [...prev, { 
      text: message, 
      isUser: true, 
      timestamp: new Date() 
    }]);
    
    setIsTyping(true);
    
    try {
      const chatResult = await sendChatMessage(message, documentId);
      
      // Add AI response
      setMessages((prev) => [...prev, { 
        text: chatResult.response, 
        isUser: false, 
        timestamp: new Date(),
        confidence: chatResult.confidence,
        sources: chatResult.sources 
      }]);
    } catch (error) {
      const errorMessage = `Γ¥î Error: ${error.message}`;
      setMessages((prev) => [...prev, { 
        text: errorMessage, 
        isUser: false, 
        timestamp: new Date() 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentDocument(null);
  };
  
  const handleDownload = async () => {
    if (!currentDocument) return;
    
    try {
      // Create a simple text file with document info
      const documentInfo = {
        name: currentDocument.name,
        size: currentDocument.size,
        uploadDate: new Date().toISOString(),
        summary: currentDocument.summary,
        statistics: currentDocument.statistics
      };
      
      const content = `Document Information:

Name: ${documentInfo.name}
Size: ${(documentInfo.size / 1024 / 1024).toFixed(2)} MB
Uploaded: ${new Date(documentInfo.uploadDate).toLocaleString()}

Summary:
${documentInfo.summary || 'No summary available'}

Statistics:
${JSON.stringify(documentInfo.statistics, null, 2)}`;
      
      const blob = new Blob([content], { type: 'text/plain' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentDocument.name}-info.txt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download error:', error);
    }
  };
  
  const handleShare = async () => {
    if (!currentDocument) return;
    
    try {
      const shareText = `I just processed "${currentDocument.name}" with PolyDoc AI!\n\nSummary: ${currentDocument.summary?.substring(0, 200)}...`;
      
      if (navigator.share) {
        await navigator.share({
          title: 'PolyDoc AI Document Analysis',
          text: shareText,
          url: window.location.href
        });
      } else {
        // Fallback: copy to clipboard
        await navigator.clipboard.writeText(shareText);
        // You could show a toast notification here
        console.log('Share text copied to clipboard!');
      }
    } catch (error) {
      console.error('Share error:', error);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check system status on component mount and periodically
  useEffect(() => {
    checkSystemStatus();
    
    // Check every 5 seconds if system is not ready
    const interval = setInterval(() => {
      if (!systemReady) {
        checkSystemStatus();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [systemReady]);
  
  // Fetch recent documents when component mounts and when system is ready
  useEffect(() => {
    if (systemReady && user?.uid) {
      fetchRecentDocuments();
    }
  }, [systemReady, user?.uid]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      {/* Floating Background Particles */}
      <FloatingParticles count={8} />
      
      {/* Background Gradient Orb */}
      <motion.div
        animate={{
          scale: [1, 1.1, 1],
          rotate: [0, 90, 180, 270, 360],
          opacity: [0.05, 0.15, 0.05]
        }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: "linear"
        }}
        className="absolute top-1/4 right-1/4 w-96 h-96 bg-gradient-to-r from-primary/10 to-purple-500/10 rounded-full blur-3xl pointer-events-none"
      />
      {/* Modern Header */}
      <header className="sticky top-0 z-50 w-full border-b border-gray-200 dark:border-gray-700 bg-white/95 dark:bg-gray-800/95 backdrop-blur-xl shadow-sm">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-4">
            <motion.button
              onClick={() => navigate('/')}
              className="p-2 text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ArrowLeft className="h-5 w-5" />
            </motion.button>
            
            <motion.div 
              className="flex items-center space-x-3"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                <FileUp className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="font-bold text-xl text-gray-900 dark:text-white">PolyDoc</span>
                <p className="text-xs text-gray-500 dark:text-gray-400">Document Processing</p>
              </div>
            </motion.div>
          </div>
          
          <div className="flex items-center gap-4">
            <ThemeToggle />
            
            <div className="relative" ref={userMenuRef}>
              <motion.button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-2 px-3 py-2 bg-gray-100 dark:bg-gray-700 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                whileHover={{ scale: 1.02 }}
              >
                <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
                  {user?.photoURL ? (
                    <img 
                      src={user.photoURL} 
                      alt={user.displayName} 
                      className="w-8 h-8 rounded-full"
                    />
                  ) : (
                    <User className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                  )}
                </div>
                <span className="text-sm text-gray-700 dark:text-gray-300 hidden sm:block">
                  {user?.displayName?.split(' ')[0] || user?.email}
                </span>
                <ChevronDown className="w-4 h-4 text-gray-500 dark:text-gray-400" />
              </motion.button>
              
              {/* Dropdown Menu */}
              {showUserMenu && (
                <motion.div
                  initial={{ opacity: 0, y: -10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  transition={{ duration: 0.15 }}
                  className="absolute right-0 top-full mt-2 w-56 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 z-50 overflow-hidden"
                >
                  <div className="p-4 border-b border-gray-100 dark:border-gray-700">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
                        {user?.photoURL ? (
                          <img 
                            src={user.photoURL} 
                            alt={user.displayName} 
                            className="w-10 h-10 rounded-full"
                          />
                        ) : (
                          <User className="w-5 h-5 text-gray-600 dark:text-gray-400" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {user?.displayName || 'User'}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                          {user?.email}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="p-2">
                    <button
                      onClick={handleSignOut}
                      className="w-full text-left px-3 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors flex items-center space-x-2"
                    >
                      <ArrowLeft className="w-4 h-4" />
                      <span>Sign out</span>
                    </button>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Connection Status Banner */}
      {initializationStatus === 'Backend connection failed' && (
        <div className="bg-red-100 dark:bg-red-900/20 border-l-4 border-red-500 p-4 mx-4 mt-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <X className="h-5 w-5 text-red-400" />
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700 dark:text-red-300">
                Unable to connect to backend server. Please ensure the backend is running on port 8000.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto max-w-7xl py-8 px-6 relative">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="grid gap-6 lg:grid-cols-3"
        >
          {/* Chat Interface */}
          <motion.div 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="lg:col-span-2"
          >
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl overflow-hidden shadow-lg">
              {/* Chat Header */}
              <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                    <Sparkles className="text-blue-600 dark:text-blue-400 h-4 w-4" />
                  </div>
                  <h2 className="font-medium text-gray-900 dark:text-white">Document Assistant</h2>
                  
                  {/* System Status Indicator */}
                  {!systemReady && initializationStatus && (
                    <div className="ml-3 px-2 py-1 bg-yellow-100 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300 text-xs rounded-full">
                      {initializationStatus}
                    </div>
                  )}
                  
                  {systemReady && (
                    <div className="ml-3 px-2 py-1 bg-green-100 dark:bg-green-900/20 text-green-700 dark:text-green-300 text-xs rounded-full">
                      System Ready
                    </div>
                  )}
                  
                  {currentDocument && (
                    <div className="ml-3 px-2 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs rounded-full">
                      Document Ready
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="ghost" size="sm" onClick={clearChat}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {/* Messages */}
              <div className="p-4 h-[400px] overflow-y-auto">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <Sparkles className="h-12 w-12 text-primary mb-4" />
                    <h3 className="text-primary text-xl mb-2">Welcome back, {user?.displayName?.split(' ')[0]}!</h3>
                    <p className="text-muted-foreground text-sm max-w-xs">
                      Upload your documents below and start asking questions. I'll help you understand and analyze them!
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                      {messages.map((msg, index) => (
                        <div
                          key={index}
                          className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
                        >
                          <div
                            className={`max-w-[80%] p-4 rounded-2xl animate-fade-in ${
                              msg.isUser
                                ? "bg-blue-600 text-white rounded-tr-none shadow-lg"
                                : msg.type === 'success'
                                ? "bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300 border border-green-200 dark:border-green-700 rounded-tl-none"
                                : msg.type === 'summary'
                                ? "bg-blue-50 dark:bg-blue-900/20 text-blue-800 dark:text-blue-300 border border-blue-200 dark:border-blue-700 rounded-tl-none shadow-sm"
                                : "bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-none"
                            }`}
                          >
                            <p className={`text-sm leading-relaxed ${
                              msg.type === 'summary' ? 'whitespace-pre-wrap' : ''
                            }`}>{msg.text}</p>
                            {msg.timestamp && (
                              <p className="text-xs opacity-70 mt-2">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </p>
                            )}
                          </div>
                        </div>
                      ))}
                    {isUploading && (
                      <div className="flex justify-start">
                        <div className="max-w-[80%] p-3 rounded-2xl bg-blue-100 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 rounded-tl-none">
                          <div className="space-y-2">
                            <div className="flex items-center space-x-2">
                              <FileUp className="w-4 h-4 animate-pulse" />
                              <span className="text-sm font-medium">Processing document...</span>
                            </div>
                            <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2">
                              <div 
                                className="bg-blue-600 dark:bg-blue-400 h-2 rounded-full transition-all duration-500 ease-out"
                                style={{ width: `${uploadProgress}%` }}
                              ></div>
                            </div>
                            <div className="text-xs text-blue-600 dark:text-blue-400">
                              {Math.round(uploadProgress)}% complete
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    {isTyping && (
                      <div className="flex justify-start">
                        <div className="max-w-[80%] p-3 rounded-2xl bg-muted text-foreground rounded-tl-none">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse delay-75"></div>
                            <div className="w-2 h-2 rounded-full bg-primary animate-pulse delay-150"></div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>

              {/* Input */}
              <div className="border-t border-black/5 dark:border-white/5 bg-gradient-to-t from-muted/20 to-background/50">
                <UploadInput 
                  onSubmit={handleSubmit}
                  placeholder={!currentDocument ? "Upload a document to start chatting..." : "Ask questions about your document..."}
                  uploadOnly={!currentDocument} // Show upload-only mode until document is uploaded
                  disabled={false} // Allow upload for testing even when system not ready
                />
              </div>
            </div>
          </motion.div>

          {/* Sidebar */}
          <motion.div 
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.4 }}
            className="space-y-6"
          >
            {/* Current Document */}
            {currentDocument && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
                className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl p-6 shadow-lg"
              >
                <h3 className="font-medium mb-3">Current Document</h3>
                <div className="flex items-center space-x-3 p-3 bg-gradient-to-r from-primary/10 to-purple-500/10 rounded-xl border border-primary/20">
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  >
                    <FileUp className="h-8 w-8 text-primary" />
                  </motion.div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{currentDocument.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(currentDocument.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <div className="flex gap-2 mt-3">
                  <Button size="sm" variant="outline" className="flex-1" onClick={handleDownload}>
                    <Download className="h-4 w-4 mr-2" />
                    Download
                  </Button>
                  <Button size="sm" variant="outline" className="flex-1" onClick={handleShare}>
                    <Share2 className="h-4 w-4 mr-2" />
                    Share
                  </Button>
                </div>
              </motion.div>
            )}

            {/* Recent Documents */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-2xl p-6 shadow-lg"
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-medium">Recent Documents</h3>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={fetchRecentDocuments}
                  className="text-xs"
                  disabled={loadingRecent}
                >
                  {loadingRecent ? 'Loading...' : 'Refresh'}
                </Button>
              </div>
              <div className="space-y-2">
                {loadingRecent ? (
                  <div className="text-sm text-muted-foreground text-center py-4">
                    Loading documents...
                  </div>
                ) : recentDocuments.length === 0 ? (
                  <div className="text-sm text-muted-foreground text-center py-4">
                    No recent documents yet
                  </div>
                ) : (
                  recentDocuments.slice(0, 5).map((doc, index) => (
                    <motion.div
                      key={doc.document_id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.2, delay: index * 0.1 }}
                      className="flex items-center space-x-3 p-2 hover:bg-muted/50 rounded-xl cursor-pointer transition-colors"
                      onClick={() => {
                        setCurrentDocument({
                          name: doc.filename,
                          document_id: doc.document_id,
                          size: 0, // Size not available in response
                          upload_date: doc.upload_date
                        });
                        clearChat(); // Clear current chat when switching documents
                      }}
                    >
                      <FileUp className="h-4 w-4 text-primary flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{doc.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {doc.page_count || 0} pages ΓÇó {doc.language || 'Unknown'}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(doc.upload_date).toLocaleDateString()}
                        </p>
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      </main>
    </div>
  );
}
