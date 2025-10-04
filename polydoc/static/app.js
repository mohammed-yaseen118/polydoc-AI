/**
 * PolyDoc AI - Frontend JavaScript Application
 * Handles document upload, real-time chat, and user interface interactions
 */

class PolyDocApp {
    constructor() {
        this.baseURL = window.location.origin;
        this.websocket = null;
        this.currentDocumentId = null;
        this.selectedLanguage = 'en';
        this.isProcessing = false;
        this.userId = this.getUserId();
        
        // Initialize the application
        this.init();
    }
    
    getUserId() {
        // Get or create a user ID for this session
        let userId = localStorage.getItem('polydoc_user_id');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
            localStorage.setItem('polydoc_user_id', userId);
        }
        return userId;
    }
    
    init() {
        this.bindEvents();
        this.showLoadingOverlay("Initializing PolyDoc AI...", "Connecting to server...");
        
        // IMMEDIATE FALLBACK: Force hide loading overlay after just 3 seconds
        setTimeout(() => {
            console.warn('FORCE HIDING LOADING OVERLAY AFTER 3 SECONDS');
            this.forceHideLoadingOverlay();
        }, 3000);
        
        // BACKUP FALLBACK: Another attempt after 5 seconds 
        setTimeout(() => {
            console.warn('BACKUP FORCE HIDING LOADING OVERLAY AFTER 5 SECONDS');
            this.forceHideLoadingOverlay();
        }, 5000);
        
        this.checkSystemHealth();
        this.loadDocuments();
        this.initializeAnimations();
    }
    
    bindEvents() {
        // File upload events
        const fileInput = document.getElementById('file-input');
        const selectFileBtn = document.getElementById('select-file-btn');
        const uploadArea = document.getElementById('upload-area');
        
        selectFileBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));
        
        // Document management events
        document.getElementById('refresh-docs-btn').addEventListener('click', () => this.loadDocuments());
        
        // Chat events
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        document.getElementById('send-chat-btn').addEventListener('click', () => this.sendMessage());
        document.getElementById('clear-chat-btn').addEventListener('click', () => this.clearChat());
        document.getElementById('close-chat-btn').addEventListener('click', () => this.closeChat());
        
        // Language selection
        document.getElementById('language-select').addEventListener('change', (e) => {
            this.selectedLanguage = e.target.value;
        });
        
        // Modal events
        document.getElementById('modal-close').addEventListener('click', () => this.closeModal());
        document.getElementById('stats-modal-close').addEventListener('click', () => this.closeStatsModal());
        document.getElementById('stats-btn').addEventListener('click', () => this.showSystemStats());
        
        // Close modals on outside click
        document.getElementById('document-modal').addEventListener('click', (e) => {
            if (e.target.id === 'document-modal') this.closeModal();
        });
        document.getElementById('stats-modal').addEventListener('click', (e) => {
            if (e.target.id === 'stats-modal') this.closeStatsModal();
        });
    }
    
    // Model Loading Progress
    startModelLoadingProgress() {
        const loadingSteps = [
            "Loading embedding model...",
            "Loading summarization model...",
            "Loading QA model...",
            "Loading classification model...",
            "Loading language detection model...",
            "Initializing vector store...",
            "System ready!"
        ];
        
        let step = 0;
        const progressInterval = setInterval(() => {
            if (step < loadingSteps.length - 1) {
                this.updateLoadingOverlay("Initializing PolyDoc AI...", loadingSteps[step]);
                step++;
            } else {
                clearInterval(progressInterval);
            }
        }, 3000); // Update every 3 seconds
        
        this.modelProgressInterval = progressInterval;
    }
    
    updateLoadingOverlay(title, subtitle, progress = null) {
        const overlay = document.getElementById('loading-overlay');
        const titleEl = overlay.querySelector('.loading-text');
        const subtitleEl = overlay.querySelector('.loading-subtext');
        const loadingTitle = overlay.querySelector('.loading-title');
        
        titleEl.textContent = subtitle || title;
        subtitleEl.textContent = progress !== null ? `${progress}% complete` : "This may take a few moments on first load";
        
        // Add or update progress bar
        let progressBar = overlay.querySelector('.loading-progress-bar');
        if (!progressBar && progress !== null) {
            progressBar = document.createElement('div');
            progressBar.className = 'loading-progress-bar';
            progressBar.innerHTML = `
                <div class="loading-progress-track">
                    <div class="loading-progress-fill" style="width: 0%"></div>
                </div>
            `;
            
            // Add CSS for progress bar
            const style = document.createElement('style');
            style.textContent = `
                .loading-progress-bar {
                    margin: 2rem 0;
                }
                .loading-progress-track {
                    width: 100%;
                    height: 6px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 3px;
                    overflow: hidden;
                }
                .loading-progress-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #10b981, #6366f1);
                    border-radius: 3px;
                    transition: width 0.5s ease;
                }
            `;
            document.head.appendChild(style);
            
            const dotsContainer = overlay.querySelector('.loading-progress-dots');
            dotsContainer.parentNode.insertBefore(progressBar, dotsContainer);
        }
        
        // Update progress bar
        if (progressBar && progress !== null) {
            const progressFill = progressBar.querySelector('.loading-progress-fill');
            if (progressFill) {
                progressFill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
            }
        }
        
        // Add typewriter animation to the loading text
        try {
            if (window.animateText && titleEl && subtitle) {
                // Only animate if text changed
                if (titleEl.textContent !== subtitle) {
                    titleEl.innerHTML = '';
                    titleEl.textContent = subtitle;
                }
            }
        } catch (error) {
            console.warn('Loading animation failed:', error);
        }
    }
    
    // Animation Initialization
    initializeAnimations() {
        // Wait a bit for DOM to be ready
        setTimeout(() => {
            try {
                // Header title animation
                const logo = document.querySelector('.logo');
                if (logo) {
                    animateText.splitText(logo, { 
                        delay: 100, 
                        direction: 'left',
                        trigger: 'immediate',
                        autoPlay: true
                    });
                }

                // Upload section animations
                const uploadHeader = document.querySelector('.upload-header h2');
                if (uploadHeader) {
                    animateText.fadeInSlide(uploadHeader, {
                        direction: 'up',
                        delay: 300
                    });
                }

                const uploadText = document.querySelector('.upload-text');
                if (uploadText) {
                    animateText.typewriter(uploadText, {
                        delay: 80,
                        autoPlay: false,
                        trigger: 'visible'
                    });
                }

                // Section headers with slide animations
                const sectionHeaders = document.querySelectorAll('.section-header h2');
                sectionHeaders.forEach((header, index) => {
                    animateText.fadeInSlide(header, {
                        direction: 'right',
                        delay: index * 200
                    });
                });

                // Welcome message animation
                const welcomeMessage = document.querySelector('.welcome-message p');
                if (welcomeMessage) {
                    animateText.typewriter(welcomeMessage, {
                        delay: 50,
                        cursor: false,
                        autoPlay: false,
                        trigger: 'visible'
                    });
                }

                // Statistics values animation (when they appear)
                this.animateStatsOnLoad();

            } catch (error) {
                console.warn('Animation initialization failed:', error);
            }
        }, 500);
    }

    animateStatsOnLoad() {
        // This will be called when stats are loaded
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    // Look for newly added stat values
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === 1) { // Element node
                            const statValues = node.querySelectorAll('.stat-value');
                            statValues.forEach((statValue, index) => {
                                if (statValue.textContent.match(/^\d+$/)) {
                                    // It's a number, animate it
                                    setTimeout(() => {
                                        animateText.countUp(statValue, {
                                            duration: 1500,
                                            delay: index * 200
                                        });
                                    }, 200);
                                }
                            });
                        }
                    });
                }
            });
        });

        // Watch for changes in modal bodies where stats appear
        const modalBody = document.getElementById('stats-modal-body');
        if (modalBody) {
            observer.observe(modalBody, { childList: true, subtree: true });
        }
    }
    
    // System Health and Initialization
    async checkSystemHealth() {
        try {
            // Set a maximum timeout for initialization (30 seconds)
            this.maxInitTimeout = setTimeout(() => {
                console.warn('Initialization timeout - force hiding loading overlay');
                this.showNotification('⚠️ Initialization taking longer than expected - loading anyway', 'warning');
                this.hideLoadingOverlay();
            }, 30000); // 30 second timeout
            
            // Start polling initialization status
            this.pollInitializationStatus();
        } catch (error) {
            this.showNotification('Failed to connect to server', 'error');
            this.hideLoadingOverlay();
            console.error('Health check failed:', error);
        }
    }
    
    async pollInitializationStatus() {
        try {
            console.log('Polling initialization status...');
            const response = await fetch(`${this.baseURL}/initialization-status`);
            const status = await response.json();
            console.log('Initialization status:', status);
            
            // Update loading overlay with real progress
            this.updateLoadingOverlay(
                "Initializing PolyDoc AI...", 
                status.message,
                status.progress
            );
            
            if (status.status === 'ready') {
                console.log('System is ready, hiding loading overlay...');
                this.showNotification('🎉 PolyDoc AI is ready!', 'success');
                this.hideLoadingOverlay();
                // Clear any existing polling and timeout
                if (this.modelProgressInterval) {
                    clearInterval(this.modelProgressInterval);
                }
                if (this.maxInitTimeout) {
                    clearTimeout(this.maxInitTimeout);
                }
                return; // Stop polling
            } else if (status.status === 'error') {
                this.showNotification(`❌ Initialization failed: ${status.error}`, 'error');
                this.updateLoadingOverlay(
                    "Initialization Failed", 
                    status.message || "Please refresh the page to try again",
                    0
                );
                // Clear any existing polling and timeout
                if (this.modelProgressInterval) {
                    clearInterval(this.modelProgressInterval);
                }
                if (this.maxInitTimeout) {
                    clearTimeout(this.maxInitTimeout);
                }
                return; // Stop polling
            } else {
                // Continue polling every 2 seconds
                setTimeout(() => this.pollInitializationStatus(), 2000);
            }
        } catch (error) {
            console.error('Failed to check initialization status:', error);
            this.showNotification('Failed to connect to server', 'error');
            this.hideLoadingOverlay();
        }
    }
    
    // File Upload Handling
    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('upload-area').classList.add('drag-over');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('upload-area').classList.remove('drag-over');
    }
    
    handleFileDrop(e) {
        e.preventDefault();
        document.getElementById('upload-area').classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.uploadFile(files[0]);
        }
    }
    
    async uploadFile(file) {
        if (this.isProcessing) {
            this.showNotification('Another file is being processed. Please wait.', 'warning');
            return;
        }
        
        // Validate file type
        const allowedTypes = ['.pdf', '.docx', '.pptx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            this.showNotification(`Unsupported file type: ${fileExtension}`, 'error');
            return;
        }
        
        // Validate file size (max 50MB)
        const maxSize = 50 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('File size exceeds 50MB limit', 'error');
            return;
        }
        
        this.isProcessing = true;
        this.showUploadProgress(`Processing ${file.name}...`);
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${this.baseURL}/upload`, {
                method: 'POST',
                headers: {
                    'user_id': this.userId
                },
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Upload failed');
            }
            
            const result = await response.json();
            this.hideUploadProgress();
            
            this.showNotification(`Document processed successfully! (${result.processing_time.toFixed(2)}s)`, 'success');
            
            // Show document summary modal
            this.showDocumentDetails(result);
            
            // Refresh documents list
            this.loadDocuments();
            
            // Clear file input
            document.getElementById('file-input').value = '';
            
        } catch (error) {
            this.hideUploadProgress();
            this.showNotification(`Upload failed: ${error.message}`, 'error');
            console.error('Upload error:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    showUploadProgress(text) {
        document.getElementById('upload-area').style.display = 'none';
        document.getElementById('upload-progress').hidden = false;
        document.getElementById('progress-text').textContent = text;
        
        // Simulate progress animation
        const progressFill = document.getElementById('progress-fill');
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            progressFill.style.width = progress + '%';
        }, 500);
        
        // Store interval for cleanup
        this.progressInterval = interval;
    }
    
    hideUploadProgress() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        document.getElementById('upload-area').style.display = 'block';
        document.getElementById('upload-progress').hidden = true;
        document.getElementById('progress-fill').style.width = '0%';
    }
    
    // Document Management
    async loadDocuments() {
        try {
            const response = await fetch(`${this.baseURL}/documents`, {
                headers: {
                    'user_id': this.userId
                }
            });
            const data = await response.json();
            
            this.displayDocuments(data.documents);
        } catch (error) {
            this.showNotification('Failed to load documents', 'error');
            console.error('Load documents error:', error);
        }
    }
    
    displayDocuments(documents) {
        const grid = document.getElementById('documents-grid');
        const noDocsElement = document.getElementById('no-documents');
        
        if (documents.length === 0) {
            noDocsElement.style.display = 'block';
            return;
        }
        
        noDocsElement.style.display = 'none';
        
        // Remove existing document cards (keep no-documents element)
        const existingCards = grid.querySelectorAll('.document-card');
        existingCards.forEach(card => card.remove());
        
        documents.forEach(doc => {
            const card = this.createDocumentCard(doc);
            grid.appendChild(card);
        });
    }
    
    createDocumentCard(doc) {
        const card = document.createElement('div');
        card.className = 'document-card';
        
        const formatTimestamp = (timestamp) => {
            const date = new Date(timestamp * 1000);
            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        };
        
        const getFileIcon = (languages) => {
            if (languages.includes('ar')) return 'fa-file-text';
            if (languages.includes('zh')) return 'fa-file-text';
            if (languages.includes('hi')) return 'fa-file-text';
            return 'fa-file-alt';
        };
        
        card.innerHTML = `
            <div class="document-header">
                <i class="fas ${getFileIcon(doc.languages)} document-icon"></i>
                <div class="document-info">
                    <h3>${doc.document_id.substring(0, 8)}...</h3>
                    <div class="document-meta">
                        Added: ${formatTimestamp(doc.added_timestamp)}
                    </div>
                </div>
            </div>
            
            <div class="document-stats">
                <div class="stat-item">
                    <i class="fas fa-file-text"></i>
                    <span>${doc.total_chunks} chunks</span>
                </div>
                <div class="stat-item">
                    <i class="fas fa-file"></i>
                    <span>${doc.pages.length} pages</span>
                </div>
                <div class="stat-item">
                    <i class="fas fa-language"></i>
                    <span>${doc.languages.join(', ')}</span>
                </div>
            </div>
            
            <div class="document-actions">
                <button class="btn btn-primary btn-sm chat-btn" data-doc-id="${doc.document_id}">
                    <i class="fas fa-comments"></i> Chat
                </button>
                <button class="btn btn-outline btn-sm analyze-btn" data-doc-id="${doc.document_id}">
                    <i class="fas fa-chart-bar"></i> Analyze
                </button>
                <button class="btn btn-outline btn-sm delete-btn" data-doc-id="${doc.document_id}">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        `;
        
        // Bind card events
        card.querySelector('.chat-btn').addEventListener('click', () => {
            this.startChat(doc.document_id);
        });
        
        card.querySelector('.analyze-btn').addEventListener('click', () => {
            this.analyzeDocument(doc.document_id);
        });
        
        card.querySelector('.delete-btn').addEventListener('click', () => {
            this.deleteDocument(doc.document_id);
        });
        
        return card;
    }
    
    // Chat Functionality
    startChat(documentId) {
        this.currentDocumentId = documentId;
        this.connectWebSocket();
        this.showChatInterface(documentId);
    }
    
    showChatInterface(documentId) {
        const chatSection = document.getElementById('chat-section');
        const chatTitle = document.getElementById('chat-title');
        
        chatTitle.textContent = `Chat with Document ${documentId.substring(0, 8)}...`;
        chatSection.hidden = false;
        
        // Clear previous messages except welcome
        const messagesContainer = document.getElementById('chat-messages');
        const messages = messagesContainer.querySelectorAll('.message');
        messages.forEach(msg => msg.remove());
        
        // Focus chat input
        document.getElementById('chat-input').focus();
    }
    
    connectWebSocket() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return;
        }
        
        const clientId = Math.random().toString(36).substring(7);
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsURL = `${wsProtocol}//${window.location.host}/ws/${clientId}`;
        
        this.updateConnectionStatus('connecting', 'Connecting...');
        
        this.websocket = new WebSocket(wsURL);
        
        this.websocket.onopen = () => {
            this.updateConnectionStatus('connected', 'Connected');
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            this.updateConnectionStatus('disconnected', 'Disconnected');
            // Try to reconnect after 3 seconds
            setTimeout(() => {
                if (document.getElementById('chat-section').hidden === false) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected', 'Connection error');
        };
    }
    
    updateConnectionStatus(status, text) {
        const indicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');
        
        indicator.className = `status-indicator ${status}`;
        statusText.textContent = text;
    }
    
    handleWebSocketMessage(data) {
        if (data.type === 'processing') {
            this.showTypingIndicator(data.message);
        } else if (data.type === 'response') {
            this.hideTypingIndicator();
            this.addMessage('assistant', data.response, {
                sources: data.sources,
                confidence: data.confidence,
                processingTime: data.processing_time
            });
        } else if (data.type === 'error') {
            this.hideTypingIndicator();
            this.addMessage('assistant', `Error: ${data.message}`, { isError: true });
        }
    }
    
    sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Add user message to chat
        this.addMessage('user', message);
        
        // Send message via WebSocket
        const messageData = {
            message: message,
            user_id: this.userId,
            document_id: this.currentDocumentId,
            language: this.selectedLanguage
        };
        
        this.websocket.send(JSON.stringify(messageData));
        
        // Clear input
        input.value = '';
    }
    
    addMessage(sender, content, metadata = {}) {
        const messagesContainer = document.getElementById('chat-messages');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        if (metadata.isError) {
            contentDiv.style.backgroundColor = 'var(--danger-color)';
            contentDiv.style.color = 'white';
        }
        
        messageDiv.appendChild(contentDiv);
        
        // Add metadata for assistant messages
        if (sender === 'assistant' && metadata.sources && metadata.sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';
            sourcesDiv.innerHTML = `
                <i class="fas fa-file-text"></i>
                Sources: Page(s) ${metadata.sources.join(', ')}
                ${metadata.confidence ? `(Confidence: ${(metadata.confidence * 100).toFixed(1)}%)` : ''}
            `;
            contentDiv.appendChild(sourcesDiv);
        }
        
        // Add timestamp
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = new Date().toLocaleTimeString();
        contentDiv.appendChild(timeDiv);
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    showTypingIndicator(message = 'AI is typing...') {
        const messagesContainer = document.getElementById('chat-messages');
        
        // Remove existing typing indicator
        const existingIndicator = messagesContainer.querySelector('.typing-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = `
            <i class="fas fa-robot"></i>
            <span>${message}</span>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.querySelector('.typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        const messages = messagesContainer.querySelectorAll('.message, .typing-indicator');
        messages.forEach(msg => msg.remove());
    }
    
    closeChat() {
        document.getElementById('chat-section').hidden = true;
        
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.currentDocumentId = null;
    }
    
    // Document Analysis
    async analyzeDocument(documentId) {
        try {
            this.showModal();
            document.getElementById('modal-title').textContent = `Document Analysis: ${documentId.substring(0, 8)}...`;
            
            const response = await fetch(`${this.baseURL}/analyze/${documentId}`);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Analysis failed');
            }
            
            this.displayAnalysisResults(data);
        } catch (error) {
            document.getElementById('modal-body').innerHTML = `
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle" style="font-size: 2rem; color: var(--danger-color); margin-bottom: 1rem;"></i>
                    <p>Failed to analyze document: ${error.message}</p>
                </div>
            `;
        }
    }
    
    displayAnalysisResults(data) {
        const modalBody = document.getElementById('modal-body');
        const analysis = data.analysis;
        
        modalBody.innerHTML = `
            <div class="analysis-results">
                <h4>Document Structure</h4>
                <div class="analysis-grid">
                    <div class="analysis-item">
                        <strong>Total Elements:</strong> ${analysis.total_elements}
                    </div>
                    <div class="analysis-item">
                        <strong>Element Types:</strong>
                        <ul style="margin-top: 0.5rem;">
                            ${Object.entries(analysis.element_types).map(([type, count]) => 
                                `<li>${type}: ${count}</li>`
                            ).join('')}
                        </ul>
                    </div>
                    <div class="analysis-item">
                        <strong>Languages Detected:</strong>
                        <ul style="margin-top: 0.5rem;">
                            ${Object.entries(analysis.languages_detected).map(([lang, count]) => 
                                `<li>${lang}: ${count} elements</li>`
                            ).join('')}
                        </ul>
                    </div>
                    <div class="analysis-item">
                        <strong>Readability Score:</strong> ${analysis.readability_score?.toFixed(1) || 'N/A'}/100
                    </div>
                    <div class="analysis-item">
                        <strong>Sentiment:</strong> ${analysis.sentiment_analysis?.sentiment || 'N/A'} 
                        ${analysis.sentiment_analysis?.confidence ? 
                            `(${(analysis.sentiment_analysis.confidence * 100).toFixed(1)}%)` : ''}
                    </div>
                </div>
                
                ${analysis.key_topics?.length ? `
                <h4 style="margin-top: 2rem;">Key Topics</h4>
                <div class="key-topics">
                    ${analysis.key_topics.map(topic => `<span class="topic-tag">${topic}</span>`).join('')}
                </div>
                ` : ''}
                
                <h4 style="margin-top: 2rem;">Vector Store Information</h4>
                <div class="analysis-item">
                    <strong>Total Chunks:</strong> ${data.total_chunks}
                </div>
            </div>
        `;
        
        // Add CSS for analysis display
        const style = document.createElement('style');
        style.textContent = `
            .analysis-grid {
                display: grid;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            .analysis-item {
                padding: 1rem;
                background: var(--bg-secondary);
                border-radius: var(--border-radius);
                border: 1px solid var(--border-color);
            }
            .key-topics {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            .topic-tag {
                padding: 0.25rem 0.75rem;
                background: var(--primary-color);
                color: white;
                border-radius: 1rem;
                font-size: 0.8rem;
            }
        `;
        document.head.appendChild(style);
    }
    
    // Document Deletion
    async deleteDocument(documentId) {
        if (!confirm(`Are you sure you want to delete document ${documentId.substring(0, 8)}...? This action cannot be undone.`)) {
            return;
        }
        
        try {
            const response = await fetch(`${this.baseURL}/documents/${documentId}`, {
                method: 'DELETE',
                headers: {
                    'user_id': this.userId
                }
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Delete failed');
            }
            
            this.showNotification('Document deleted successfully', 'success');
            this.loadDocuments();
            
            // Close chat if it was for this document
            if (this.currentDocumentId === documentId) {
                this.closeChat();
            }
            
        } catch (error) {
            this.showNotification(`Failed to delete document: ${error.message}`, 'error');
        }
    }
    
    // System Statistics
    async showSystemStats() {
        try {
            this.showStatsModal();
            
            const response = await fetch(`${this.baseURL}/stats`);
            const data = await response.json();
            
            this.displaySystemStats(data);
        } catch (error) {
            document.getElementById('stats-modal-body').innerHTML = `
                <div class="text-center">
                    <i class="fas fa-exclamation-triangle" style="font-size: 2rem; color: var(--danger-color); margin-bottom: 1rem;"></i>
                    <p>Failed to load system statistics: ${error.message}</p>
                </div>
            `;
        }
    }
    
    displaySystemStats(data) {
        const modalBody = document.getElementById('stats-modal-body');
        
        modalBody.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card">
                    <h4>System Status</h4>
                    <div class="stat-value">${data.system_status}</div>
                </div>
                
                <div class="stat-card">
                    <h4>Active Connections</h4>
                    <div class="stat-value">${data.active_websocket_connections}</div>
                </div>
                
                ${data.vector_store ? `
                <div class="stat-card">
                    <h4>Total Documents</h4>
                    <div class="stat-value">${data.vector_store.total_documents}</div>
                </div>
                
                <div class="stat-card">
                    <h4>Total Chunks</h4>
                    <div class="stat-value">${data.vector_store.total_chunks}</div>
                </div>
                
                <div class="stat-card">
                    <h4>Languages Supported</h4>
                    <div class="stat-value">${data.vector_store.languages.join(', ')}</div>
                </div>
                ` : ''}
                
                ${data.ai_models ? `
                <div class="stat-card full-width">
                    <h4>AI Models</h4>
                    <div class="model-info">
                        <p><strong>Device:</strong> ${data.ai_models.device}</p>
                        <p><strong>Embedding Model:</strong> ${data.ai_models.embedding_model}</p>
                        <p><strong>QA Model:</strong> ${data.ai_models.qa_model}</p>
                        <p><strong>Summarizer:</strong> ${data.ai_models.summarizer}</p>
                        <p><strong>Multilingual Support:</strong> ${data.ai_models.multilingual_support ? 'Yes' : 'No'}</p>
                        <p><strong>Cost:</strong> ${data.ai_models.cost}</p>
                    </div>
                </div>
                ` : ''}
            </div>
        `;
        
        // Add CSS for stats display
        const style = document.createElement('style');
        style.textContent = `
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            .stat-card {
                padding: 1.5rem;
                background: var(--bg-secondary);
                border-radius: var(--border-radius);
                border: 1px solid var(--border-color);
                text-align: center;
            }
            .stat-card h4 {
                margin-bottom: 1rem;
                color: var(--text-secondary);
                font-size: 0.9rem;
                font-weight: 500;
                text-transform: uppercase;
            }
            .stat-value {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--primary-color);
            }
            .stat-card.full-width {
                grid-column: 1 / -1;
                text-align: left;
            }
            .model-info p {
                margin-bottom: 0.5rem;
                font-size: 0.9rem;
            }
        `;
        if (!document.head.querySelector('style[data-stats]')) {
            style.setAttribute('data-stats', 'true');
            document.head.appendChild(style);
        }
    }
    
    // Modal Management
    showModal() {
        document.getElementById('document-modal').classList.add('show');
    }
    
    closeModal() {
        document.getElementById('document-modal').classList.remove('show');
    }
    
    showStatsModal() {
        document.getElementById('stats-modal').classList.add('show');
    }
    
    closeStatsModal() {
        document.getElementById('stats-modal').classList.remove('show');
    }
    
    showDocumentDetails(docData) {
        this.showModal();
        document.getElementById('modal-title').textContent = `Document Processed: ${docData.filename}`;
        
        document.getElementById('modal-body').innerHTML = `
            <div class="document-details">
                <div class="detail-section">
                    <h4>Processing Summary</h4>
                    <p><strong>Status:</strong> ${docData.status}</p>
                    <p><strong>Processing Time:</strong> ${docData.processing_time.toFixed(2)} seconds</p>
                    <p><strong>Document ID:</strong> ${docData.document_id}</p>
                </div>
                
                <div class="detail-section">
                    <h4>Statistics</h4>
                    <div class="stats-list">
                        ${Object.entries(docData.statistics).map(([key, value]) => `
                            <div class="stat-row">
                                <span class="stat-key">${key.replace(/_/g, ' ').toUpperCase()}:</span>
                                <span class="stat-value">${typeof value === 'object' ? JSON.stringify(value) : value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                
                ${docData.summary ? `
                <div class="detail-section">
                    <h4>Document Summary</h4>
                    <div class="summary-text">${docData.summary}</div>
                </div>
                ` : ''}
                
                <div class="detail-actions">
                    <button class="btn btn-primary" onclick="app.startChat('${docData.document_id}'); app.closeModal();">
                        <i class="fas fa-comments"></i> Start Chat
                    </button>
                </div>
            </div>
        `;
        
        // Add CSS for document details
        const style = document.createElement('style');
        style.textContent = `
            .document-details .detail-section {
                margin-bottom: 2rem;
            }
            .document-details h4 {
                margin-bottom: 1rem;
                color: var(--primary-color);
                border-bottom: 1px solid var(--border-color);
                padding-bottom: 0.5rem;
            }
            .stats-list .stat-row {
                display: flex;
                justify-content: space-between;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--border-color);
            }
            .stats-list .stat-key {
                font-weight: 500;
                color: var(--text-secondary);
            }
            .summary-text {
                background: var(--bg-secondary);
                padding: 1rem;
                border-radius: var(--border-radius);
                border-left: 4px solid var(--primary-color);
                line-height: 1.6;
            }
            .detail-actions {
                text-align: center;
                padding-top: 1rem;
                border-top: 1px solid var(--border-color);
            }
        `;
        if (!document.head.querySelector('style[data-details]')) {
            style.setAttribute('data-details', 'true');
            document.head.appendChild(style);
        }
    }
    
    // Loading Overlay
    showLoadingOverlay(title = "Loading...", subtitle = "") {
        const overlay = document.getElementById('loading-overlay');
        const titleEl = overlay.querySelector('.loading-text');
        const subtitleEl = overlay.querySelector('.loading-subtext');
        
        titleEl.textContent = title;
        subtitleEl.textContent = subtitle;
        overlay.hidden = false;
    }
    
    hideLoadingOverlay() {
        console.log('hideLoadingOverlay() called');
        const overlay = document.getElementById('loading-overlay');
        console.log('Loading overlay element:', overlay);
        if (overlay) {
            overlay.hidden = true;
            console.log('Loading overlay hidden successfully');
        } else {
            console.error('Loading overlay element not found!');
        }
    }
    
    forceHideLoadingOverlay() {
        console.log('FORCE HIDE LOADING OVERLAY - AGGRESSIVE APPROACH');
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            // Multiple ways to hide it
            overlay.hidden = true;
            overlay.style.display = 'none';
            overlay.style.visibility = 'hidden';
            overlay.style.opacity = '0';
            overlay.style.zIndex = '-999';
            
            // Remove it from DOM entirely as last resort
            overlay.remove();
            console.log('Loading overlay FORCE REMOVED from DOM');
        }
        
        // Also try with different selectors in case ID fails
        const overlays = document.querySelectorAll('.loading-overlay, [id*="loading"], [class*="loading"]');
        overlays.forEach((el, index) => {
            el.style.display = 'none';
            el.hidden = true;
            console.log(`Force hidden loading element ${index}:`, el);
        });
    }
    
    // Notification System
    showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icon = this.getNotificationIcon(type);
        notification.innerHTML = `
            <i class="${icon}"></i>
            <div>
                <div>${message}</div>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
    
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'fas fa-check-circle';
            case 'error': return 'fas fa-exclamation-circle';
            case 'warning': return 'fas fa-exclamation-triangle';
            default: return 'fas fa-info-circle';
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PolyDocApp();
});
