# 🚀 PolyDoc AI - Intelligent Document Processing & Analysis Platform

<div align="center">



**Transform any document into intelligent, searchable knowledge**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-green.svg)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-green.svg)](https://mongodb.com)

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

## 📖 Overview

**PolyDoc AI** is an advanced, multilingual document processing and analysis platform that transforms unstructured documents into intelligent, searchable knowledge bases. Built with cutting-edge AI technologies, it supports 15+ file formats and provides context-aware answers through natural language chat interfaces.

### 🎯 Key Highlights

- **🌏 Multilingual Support**: Native support for **11+ Indian languages** (Hindi, Kannada, Telugu, Tamil, Bengali, Gujarati, Malayalam, Marathi, Punjabi, Odia, Assamese) + English
- **📄 Universal Document Processing**: Supports PDF, DOCX, PPTX, images, CSV, Excel, Markdown, HTML, ODT, and more
- **🤖 AI-Powered Analysis**: Context-aware Q&A, intelligent summarization, and document insights
- **🔍 Advanced Search**: Vector-based semantic search with intelligent fallbacks
- **⚡ Real-time Chat**: WebSocket-powered chat interface with streaming responses
- **🔧 Robust Architecture**: MongoDB + FastAPI + React with comprehensive error handling

## ✨ Features

### 📄 Document Processing
- **Multi-format Support**: PDF, DOCX, PPTX, images (PNG, JPG), CSV, Excel, Markdown, HTML, ODT
- **Advanced OCR**: EasyOCR + Tesseract with preprocessing pipelines for optimal text extraction
- **Language Detection**: Automatic detection with specialized Indian language recognition
- **Metadata Extraction**: Comprehensive document statistics and content analysis

### 🤖 AI-Powered Intelligence
- **Context-Aware Q&A**: Ask questions about your documents and get intelligent, context-specific answers
- **Intelligent Summarization**: Extractive and abstractive summarization with bilingual support
- **Semantic Search**: Vector-based similarity search with multiple fallback strategies
- **Indian Language AI**: Specialized models for Hindi, Kannada, Telugu, Tamil, and other Indian languages

### 🌐 Modern Web Interface
- **Responsive Design**: Modern React UI with Tailwind CSS
- **Real-time Chat**: WebSocket-powered chat with typing indicators
- **File Upload**: Drag-and-drop interface with progress tracking
- **User Authentication**: Firebase-based authentication system
- **Theme Support**: Light/dark mode with smooth transitions

### 🔧 Technical Excellence
- **Scalable Architecture**: Microservices with MongoDB for document storage
- **Error Resilience**: Comprehensive fallback mechanisms for reliable operation
- **Performance Optimized**: Efficient model loading, caching, and memory management
- **API-First Design**: RESTful API with OpenAPI/Swagger documentation

## 🎬 Demo

### Document Upload & Processing
```bash
# Upload a document
curl -X POST "http://localhost:8000/api/v1/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "user_id=demo_user"
```

### Intelligent Q&A
```python
# Ask questions about your documents
response = await chat_with_document(
    question="What are the main findings in this research?",
    user_id="demo_user",
    language="en"
)
```

### Multilingual Support
```python
# Process Hindi document
result = await process_document("हिंदी_दस्तावेज़.pdf")
# Get bilingual summary
summary = await get_summary(text, language="hi")  # Returns Hindi + English
```

## 🚀 Installation

### Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **MongoDB 6.0+**
- **4GB+ RAM** (recommended 8GB)
- **5GB+ free disk space** for AI models

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/polydoc-ai.git
   cd polydoc-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup MongoDB**
   ```bash
   # Install MongoDB locally or use MongoDB Atlas
   # Update connection string in src/core/mongodb_store.py
   ```

5. **Start backend server**
   ```bash
   python start_backend.py
   # or
   uvicorn src.api.main_mongodb:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Install Node.js dependencies**
   ```bash
   npm install
   ```

2. **Configure Firebase** (optional, for authentication)
   ```bash
   # Add your Firebase config to src/config/firebase.js
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

### Quick Start with Docker (Coming Soon)
```bash
docker-compose up -d
```

## 📘 Usage

### 1. Document Upload
```javascript
// Upload document via API
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('user_id', 'your_user_id');

const response = await fetch('/api/v1/upload', {
  method: 'POST',
  body: formData
});
```

### 2. Chat with Documents
```javascript
// WebSocket chat
const ws = new WebSocket('ws://localhost:8000/ws/chat/your_user_id');
ws.send(JSON.stringify({
  message: "What are the key points in the uploaded document?",
  language: "en"
}));
```

### 3. Document Analysis
```python
# Python API example
from polydoc_ai import PolyDocClient

client = PolyDocClient("http://localhost:8000")

# Upload and process
doc_id = await client.upload_document("report.pdf", user_id="user123")

# Get summary
summary = await client.get_summary(doc_id, language="en")

# Ask questions
answer = await client.ask_question(
    "What is the main conclusion?", 
    user_id="user123"
)
```

## 🏗️ Architecture

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI API   │    │   MongoDB       │
│   - Modern UI   │◄──►│   - REST/WS     │◄──►│   - Documents   │
│   - Real-time   │    │   - AI Models   │    │   - Vectors     │
│   - Auth        │    │   - Processing  │    │   - Metadata    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│   AI Pipeline   │◄─────────────┘
                        │   - NLP Models  │
                        │   - OCR Engine  │
                        │   - Embeddings  │
                        └─────────────────┘
```

### Data Flow
1. **Document Upload** → **Processing Pipeline** → **MongoDB Storage**
2. **User Query** → **Vector Search** → **AI Response Generation** → **Formatted Output**
3. **Fallback Chain**: Vector Search → Text Search → General Chunks → Emergency Fallback

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 18, Tailwind CSS, Vite | Modern responsive UI |
| **Backend API** | FastAPI, Python 3.8+ | RESTful API & WebSocket |
| **Database** | MongoDB 6.0+ | Document & vector storage |
| **AI Models** | Transformers, Sentence-BERT | NLP & embeddings |
| **OCR Engine** | EasyOCR, Tesseract | Text extraction |
| **Search** | Vector similarity + fallbacks | Semantic search |

## 🌟 Advanced Features

### Multilingual AI Pipeline
- **Language Detection**: Automatic detection with Indian language specialization
- **Bilingual Summarization**: Generate summaries in both original language and English
- **Cross-lingual Search**: Search in English, get results from multilingual documents
- **Script Support**: Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, and more

### Intelligent Fallback System
```python
# Robust search with multiple fallback strategies
1. Vector Similarity Search (primary)
2. Regex Text Search (fallback)
3. General Document Chunks (fallback)
4. Emergency Content Retrieval (last resort)
```

### Performance Optimizations
- **Lazy Model Loading**: Models loaded on-demand to reduce startup time
- **Efficient Caching**: HuggingFace model caching with cleanup
- **Memory Management**: Optimized for resource-constrained environments
- **Batch Processing**: Efficient document processing pipelines

## 📊 Supported Formats

| Category | Formats | Features |
|----------|---------|----------|
| **Documents** | PDF, DOCX, PPTX, ODT, RTF | Text, images, tables, metadata |
| **Spreadsheets** | CSV, XLSX, XLS | Data extraction, formatting |
| **Images** | PNG, JPG, JPEG, TIFF, BMP | OCR with preprocessing |
| **Web** | HTML, Markdown, XML | Structure-aware parsing |
| **Text** | TXT, Log files | Encoding detection |

## 🔧 Configuration

### Environment Variables
```bash
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=polydoc_ai

# AI Models
HUGGINGFACE_CACHE_DIR=~/.cache/huggingface
MODEL_DEVICE=cpu  # or cuda for GPU

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG_MODE=false

# Firebase (optional)
FIREBASE_CONFIG_PATH=./firebase-config.json
```

### Model Configuration
```python
# Customize AI models in src/models/ai_models.py
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "deepset/roberta-base-squad2"
```

## 🧪 Testing

### Run Tests
```bash
# Backend tests
python -m pytest test-backend/ -v

# Lightweight functionality test
python test_chat_lightweight.py

# Integration tests
python test-backend/run_tests.py
```

### Test Coverage
- ✅ Document processing pipeline
- ✅ AI model integration
- ✅ Database operations
- ✅ API endpoints
- ✅ WebSocket functionality
- ✅ Error handling & fallbacks

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style
- **Python**: Follow PEP 8, use `black` formatter
- **JavaScript/React**: Follow ESLint configuration
- **Documentation**: Update README and docstrings for new features

## 📈 Performance

### Benchmarks (on typical hardware)
- **Document Processing**: 10-50 pages/minute (depends on content)
- **OCR Speed**: 1-5 pages/minute (image quality dependent)
- **Search Response**: <500ms for most queries
- **Memory Usage**: 2-6GB (varies with loaded models)

### Optimization Tips
- Use SSD storage for better model loading
- Enable GPU acceleration for faster processing
- Configure adequate virtual memory for large models
- Use MongoDB Atlas for production deployments

## 🛡️ Security

- **Input Validation**: Comprehensive file type and size validation
- **Sanitization**: Clean user inputs and uploaded content
- **Authentication**: Firebase-based secure authentication
- **API Security**: Rate limiting and request validation
- **Data Privacy**: No sensitive data stored in logs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and libraries
- **MongoDB** for flexible document storage
- **FastAPI** for high-performance web framework
- **React** community for UI components and patterns
- **EasyOCR** and **Tesseract** for OCR capabilities

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/polydoc-ai/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/polydoc-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/polydoc-ai/discussions)
- **Email**: support@polydoc-ai.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ by [Your Name](https://github.com/yourusername)**

</div>

---

## 🗺️ Roadmap

### 🎯 Current Version (v2.0.0)
- ✅ Multilingual document processing
- ✅ AI-powered Q&A system
- ✅ Modern React frontend
- ✅ MongoDB integration
- ✅ Indian language support

### 🚀 Upcoming Features (v2.1.0)
- [ ] **Docker Support**: One-click deployment with Docker Compose
- [ ] **API Keys**: Secure API access management
- [ ] **Batch Processing**: Upload and process multiple documents
- [ ] **Export Options**: PDF reports, summaries, and insights
- [ ] **Advanced Analytics**: Document statistics and trends

### 🌟 Future Vision (v3.0.0)
- [ ] **Cloud Integration**: AWS S3, Google Drive, OneDrive support
- [ ] **Collaboration**: Multi-user document sharing and collaboration
- [ ] **Custom Models**: Fine-tune models for specific domains
- [ ] **Mobile App**: React Native mobile application
- [ ] **Enterprise Features**: SSO, audit logs, enterprise security