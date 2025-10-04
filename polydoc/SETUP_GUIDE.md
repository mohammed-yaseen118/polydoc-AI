# PolyDoc AI - Complete Setup Guide

## 🚀 Quick Start for New Laptop

This guide helps you set up PolyDoc AI system on a fresh Windows machine after cloning from GitHub.

---

## ⚡ Prerequisites Check

### 1. **System Requirements**
- **OS**: Windows 10/11
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Disk Space**: **At least 10GB free** (for AI models and cache)
- **Internet**: Required for initial model downloads

### 2. **Required Software**

#### **Python 3.9+**
```bash
# Check if Python is installed
python --version

# If not installed, download from: https://python.org/downloads/
# ✅ IMPORTANT: Check "Add Python to PATH" during installation
```

#### **Node.js 18+**
```bash
# Check if Node.js is installed
node --version
npm --version

# If not installed, download from: https://nodejs.org/
```

#### **MongoDB (Optional but Recommended)**
```bash
# Check if MongoDB is installed
mongod --version

# If not installed, download from: https://mongodb.com/try/download/community
```

#### **Git**
```bash
# Check if Git is installed
git --version

# If not installed, download from: https://git-scm.com/download/win
```

---

## 📥 Installation Steps

### Step 1: Clone the Repository
```bash
# Clone the project
git clone <YOUR_GITHUB_REPO_URL>
cd polydoc

# IMPORTANT: Configure Git on new laptop
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"

# Verify all files are present
dir
```

### Step 2: Install Python Dependencies
```bash
# Install all Python packages
pip install -r requirements.txt

# If you get permission errors, try:
pip install --user -r requirements.txt
```

### Step 3: Install Node.js Dependencies
```bash
# Install frontend dependencies
npm install
```

### Step 4: Check Disk Space
```bash
# Run the cache optimizer to check space
python optimize_cache.py
```
**🚨 CRITICAL**: You need **at least 6GB free space** for AI models to download and cache properly.

### Step 5: Create Required Directories
```bash
# Create necessary directories (if not exist)
mkdir uploads static templates
```

---

## 🧪 Pre-Flight Test

Before running the full system, test individual components:

### Test 1: Python Backend Dependencies
```bash
# Test if core libraries work
python -c "import torch, transformers, sentence_transformers; print('✅ Core AI libraries working')"
```

### Test 2: Document Processing
```bash
# Test OCR and document processing
python -c "from src.core.document_processor import DocumentProcessor; print('✅ Document processor ready')"
```

### Test 3: Language Detection
```bash
# Test Hindi/Kannada language detection
python -c "from src.utils.indian_language_detector import IndianLanguageDetector; print('✅ Language detection ready')"
```

### Test 4: Frontend Build
```bash
# Test if frontend builds successfully
npm run build
```

---

## 🚀 Running the System

### Option A: Full System (Recommended)
```bash
# Run the complete system with all AI models
start-all.bat
```

### Option B: Simple Backend (If Space Issues)
```bash
# Run lightweight version without heavy AI models
python simple_backend.py
# In another terminal:
npm run dev
```

---

## 🔧 Troubleshooting Common Issues

### Issue 1: "Cannot connect to backend"
**Solutions:**
- Wait 2-3 minutes for models to load completely
- Check if port 8000 is free: `netstat -an | findstr :8000`
- Restart the backend: Close cmd window and re-run `start-all.bat`

### Issue 2: "Models are still loading..."
**Solutions:**
- First run takes 5-10 minutes to download all models
- Subsequent runs should be much faster (using cached models)
- Check disk space: `python optimize_cache.py`

### Issue 3: "Out of disk space" or "Paging file too small"
**Solutions:**
- Free up at least 10GB disk space
- Increase virtual memory:
  1. Control Panel → System → Advanced → Performance Settings
  2. Advanced → Virtual Memory → Change
  3. Set to "System managed size" or minimum 8GB

### Issue 4: "Module not found" errors
**Solutions:**
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# For specific missing modules:
pip install <module_name>
```

### Issue 5: Frontend shows "Upload failed"
**Solutions:**
- Ensure backend is fully loaded (check for "All systems ready!" message)
- Check backend logs in the cmd window
- Verify file size is under 10MB

### Issue 6: Git push exits early or fails
**Root Cause**: New laptop needs Git authentication setup

**Solutions:**
```bash
# Step 1: Configure Git identity
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"

# Step 2: Set up GitHub authentication
# OPTION A: Personal Access Token (Recommended)
# 1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
# 2. Generate new token (classic) with 'repo' permissions
# 3. Copy the token
# 4. When git asks for password, use the token instead

# OPTION B: SSH Key (Advanced)
# 1. Generate SSH key:
ssh-keygen -t ed25519 -C "your.email@example.com"
# 2. Add to SSH agent:
ssh-add ~/.ssh/id_ed25519
# 3. Copy public key to GitHub SSH settings:
cat ~/.ssh/id_ed25519.pub

# Step 3: Check branch name and push correctly
git branch                    # Check current branch
git push origin main         # If branch is 'main'
# OR
git push origin master       # If branch is 'master'
```

---

## 📁 Project Structure Overview

```
polydoc/
├── src/                     # Core Python backend code
│   ├── api/                 # FastAPI backend
│   ├── core/                # Document processing
│   ├── models/              # AI model management
│   └── utils/               # Utilities (language detection)
├── static/                  # Frontend static files
├── uploads/                 # Uploaded documents
├── templates/               # HTML templates
├── requirements.txt         # Python dependencies
├── package.json            # Node.js dependencies
├── vite.config.js          # Frontend build config
├── main.py                 # Main backend entry point
├── simple_backend.py       # Lightweight backend
├── start-all.bat          # One-click startup script
└── optimize_cache.py      # Cache optimization utility
```

---

## 🌐 System URLs

After successful startup, access these URLs:

- **Frontend UI**: http://localhost:3003
- **Backend API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

---

## 📋 Features Available

### ✅ Core Features
- **Multi-format Document Upload**: PDF, DOCX, PPTX, TXT, Images
- **Hindi/Kannada OCR**: Advanced text extraction
- **Language Detection**: Automatic language identification
- **Document Processing**: Text extraction and analysis
- **File Management**: Upload, view, delete documents

### ✅ AI Features (Full Backend Only)
- **Document Summarization**: AI-generated summaries
- **Question Answering**: Chat with your documents
- **Multilingual Support**: English, Hindi, Kannada processing
- **Semantic Search**: Vector-based document search
- **MongoDB Integration**: Document storage and retrieval

---

## 🆘 Getting Help

### If You Encounter Issues:

1. **Check the Logs**: Look at the backend cmd window for error messages
2. **Run Diagnostics**: 
   ```bash
   python optimize_cache.py
   python -c "import sys; print('Python:', sys.version)"
   node --version
   npm --version
   ```
3. **Create GitHub Issue**: Include:
   - Error messages from cmd window
   - Your system specs (RAM, disk space)
   - Steps that led to the error

### Performance Tips:
- **First Run**: Takes 5-10 minutes (downloading models)
- **Subsequent Runs**: Should start in 30-60 seconds
- **Optimal Performance**: 16GB RAM, 20GB+ free disk space
- **Basic Functionality**: 8GB RAM, 10GB+ free disk space

---

## 🎯 Success Indicators

You know everything is working when you see:

1. **Backend**: "🎉 PolyDoc AI fully loaded - Hindi/Kannada processing ready!"
2. **Frontend**: UI loads at http://localhost:3003 without "Cannot connect" errors
3. **Upload Test**: Successfully upload and process a sample PDF/DOCX file
4. **Language Test**: Upload Hindi/Kannada document and see correct language detection

---

## 🔄 Updates and Maintenance

### Updating the System:
```bash
# Pull latest changes
git pull origin main

# Update Python dependencies
pip install --upgrade -r requirements.txt

# Update Node.js dependencies
npm install

# Clear cache if needed
python optimize_cache.py
```

### Backup Important Data:
- `uploads/` folder (your processed documents)
- Any configuration files you've modified

---

**🎉 You're Ready to Go!**

Run `start-all.bat` and start processing your Hindi/Kannada documents with AI-powered analysis!
