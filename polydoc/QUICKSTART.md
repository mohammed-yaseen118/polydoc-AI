# 🚀 Quick Start Guide - PolyDoc AI

Get PolyDoc AI up and running in 5 minutes!

## ⚡ One-Line Install & Run

```bash
# Clone, install, and start (requires Python 3.8+, Node.js 18+, MongoDB)
git clone https://github.com/yourusername/polydoc-ai.git && cd polydoc-ai && pip install -r requirements.txt && npm install && python start_backend.py
```

## 🎯 What You'll Get

- **📄 Multi-format document processing** (PDF, DOCX, images, and more)
- **🤖 AI-powered chat** with your documents
- **🌍 11+ Indian languages support** (Hindi, Kannada, Telugu, Tamil, etc.)
- **🔍 Smart search** with fallback mechanisms
- **💬 Real-time WebSocket chat**

## 📋 Prerequisites Check

```bash
# Check if you have the requirements
python --version   # Should be 3.8+
node --version     # Should be 18+
npm --version      # Should be included with Node.js
```

## 🚀 Step-by-Step Setup

### 1. Get the Code
```bash
git clone https://github.com/yourusername/polydoc-ai.git
cd polydoc-ai
```

### 2. Backend Setup (2 minutes)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Frontend Setup (1 minute)
```bash
# Install Node.js packages
npm install
```

### 4. Start the Application (30 seconds)
```bash
# Terminal 1: Start backend
python start_backend.py

# Terminal 2: Start frontend
npm run dev
```

### 5. Open & Test 🎉
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Upload a document** and start chatting!

## 🧪 Quick Test

```bash
# Test if everything is working
python test_chat_lightweight.py
```

You should see:
```
✅ All core components can be imported successfully
✅ All required methods are present
✅ Language detection is working
✅ Indian language support is available
```

## 🐛 Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| **Python/pip not found** | Install Python 3.8+ from python.org |
| **Node/npm not found** | Install Node.js 18+ from nodejs.org |
| **Memory errors** | Close other applications, ensure 4GB+ RAM free |
| **Port 8000 busy** | Change port in `start_backend.py` |
| **MongoDB connection** | Install MongoDB locally or use Atlas (cloud) |

## 📱 Usage Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@your_document.pdf" \
  -F "user_id=test_user"
```

### Chat via WebSocket
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/test_user');
ws.send(JSON.stringify({
  message: "What is this document about?",
  language: "en"
}));
```

### API Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Summarize the document", "user_id": "test_user"}'
```

## 🌟 Key Features to Try

1. **📄 Upload Different Formats**: Try PDF, DOCX, images (PNG/JPG)
2. **🗣️ Ask Questions**: "What are the main points?", "Summarize this"
3. **🌍 Test Indian Languages**: Upload Hindi/Kannada documents
4. **🔍 Search**: Try both specific and general queries
5. **💬 Real-time Chat**: Use the WebSocket interface

## 🎯 Next Steps

- **Read the [full README](README.md)** for comprehensive documentation
- **Check out the [API documentation](http://localhost:8000/docs)** 
- **Browse [example documents](test-docs/)** to test with
- **Review [test results](CHAT_FIXES_SUMMARY.md)** to understand capabilities

## 🆘 Need Help?

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the full README.md
- **Test Problems**: Run `python test_chat_lightweight.py`
- **Performance Issues**: Ensure 4GB+ RAM and 5GB+ disk space

---

**🎉 Congratulations! You now have PolyDoc AI running locally!**

Start uploading documents and chatting with your AI-powered document assistant! 🤖📄