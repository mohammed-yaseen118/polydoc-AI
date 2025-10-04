# 🤝 Contributing to PolyDoc AI

Thank you for your interest in contributing to PolyDoc AI! This guide will help you get started with contributing to our multilingual document processing platform.

## 🎯 Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Report it with detailed steps to reproduce
- 💡 **Feature Requests**: Have an idea? Share it with us through issues
- 📝 **Documentation**: Help improve our docs, README, or code comments
- 🔧 **Code Contributions**: Fix bugs, add features, or improve performance
- 🌍 **Localization**: Add support for more languages or improve existing ones
- 🧪 **Testing**: Add tests, improve test coverage, or test on different platforms

## 🚀 Getting Started

### 1. Fork the Repository
Click the "Fork" button at the top of the repository page.

### 2. Clone Your Fork
```bash
git clone https://github.com/yourusername/polydoc-ai.git
cd polydoc-ai
```

### 3. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install frontend dependencies
npm install
```

### 4. Create a Feature Branch
```bash
git checkout -b feature/amazing-new-feature
# or
git checkout -b fix/bug-description
```

## 🔧 Development Guidelines

### Code Style

#### Python
- Follow **PEP 8** standards
- Use **type hints** where possible
- Write **docstrings** for all functions and classes
- Use **black** formatter: `black src/`
- Use **isort** for imports: `isort src/`

```python
async def process_document(
    file_path: Path, 
    language: str = 'en'
) -> ProcessedDocument:
    """
    Process a document and extract content.
    
    Args:
        file_path: Path to the document file
        language: Target language for processing
        
    Returns:
        ProcessedDocument containing extracted content
    """
    # Implementation here
    pass
```

#### JavaScript/React
- Follow **ESLint** configuration
- Use **functional components** with hooks
- Write **JSDoc** comments for complex functions
- Use **Prettier** for formatting

```javascript
/**
 * Upload and process a document
 * @param {File} file - The document file
 * @param {string} userId - User identifier
 * @returns {Promise<Object>} Processing result
 */
const uploadDocument = async (file, userId) => {
  // Implementation here
};
```

### Testing

#### Backend Tests
```bash
# Run all tests
python -m pytest test-backend/ -v

# Run specific test file
python -m pytest test-backend/test_document_processor.py -v

# Run with coverage
python -m pytest --cov=src test-backend/
```

#### Frontend Tests
```bash
# Run React tests
npm test

# Run tests with coverage
npm test -- --coverage
```

#### Integration Tests
```bash
# Test core functionality
python test_chat_lightweight.py

# Run comprehensive tests
python test-backend/run_tests.py
```

### Adding New Features

#### 1. Document Processing Features
- Add new file format support in `src/core/document_processor.py`
- Update supported formats in README
- Add tests in `test-backend/test_document_processor.py`

#### 2. AI Model Features
- Add new models in `src/models/ai_models.py`
- Consider memory and performance impact
- Add fallback mechanisms for reliability

#### 3. Language Support
- Update `src/utils/indian_language_detector.py`
- Add language-specific processing logic
- Test with sample documents in the target language

#### 4. API Endpoints
- Add new routes in `src/api/main_mongodb.py`
- Include proper error handling
- Update API documentation

### Commit Guidelines

Use **conventional commits** format:

```bash
# Feature
git commit -m "feat: add support for Excel file processing"

# Bug fix  
git commit -m "fix: resolve memory leak in document processing"

# Documentation
git commit -m "docs: update installation instructions"

# Tests
git commit -m "test: add unit tests for language detection"

# Refactor
git commit -m "refactor: optimize vector search performance"
```

## 🧪 Testing Your Changes

### Before Submitting
1. **Run all tests**: `python -m pytest test-backend/ -v`
2. **Test functionality**: `python test_chat_lightweight.py`
3. **Check code style**: `black src/ && isort src/`
4. **Test on different environments** (if possible)

### Manual Testing Checklist
- [ ] Document upload works for supported formats
- [ ] Chat functionality returns relevant responses
- [ ] Language detection works correctly
- [ ] Error handling provides helpful messages
- [ ] Performance is acceptable

## 📝 Pull Request Process

### 1. Update Documentation
- Update README.md if needed
- Add docstrings to new functions
- Update API documentation if adding endpoints

### 2. Create Pull Request
- Use a descriptive title
- Fill out the PR template
- Link to related issues
- Include screenshots/videos for UI changes

### 3. PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other: ___

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🌟 Areas We Need Help

### High Priority
- **Docker Support**: Create Docker containers and compose files
- **Performance Optimization**: Improve model loading and processing speed
- **More File Formats**: Add support for RTF, EPUB, etc.
- **Better Error Handling**: Improve user-facing error messages

### Medium Priority
- **Mobile Responsiveness**: Improve mobile UI/UX
- **Batch Processing**: Support multiple file uploads
- **Export Features**: PDF reports, summaries export
- **Advanced Search**: Filters, sorting, faceted search

### Language Support
- **More Indian Languages**: Sindhi, Manipuri, Sanskrit, etc.
- **International Languages**: Arabic, Chinese, Japanese, etc.
- **Better Translation**: Improve cross-lingual capabilities

## 🐛 Bug Reports

### Good Bug Report Includes:
1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment details** (OS, Python version, etc.)
5. **Error messages** or logs
6. **Sample files** that cause the issue (if applicable)

### Bug Report Template:
```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
1. Go to '...'
2. Click on '...'
3. Upload file '...'
4. See error

**Expected Behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.9.1]
- Browser: [e.g., Chrome 91]

**Additional Context**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Good Feature Request Includes:
1. **Clear description** of the feature
2. **Use case** or problem it solves
3. **Proposed solution** (if you have ideas)
4. **Alternative solutions** considered
5. **Additional context** or examples

## 🌍 Localization

### Adding Language Support
1. **Language Detection**: Update detection algorithms
2. **OCR Support**: Ensure OCR works with the script
3. **AI Models**: Test existing models or add specialized ones
4. **UI Translation**: Add UI strings for the language
5. **Documentation**: Update supported languages list

## 🏷️ Issue Labels

We use these labels to organize issues:

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `documentation` - Improvements to documentation
- `performance` - Performance improvements
- `language-support` - Adding new language support
- `frontend` - React/UI related
- `backend` - Python/API related
- `ai-models` - AI/ML model related

## 🎉 Recognition

Contributors will be:
- **Listed in README**: All contributors get recognition
- **Mentioned in releases**: Significant contributions highlighted
- **Given credit**: Proper attribution in code and documentation

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Email**: [maintainer@polydoc-ai.com] for private matters

## 📄 Code of Conduct

- **Be respectful** and inclusive
- **Be patient** with newcomers
- **Be constructive** in feedback
- **Focus on what's best** for the community
- **Show empathy** towards other community members

---

Thank you for contributing to PolyDoc AI! Every contribution, no matter how small, makes a difference. 🙏

**Happy Coding! 🚀**