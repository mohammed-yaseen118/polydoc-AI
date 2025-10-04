# PolyDoc Backend Testing Pipeline

## Overview

This directory contains a comprehensive testing framework for PolyDoc's machine learning capabilities. The pipeline allows you to test various AI/ML components including document processing, classification, question-answering, and sentiment analysis without requiring the full backend infrastructure.

## Features

- **ML Training Framework**: Test machine learning models with sample data
- **Document Classification**: Test document categorization capabilities
- **Question-Answering**: Test QA model performance
- **Sentiment Analysis**: Test text sentiment classification
- **Robustness Testing**: Validate model stability and performance
- **Custom Data Testing**: Use your own CSV datasets for testing

## Files Structure

```
test-backend/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_tests.py                # Main test runner
├── ml_trainer.py               # ML training framework
├── analyze_results.py          # Results analysis tools
├── sample_training_data.csv    # Sample training dataset
├── sample_test_data.csv        # Sample testing dataset
├── sample_validation_data.csv  # Sample validation dataset
└── ml_training.log            # Training logs
```

## Quick Start

# Long form

python analyze_results.py --section classification
python analyze_results.py --section qa
python analyze_results.py --section sentiment
python analyze_results.py --section robustness
python analyze_results.py --section summary
python analyze_results.py --section logs

# Short form (NEW!)

python analyze_results.py --classification
python analyze_results.py --qa
python analyze_results.py --sentiment
python analyze_results.py --robustness
python analyze_results.py --summary
python analyze_results.py --logs

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Tests

```bash
# Run all tests with sample data
python run_tests.py --test basic

# Run specific component tests
python run_tests.py --test classification
python run_tests.py --test qa
python run_tests.py --test sentiment
python run_tests.py --test robustness
```

### 3. Use Custom Data

```bash
# Test with your own CSV file
python run_tests.py --test custom --csv-path /path/to/your/data.csv
```

## Test Types

### Basic Test (`--test basic`)

Runs the complete ML pipeline with sample data including:

- Document classification
- Question-answering
- Sentiment analysis
- Model validation

### Classification Test (`--test classification`)

Tests document/text classification capabilities:

- Requires CSV with 'text' and 'label' columns
- Outputs accuracy, F1-score, and class performance

### Question-Answering Test (`--test qa`)

Tests QA model performance:

- Requires CSV with 'question', 'answer', and 'context' columns
- Measures similarity scores and confidence levels

### Sentiment Analysis Test (`--test sentiment`)

Tests sentiment classification:

- Requires CSV with 'text' column
- Analyzes sentiment polarity and confidence

### Robustness Test (`--test robustness`)

Tests model stability and performance:

- Uses validation data to test edge cases
- Measures processing time and success rates

### Custom Test (`--test custom`)

Tests with your own data:

- Automatically detects available columns
- Runs appropriate tests based on data structure

## CSV Data Format

### For Classification:

```csv
text,label
"Document content here...",category1
"Another document...",category2
```

### For Question-Answering:

```csv
question,answer,context
"What is AI?","Artificial Intelligence","AI is a field of computer science..."
```

### For Sentiment Analysis:

```csv
text,sentiment
"This is great!",positive
"This is terrible.",negative
```

## Sample Data

The pipeline includes three sample CSV files:

1. **sample_training_data.csv**: Training dataset with mixed content types
2. **sample_test_data.csv**: Testing dataset for validation
3. **sample_validation_data.csv**: Validation dataset for robustness testing

## Usage Examples

### Run Complete Pipeline

```bash
python run_tests.py --test basic
```

### Test Only Classification

```bash
python run_tests.py --test classification
```

### Test with Custom Data

```bash
python run_tests.py --test custom --csv-path my_documents.csv
```

### Analyze Results

```bash
python analyze_results.py --section classification
python analyze_results.py --section logs --log-file ml_training.log
```

## Performance Metrics

The pipeline tracks various metrics:

- **Accuracy**: Classification accuracy percentage
- **F1-Score**: Harmonic mean of precision and recall
- **Similarity Score**: Semantic similarity for QA tasks
- **Confidence Score**: Model confidence in predictions
- **Processing Time**: Time taken for each operation
- **Success Rate**: Percentage of successful operations

## Troubleshooting

### Common Issues:

1. **Missing Dependencies**: Run `pip install -r requirements.txt`
2. **CSV Format Errors**: Ensure proper column names and format
3. **Memory Issues**: Reduce batch size in ml_trainer.py
4. **Model Loading Errors**: Check internet connection for model downloads
5. **Import Errors**: The framework will run in mock mode if PolyDoc models aren't available
6. **Argument Parsing**: Both `--test` and `--test-type` arguments are supported for backward compatibility

### Testing the Framework:

Before running the full tests, you can verify the framework is working:

```bash
# Test the framework itself
python test_framework.py
```

This will run basic validation tests without requiring the full backend.

### Log Files:

Check `ml_training.log` for detailed execution logs and error messages.

## Integration with Main Backend

This testing framework uses lightweight versions of the models to avoid loading the full backend. For production testing:

1. Ensure main backend is running on port 8000
2. Use the frontend testing interface for full pipeline testing
3. Run integration tests through the web interface

## Notes

- **No Model Persistence**: Tests don't save trained models to conserve space
- **Lightweight Models**: Uses smaller model variants for faster testing
- **Memory Efficient**: Optimized for systems with limited resources
- **Offline Capable**: Most tests can run without internet after initial setup

## Contributing

When adding new tests:

1. Add test functions to `run_tests.py`
2. Update the argument parser for new test types
3. Document new CSV format requirements
4. Add sample data if needed
5. Update this README

## License

Part of the PolyDoc AI project. See main project license for details.
