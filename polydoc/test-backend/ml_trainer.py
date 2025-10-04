"""
PolyDoc ML Training and Validation Framework
============================================

This module provides training, testing, and validation capabilities for PolyDoc AI models
using CSV datasets. It includes functionality for:
- Document classification training
- Question-answering validation
- Sentiment analysis testing
- Performance metrics calculation
"""

import pandas as pd
import numpy as np
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add src to path to import our models
project_root = os.path.dirname(os.path.dirname(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Try to import models with fallback
try:
    from models.ai_models import AIModelManager, DocumentAnalyzer
    from core.document_processor import DocumentProcessor
    from utils.indian_language_detector import IndianLanguageDetector, detect_indian_language
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import PolyDoc models: {e}")
    print("Running in mock mode for testing framework only...")
    MODELS_AVAILABLE = False
    
    # Create mock classes for testing
    class MockAIModelManager:
        async def analyze_sentiment(self, text):
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        async def generate_summary(self, text, max_length=150):
            return f"Summary of: {text[:50]}..."
        
        async def answer_question(self, question, context):
            return {'answer': 'Mock answer', 'confidence': 0.5}
    
    class MockDocumentAnalyzer:
        def __init__(self, ai_models):
            self.ai_models = ai_models
    
    class MockDocumentProcessor:
        def process_document(self, content):
            return {'processed': content}
    
    class MockIndianLanguageDetector:
        def detect_language(self, text):
            return {'language': 'en', 'confidence': 0.9}
    
    def detect_indian_language(text):
        return 'en'
    
    AIModelManager = MockAIModelManager
    DocumentAnalyzer = MockDocumentAnalyzer
    DocumentProcessor = MockDocumentProcessor
    IndianLanguageDetector = MockIndianLanguageDetector

class MLTrainingFramework:
    """Main framework for training, testing, and validating PolyDoc AI models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Initialize AI components
        self.ai_models = None
        self.document_analyzer = None
        self.document_processor = None
        
        # Training data storage
        self.training_data = {}
        self.test_data = {}
        self.validation_data = {}
        
        # Results storage
        self.training_results = {}
        self.test_results = {}
        self.validation_results = {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_training.log'),
                logging.StreamHandler()
            ]
        )
    
    async def initialize_models(self):
        """Initialize AI models for training and testing"""
        try:
            if MODELS_AVAILABLE:
                self.logger.info("Initializing PolyDoc AI models...")
            else:
                self.logger.info("Initializing mock AI models for testing...")
            
            self.ai_models = AIModelManager()
            self.document_analyzer = DocumentAnalyzer(self.ai_models)
            self.document_processor = DocumentProcessor()
            
            if MODELS_AVAILABLE:
                self.logger.info("PolyDoc AI models initialized successfully")
            else:
                self.logger.info("Mock AI models initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def load_csv_dataset(self, csv_path: str, dataset_type: str = 'training') -> pd.DataFrame:
        """
        Load CSV dataset for training, testing, or validation
        
        Expected CSV format:
        - text: Input text
        - label: Target label (for classification tasks)
        - question: Question text (for QA tasks)
        - answer: Expected answer (for QA tasks)
        - sentiment: Expected sentiment (for sentiment analysis)
        """
        try:
            self.logger.info(f"Loading {dataset_type} dataset from {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['text']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Store dataset
            if dataset_type == 'training':
                self.training_data['dataframe'] = df
            elif dataset_type == 'testing':
                self.test_data['dataframe'] = df
            elif dataset_type == 'validation':
                self.validation_data['dataframe'] = df
            
            self.logger.info(f"Loaded {len(df)} samples for {dataset_type}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV dataset: {e}")
            raise
    
    async def train_classification_model(self, dataset: pd.DataFrame, target_column: str = 'label') -> Dict[str, Any]:
        """
        Train a text classification model using the provided dataset
        Note: This creates a performance baseline rather than actual training
        since we're using pre-trained models
        """
        try:
            self.logger.info("Starting classification training/evaluation...")
            
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            # Prepare data
            texts = dataset['text'].tolist()
            true_labels = dataset[target_column].tolist()
            
            # Use label encoder for consistency
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(true_labels)
            
            # Split data for training/validation
            X_train, X_val, y_train, y_val = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
            )
            
            # Since we're using pre-trained models, we'll evaluate performance
            # on the validation set to establish a baseline
            predictions = []
            confidences = []
            
            for text in X_val:
                # Use sentiment analysis as a proxy for classification
                sentiment_result = await self.ai_models.analyze_sentiment(text)
                
                # Map sentiment to encoded labels (simple mapping for demo)
                sentiment_to_label = {
                    'positive': 1,
                    'negative': 0,
                    'neutral': 2,
                    'POSITIVE': 1,
                    'NEGATIVE': 0,
                    'NEUTRAL': 2
                }
                
                pred_label = sentiment_to_label.get(sentiment_result['sentiment'], 2)
                predictions.append(pred_label)
                confidences.append(sentiment_result['confidence'])
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_val, predictions, average='weighted', zero_division=0)
            
            # Store results
            training_results = {
                'task_type': 'classification',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_confidence': np.mean(confidences),
                'label_encoder': label_encoder,
                'num_classes': len(label_encoder.classes_),
                'class_names': label_encoder.classes_.tolist(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'confusion_matrix': confusion_matrix(y_val, predictions).tolist()
            }
            
            self.training_results['classification'] = training_results
            self.logger.info(f"Classification training completed. Accuracy: {accuracy:.4f}")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in classification training: {e}")
            raise
    
    async def train_qa_model(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate Question-Answering model performance using the provided dataset
        """
        try:
            self.logger.info("Starting QA model evaluation...")
            
            required_columns = ['question', 'context', 'answer']
            missing_columns = [col for col in required_columns if col not in dataset.columns]
            if missing_columns:
                # Try alternative column names
                if 'text' in dataset.columns:
                    dataset['context'] = dataset['text']
                else:
                    raise ValueError(f"Missing required columns for QA: {missing_columns}")
            
            # Prepare data
            questions = dataset['question'].tolist()
            contexts = dataset['context'].tolist()
            true_answers = dataset['answer'].tolist()
            
            # Split data
            split_idx = int(0.8 * len(questions))
            train_questions = questions[:split_idx]
            train_contexts = contexts[:split_idx]
            train_answers = true_answers[:split_idx]
            
            val_questions = questions[split_idx:]
            val_contexts = contexts[split_idx:]
            val_answers = true_answers[split_idx:]
            
            # Evaluate on validation set
            predictions = []
            confidences = []
            
            for question, context in zip(val_questions, val_contexts):
                try:
                    response = await self.ai_models.answer_question(question, context)
                    predictions.append(response.content)
                    confidences.append(response.confidence)
                except Exception as e:
                    self.logger.warning(f"Error processing QA pair: {e}")
                    predictions.append("")
                    confidences.append(0.0)
            
            # Calculate BLEU-like similarity (simplified)
            similarities = []
            for pred, true_ans in zip(predictions, val_answers):
                # Simple token-based similarity
                pred_tokens = set(pred.lower().split())
                true_tokens = set(true_ans.lower().split())
                
                if len(true_tokens) == 0:
                    similarity = 1.0 if len(pred_tokens) == 0 else 0.0
                else:
                    similarity = len(pred_tokens.intersection(true_tokens)) / len(true_tokens.union(pred_tokens))
                
                similarities.append(similarity)
            
            # Store results
            qa_results = {
                'task_type': 'question_answering',
                'average_similarity': np.mean(similarities),
                'average_confidence': np.mean(confidences),
                'training_samples': len(train_questions),
                'validation_samples': len(val_questions),
                'sample_predictions': list(zip(val_questions[:5], predictions[:5], val_answers[:5]))
            }
            
            self.training_results['qa'] = qa_results
            self.logger.info(f"QA evaluation completed. Average similarity: {np.mean(similarities):.4f}")
            
            return qa_results
            
        except Exception as e:
            self.logger.error(f"Error in QA evaluation: {e}")
            raise
    
    async def validate_model_performance(self, validation_dataset: pd.DataFrame, model_type: str = 'all') -> Dict[str, Any]:
        """
        Validate model performance on a separate validation dataset
        """
        try:
            self.logger.info(f"Starting model validation for {model_type}")
            
            validation_results = {
                'validation_timestamp': time.time(),
                'validation_samples': len(validation_dataset),
                'model_type': model_type
            }
            
            if model_type in ['all', 'classification']:
                if 'label' in validation_dataset.columns:
                    classification_results = await self.train_classification_model(validation_dataset, 'label')
                    validation_results['classification'] = classification_results
            
            if model_type in ['all', 'qa']:
                if all(col in validation_dataset.columns for col in ['question', 'answer']):
                    qa_results = await self.train_qa_model(validation_dataset)
                    validation_results['qa'] = qa_results
            
            if model_type in ['all', 'sentiment']:
                if 'text' in validation_dataset.columns:
                    sentiment_results = await self.validate_sentiment_analysis(validation_dataset)
                    validation_results['sentiment'] = sentiment_results
            
            self.validation_results[model_type] = validation_results
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {e}")
            raise
    
    async def validate_sentiment_analysis(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Validate sentiment analysis performance"""
        try:
            texts = dataset['text'].tolist()
            predictions = []
            confidences = []
            
            for text in texts:
                result = await self.ai_models.analyze_sentiment(text)
                predictions.append(result['sentiment'])
                confidences.append(result['confidence'])
            
            # If we have true sentiment labels, calculate accuracy
            accuracy = None
            if 'sentiment' in dataset.columns:
                true_sentiments = dataset['sentiment'].tolist()
                accuracy = accuracy_score(true_sentiments, predictions)
            
            return {
                'task_type': 'sentiment_analysis',
                'accuracy': accuracy,
                'average_confidence': np.mean(confidences),
                'prediction_distribution': pd.Series(predictions).value_counts().to_dict(),
                'sample_predictions': list(zip(texts[:5], predictions[:5]))
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment validation: {e}")
            raise
    
    async def test_model_robustness(self, test_dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Test model robustness with various edge cases and adversarial examples
        """
        try:
            self.logger.info("Starting robustness testing...")
            
            robustness_tests = {
                'empty_text': [],
                'very_long_text': [],
                'special_characters': [],
                'multilingual': [],
                'processing_times': []
            }
            
            # Test different scenarios
            test_cases = [
                ("", "empty_text"),
                ("a" * 10000, "very_long_text"),
                ("!@#$%^&*()_+{}|:<>?", "special_characters"),
                ("Hello world! Bonjour le monde! Привет мир!", "multilingual")
            ]
            
            for text, category in test_cases:
                start_time = time.time()
                try:
                    # Test embedding generation
                    embeddings = await self.ai_models.generate_embeddings([text])
                    
                    # Test sentiment analysis
                    sentiment = await self.ai_models.analyze_sentiment(text)
                    
                    # Test summarization
                    summary = await self.ai_models.summarize_text(text)
                    
                    processing_time = time.time() - start_time
                    
                    result = {
                        'text_length': len(text),
                        'embedding_shape': embeddings.shape if embeddings is not None else None,
                        'sentiment': sentiment['sentiment'],
                        'sentiment_confidence': sentiment['confidence'],
                        'summary_length': len(summary.content),
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    robustness_tests[category].append(result)
                    robustness_tests['processing_times'].append(processing_time)
                    
                except Exception as e:
                    result = {
                        'text_length': len(text),
                        'error': str(e),
                        'success': False,
                        'processing_time': time.time() - start_time
                    }
                    robustness_tests[category].append(result)
                    robustness_tests['processing_times'].append(time.time() - start_time)
            
            # Calculate overall robustness metrics
            total_tests = sum(len(tests) for tests in robustness_tests.values() if isinstance(tests, list))
            successful_tests = sum(
                sum(1 for test in tests if isinstance(test, dict) and test.get('success', False))
                for tests in robustness_tests.values()
                if isinstance(tests, list)
            )
            
            robustness_summary = {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / max(total_tests, 1),
                'average_processing_time': np.mean(robustness_tests['processing_times']),
                'max_processing_time': np.max(robustness_tests['processing_times']),
                'min_processing_time': np.min(robustness_tests['processing_times']),
                'detailed_results': robustness_tests
            }
            
            self.test_results['robustness'] = robustness_summary
            self.logger.info(f"Robustness testing completed. Success rate: {robustness_summary['success_rate']:.2%}")
            
            return robustness_summary
            
        except Exception as e:
            self.logger.error(f"Error in robustness testing: {e}")
            raise
    
    async def test_indian_language_detection(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Test Indian language detection accuracy and performance
        """
        try:
            self.logger.info("Starting Indian language detection testing...")
            
            language_detector = IndianLanguageDetector()
            texts = dataset['text'].tolist()
            
            # Test language detection on each text
            detection_results = []
            processing_times = []
            
            for text in texts:
                start_time = time.time()
                try:
                    detection = language_detector.detect_language(text)
                    processing_time = time.time() - start_time
                    
                    result = {
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'detected_language': detection.language_code,
                        'language_name': detection.language_name,
                        'native_name': detection.native_name,
                        'confidence': detection.confidence,
                        'script': detection.script,
                        'family': detection.family,
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    # Check accuracy if we have true language labels
                    if 'language' in dataset.columns:
                        true_lang = dataset[dataset['text'] == text]['language'].iloc[0]
                        result['true_language'] = true_lang
                        result['correct_prediction'] = (detection.language_code == true_lang)
                    
                    detection_results.append(result)
                    processing_times.append(processing_time)
                    
                except Exception as e:
                    result = {
                        'text': text[:100] + '...' if len(text) > 100 else text,
                        'error': str(e),
                        'processing_time': time.time() - start_time,
                        'success': False
                    }
                    detection_results.append(result)
                    processing_times.append(time.time() - start_time)
            
            # Calculate metrics
            successful_detections = [r for r in detection_results if r.get('success', False)]
            language_distribution = {}
            confidence_scores = []
            
            for result in successful_detections:
                lang = result['detected_language']
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                confidence_scores.append(result['confidence'])
            
            # Calculate accuracy if true labels are available
            accuracy = None
            if 'language' in dataset.columns:
                correct_predictions = [r for r in successful_detections 
                                     if r.get('correct_prediction', False)]
                accuracy = len(correct_predictions) / len(successful_detections) if successful_detections else 0
            
            language_test_results = {
                'task_type': 'indian_language_detection',
                'total_samples': len(texts),
                'successful_detections': len(successful_detections),
                'success_rate': len(successful_detections) / len(texts) if texts else 0,
                'accuracy': accuracy,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'average_processing_time': np.mean(processing_times),
                'language_distribution': language_distribution,
                'supported_languages': language_detector.get_supported_languages(),
                'sample_detections': detection_results[:10]  # First 10 for review
            }
            
            self.test_results['indian_language_detection'] = language_test_results
            self.logger.info(f"Language detection testing completed. Success rate: {language_test_results['success_rate']:.2%}")
            
            return language_test_results
            
        except Exception as e:
            self.logger.error(f"Error in language detection testing: {e}")
            raise
    
    async def test_multilingual_summary_generation(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Test multilingual summary generation for Indian languages
        """
        try:
            self.logger.info("Starting multilingual summary generation testing...")
            
            texts = dataset['text'].tolist()
            summary_results = []
            processing_times = []
            
            # Test various Indian languages if available in dataset
            indian_languages = {'hi', 'kn', 'mr', 'te', 'ta', 'bn', 'gu', 'pa', 'ml', 'or', 'as'}
            
            for i, text in enumerate(texts):
                start_time = time.time()
                try:
                    # Detect language first
                    detected_lang = await self.ai_models.detect_language_advanced(text)
                    
                    # Generate summary with potential bilingual output
                    summary_response = await self.ai_models.summarize_text(
                        text, 
                        max_length=200, 
                        min_length=50,
                        language=detected_lang
                    )
                    
                    processing_time = time.time() - start_time
                    
                    result = {
                        'text_sample': text[:150] + '...' if len(text) > 150 else text,
                        'detected_language': detected_lang,
                        'is_indian_language': detected_lang in indian_languages,
                        'summary_content': summary_response.content,
                        'is_bilingual': summary_response.metadata.get('bilingual', False),
                        'confidence': summary_response.confidence,
                        'processing_time': processing_time,
                        'original_length': len(text),
                        'summary_length': len(summary_response.content),
                        'compression_ratio': len(summary_response.content) / len(text) if text else 0,
                        'success': True
                    }
                    
                    summary_results.append(result)
                    processing_times.append(processing_time)
                    
                except Exception as e:
                    result = {
                        'text_sample': text[:150] + '...' if len(text) > 150 else text,
                        'error': str(e),
                        'processing_time': time.time() - start_time,
                        'success': False
                    }
                    summary_results.append(result)
                    processing_times.append(time.time() - start_time)
            
            # Calculate metrics
            successful_summaries = [r for r in summary_results if r.get('success', False)]
            bilingual_summaries = [r for r in successful_summaries if r.get('is_bilingual', False)]
            indian_language_texts = [r for r in successful_summaries if r.get('is_indian_language', False)]
            
            # Language distribution
            language_distribution = {}
            confidence_scores = []
            compression_ratios = []
            
            for result in successful_summaries:
                lang = result.get('detected_language', 'unknown')
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                confidence_scores.append(result['confidence'])
                compression_ratios.append(result['compression_ratio'])
            
            multilingual_test_results = {
                'task_type': 'multilingual_summary_generation',
                'total_samples': len(texts),
                'successful_summaries': len(successful_summaries),
                'success_rate': len(successful_summaries) / len(texts) if texts else 0,
                'bilingual_summaries': len(bilingual_summaries),
                'bilingual_rate': len(bilingual_summaries) / len(successful_summaries) if successful_summaries else 0,
                'indian_language_texts': len(indian_language_texts),
                'indian_language_rate': len(indian_language_texts) / len(successful_summaries) if successful_summaries else 0,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'average_processing_time': np.mean(processing_times),
                'average_compression_ratio': np.mean(compression_ratios) if compression_ratios else 0,
                'language_distribution': language_distribution,
                'supported_indian_languages': list(indian_languages),
                'sample_summaries': summary_results[:10]  # First 10 for review
            }
            
            self.test_results['multilingual_summary_generation'] = multilingual_test_results
            self.logger.info(f"Multilingual summary testing completed. Success rate: {multilingual_test_results['success_rate']:.2%}, Bilingual rate: {multilingual_test_results['bilingual_rate']:.2%}")
            
            return multilingual_test_results
            
        except Exception as e:
            self.logger.error(f"Error in multilingual summary testing: {e}")
            raise
    
    async def test_multilingual_qa(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Test multilingual question-answering for Indian languages
        """
        try:
            self.logger.info("Starting multilingual QA testing...")
            
            required_columns = ['question', 'context', 'answer']
            missing_columns = [col for col in required_columns if col not in dataset.columns]
            if missing_columns:
                # Try alternative column names
                if 'text' in dataset.columns:
                    dataset['context'] = dataset['text']
                else:
                    raise ValueError(f"Missing required columns for multilingual QA: {missing_columns}")
            
            questions = dataset['question'].tolist()
            contexts = dataset['context'].tolist()
            true_answers = dataset['answer'].tolist()
            
            qa_results = []
            processing_times = []
            indian_languages = {'hi', 'kn', 'mr', 'te', 'ta', 'bn', 'gu', 'pa', 'ml', 'or', 'as'}
            
            for question, context, true_answer in zip(questions, contexts, true_answers):
                start_time = time.time()
                try:
                    # Detect context language
                    detected_lang = await self.ai_models.detect_language_advanced(context)
                    
                    # Get answer with potential bilingual response
                    response = await self.ai_models.answer_question(
                        question, context, language=detected_lang
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Simple similarity calculation
                    pred_tokens = set(response.content.lower().split())
                    true_tokens = set(true_answer.lower().split())
                    
                    if len(true_tokens) == 0:
                        similarity = 1.0 if len(pred_tokens) == 0 else 0.0
                    else:
                        similarity = len(pred_tokens.intersection(true_tokens)) / len(true_tokens.union(pred_tokens))
                    
                    result = {
                        'question': question,
                        'context_sample': context[:200] + '...' if len(context) > 200 else context,
                        'true_answer': true_answer,
                        'predicted_answer': response.content,
                        'detected_language': detected_lang,
                        'is_indian_language': detected_lang in indian_languages,
                        'is_bilingual': response.metadata.get('bilingual', False),
                        'confidence': response.confidence,
                        'similarity': similarity,
                        'processing_time': processing_time,
                        'success': True
                    }
                    
                    qa_results.append(result)
                    processing_times.append(processing_time)
                    
                except Exception as e:
                    result = {
                        'question': question,
                        'context_sample': context[:200] + '...' if len(context) > 200 else context,
                        'error': str(e),
                        'processing_time': time.time() - start_time,
                        'success': False
                    }
                    qa_results.append(result)
                    processing_times.append(time.time() - start_time)
            
            # Calculate metrics
            successful_qa = [r for r in qa_results if r.get('success', False)]
            bilingual_responses = [r for r in successful_qa if r.get('is_bilingual', False)]
            indian_language_contexts = [r for r in successful_qa if r.get('is_indian_language', False)]
            
            similarities = [r['similarity'] for r in successful_qa if 'similarity' in r]
            confidences = [r['confidence'] for r in successful_qa if 'confidence' in r]
            
            # Language distribution
            language_distribution = {}
            for result in successful_qa:
                lang = result.get('detected_language', 'unknown')
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
            
            multilingual_qa_results = {
                'task_type': 'multilingual_question_answering',
                'total_samples': len(questions),
                'successful_responses': len(successful_qa),
                'success_rate': len(successful_qa) / len(questions) if questions else 0,
                'bilingual_responses': len(bilingual_responses),
                'bilingual_rate': len(bilingual_responses) / len(successful_qa) if successful_qa else 0,
                'indian_language_contexts': len(indian_language_contexts),
                'indian_language_rate': len(indian_language_contexts) / len(successful_qa) if successful_qa else 0,
                'average_similarity': np.mean(similarities) if similarities else 0,
                'average_confidence': np.mean(confidences) if confidences else 0,
                'average_processing_time': np.mean(processing_times),
                'language_distribution': language_distribution,
                'sample_qa_pairs': qa_results[:5]  # First 5 for review
            }
            
            self.test_results['multilingual_qa'] = multilingual_qa_results
            self.logger.info(f"Multilingual QA testing completed. Success rate: {multilingual_qa_results['success_rate']:.2%}, Bilingual rate: {multilingual_qa_results['bilingual_rate']:.2%}")
            
            return multilingual_qa_results
            
        except Exception as e:
            self.logger.error(f"Error in multilingual QA testing: {e}")
            raise
    
    def save_results(self, output_path: str = None):
        """Save all training, testing, and validation results to JSON files"""
        try:
            if output_path is None:
                output_path = Path(__file__).parent / "results"
            else:
                output_path = Path(output_path)
            
            output_path.mkdir(exist_ok=True)
            
            # Save training results
            if self.training_results:
                with open(output_path / "training_results.json", "w") as f:
                    json.dump(self.training_results, f, indent=2, default=str)
            
            # Save test results
            if self.test_results:
                with open(output_path / "test_results.json", "w") as f:
                    json.dump(self.test_results, f, indent=2, default=str)
            
            # Save validation results
            if self.validation_results:
                with open(output_path / "validation_results.json", "w") as f:
                    json.dump(self.validation_results, f, indent=2, default=str)
            
            # Save combined summary
            summary = {
                'timestamp': time.time(),
                'training_results': self.training_results,
                'test_results': self.test_results,
                'validation_results': self.validation_results
            }
            
            with open(output_path / "complete_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    async def run_complete_training_pipeline(
        self,
        training_csv: str,
        test_csv: str = None,
        validation_csv: str = None,
        model_types: List[str] = ['classification', 'qa', 'sentiment']
    ):
        """
        Run the complete training, testing, and validation pipeline
        """
        try:
            self.logger.info("Starting complete ML pipeline...")
            
            # Initialize models
            await self.initialize_models()
            
            # Load datasets
            training_data = self.load_csv_dataset(training_csv, 'training')
            
            if test_csv:
                test_data = self.load_csv_dataset(test_csv, 'testing')
            
            if validation_csv:
                validation_data = self.load_csv_dataset(validation_csv, 'validation')
            
            # Run training/evaluation for each model type
            for model_type in model_types:
                self.logger.info(f"Processing {model_type} model...")
                
                if model_type == 'classification' and 'label' in training_data.columns:
                    await self.train_classification_model(training_data)
                
                elif model_type == 'qa' and all(col in training_data.columns for col in ['question', 'answer']):
                    await self.train_qa_model(training_data)
                
                elif model_type == 'sentiment':
                    await self.validate_sentiment_analysis(training_data)
            
            # Run robustness tests if test data is available
            if test_csv:
                await self.test_model_robustness(test_data)
            
            # Run validation if validation data is available
            if validation_csv:
                await self.validate_model_performance(validation_data)
            
            # Save all results
            self.save_results()
            
            self.logger.info("Complete ML pipeline finished successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in complete pipeline: {e}")
            raise


# Utility function for easy usage
async def run_ml_training(training_csv: str, test_csv: str = None, validation_csv: str = None):
    """
    Easy-to-use function to run the complete ML training pipeline
    """
    framework = MLTrainingFramework()
    await framework.run_complete_training_pipeline(
        training_csv=training_csv,
        test_csv=test_csv,
        validation_csv=validation_csv
    )
    return framework


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="PolyDoc ML Training Framework")
    parser.add_argument("--training-csv", required=True, help="Path to training CSV file")
    parser.add_argument("--test-csv", help="Path to test CSV file")
    parser.add_argument("--validation-csv", help="Path to validation CSV file")
    parser.add_argument("--model-types", nargs="+", default=['classification', 'qa', 'sentiment'],
                        help="Types of models to train/evaluate")
    
    args = parser.parse_args()
    
    # Run the training pipeline
    asyncio.run(run_ml_training(
        training_csv=args.training_csv,
        test_csv=args.test_csv,
        validation_csv=args.validation_csv
    ))
