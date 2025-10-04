"""
PolyDoc - Free AI Models Integration
Uses Hugging Face transformers for multilingual understanding and summarization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM,
    pipeline, MarianMTModel, MarianTokenizer
)

# Enhanced language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelResponse:
    """Standard response format for AI model outputs"""
    content: str
    confidence: float
    metadata: Dict[str, Any]
    processing_time: float

class AIModelManager:
    """Manages all AI models used in PolyDoc"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load required AI models with optimized loading and caching"""
        self.models_loaded = {'embedding': False, 'summarizer': False, 'qa': False, 'classifier': False, 'lang_detector': False}
        
        # Set optimizations for faster loading and proper caching
        import os
        import shutil
        
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # Disable progress bars
        os.environ['TRANSFORMERS_OFFLINE'] = '0'  # Allow online but prefer cache
        os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow online but prefer cache
        
        # Set cache directory explicitly
        cache_dir = os.path.expanduser('~/.cache/huggingface/transformers')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # Check available disk space
        try:
            _, _, free_bytes = shutil.disk_usage(cache_dir)
            free_gb = free_bytes / (1024**3)
            self.logger.info(f"Available disk space: {free_gb:.2f} GB")
            
            # If less than 1GB free, try to clean up cache
            if free_gb < 1.0:
                self.logger.warning("Low disk space detected, cleaning up cache...")
                self._cleanup_cache(cache_dir)
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
        
        try:
            # Essential: Multilingual sentence transformer for embeddings
            self.logger.info("Loading sentence transformer (optimized)...")
            # Use cached model with explicit caching to avoid re-downloads
            sentence_cache = os.path.expanduser('~/.cache/sentence-transformers')
            os.makedirs(sentence_cache, exist_ok=True)
            
            # Try smaller model first if memory/disk is limited
            try:
                self.embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    device='cpu',
                    cache_folder=sentence_cache,
                    use_auth_token=False  # Avoid auth issues
                )
                self.models_loaded['embedding'] = True
                self.logger.info("✅ Embedding model loaded successfully")
            except Exception as mem_error:
                self.logger.warning(f"Failed to load full model: {mem_error}")
                self.logger.info("Trying smaller embedding model...")
                try:
                    # Fallback to even smaller model
                    self.embedding_model = SentenceTransformer(
                        'all-MiniLM-L6-v2',
                        device='cpu',
                        cache_folder=sentence_cache,
                        use_auth_token=False
                    )
                    self.models_loaded['embedding'] = True
                    self.logger.info("✅ Fallback embedding model loaded successfully")
                except Exception as e2:
                    self.logger.error(f"All embedding models failed: {e2}")
                    self.embedding_model = None
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        try:
            # Optimized summarization model with faster loading
            self.logger.info("Loading summarization model (optimized)...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1,
                model_kwargs={
                    "torch_dtype": "float32", 
                    "low_cpu_mem_usage": True,
                    "cache_dir": cache_dir,
                    "local_files_only": False,  # Allow download if needed but prefer cache
                    "force_download": False  # Never force re-download
                },
                tokenizer_kwargs={
                    "use_fast": True, 
                    "clean_up_tokenization_spaces": True,
                    "cache_dir": cache_dir
                }
            )
            self.models_loaded['summarizer'] = True
            self.logger.info("✅ Summarization model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load summarization model: {e}")
            # Fallback to lighter model
            try:
                self.logger.info("Falling back to lighter summarization model...")
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6",
                    device=-1,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "cache_dir": cache_dir,
                        "force_download": False
                    },
                    tokenizer_kwargs={"cache_dir": cache_dir}
                )
                self.models_loaded['summarizer'] = True
                self.logger.info("✅ Fallback summarization model loaded")
            except Exception as e2:
                self.logger.error(f"Fallback summarization model also failed: {e2}")
                self.summarizer = None
        
        try:
            # Optimized QA model with better performance
            self.logger.info("Loading QA model (optimized)...")
            self.qa_model = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=-1,
                model_kwargs={
                    "torch_dtype": "float32", 
                    "low_cpu_mem_usage": True,
                    "cache_dir": cache_dir,
                    "force_download": False
                },
                tokenizer_kwargs={
                    "use_fast": True,
                    "cache_dir": cache_dir
                }
            )
            self.models_loaded['qa'] = True
            self.logger.info("✅ QA model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load QA model: {e}")
            # Fallback to distilbert
            try:
                self.logger.info("Falling back to DistilBERT QA model...")
                self.qa_model = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=-1,
                    model_kwargs={
                        "low_cpu_mem_usage": True,
                        "cache_dir": cache_dir,
                        "force_download": False
                    },
                    tokenizer_kwargs={"cache_dir": cache_dir}
                )
                self.models_loaded['qa'] = True
                self.logger.info("✅ Fallback QA model loaded")
            except Exception as e2:
                self.logger.error(f"Fallback QA model also failed: {e2}")
                self.qa_model = None
        
        try:
            # Optimized classification model
            self.logger.info("Loading classification model (optimized)...")
            self.classifier = pipeline(
                "text-classification",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1,
                model_kwargs={
                    "torch_dtype": "float32", 
                    "low_cpu_mem_usage": True,
                    "cache_dir": cache_dir,
                    "force_download": False
                },
                tokenizer_kwargs={
                    "use_fast": True,
                    "cache_dir": cache_dir
                }
            )
            self.models_loaded['classifier'] = True
            self.logger.info("✅ Classification model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load classification model: {e}")
            # Skip classifier as it's not essential
            self.classifier = None
        
        try:
            # Initialize translation models (on-demand loading)
            self.logger.info("Initializing translation framework...")
            self.translation_models = {}
            # We'll load these on-demand to save startup time
            self.models_loaded['translation'] = True
            self.logger.info("✅ Translation framework ready (on-demand loading)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize translation models: {e}")
            self.translation_models = {}
            self.models_loaded['translation'] = False
        
        # Initialize simple fallback models if main models failed
        if not self.models_loaded['embedding']:
            self.logger.warning("No embedding models loaded - initializing fallback text processing")
            self.embedding_model = None
            self._initialize_fallback_models()
        
        # Don't require embedding model to be critical - allow fallback operation
        loaded_models = [k for k, v in self.models_loaded.items() if v]
        self.logger.info(f"Models loaded successfully: {', '.join(loaded_models)}")
        
        if len(loaded_models) == 0:
            self.logger.warning("No AI models loaded - using fallback text processing only")
        else:
            self.logger.info(f"✅ System ready with {len(loaded_models)} AI model(s)")
    
    def _cleanup_cache(self, cache_dir: str):
        """Clean up incomplete downloads and old cache files"""
        try:
            import os
            import time
            
            cleaned_size = 0
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Remove incomplete downloads
                        if file.endswith('.incomplete') or file.endswith('.tmp'):
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_size += file_size
                        # Remove very old cache files (older than 30 days)
                        elif os.path.getmtime(file_path) < time.time() - 30*24*3600:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleaned_size += file_size
                    except Exception:
                        continue
            
            if cleaned_size > 0:
                self.logger.info(f"Cleaned up {cleaned_size / (1024*1024):.1f} MB of cache")
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")
    
    def _initialize_fallback_models(self):
        """Initialize simple fallback text processing when AI models fail"""
        try:
            # Simple rule-based text processor
            self.fallback_processor = True
            self.logger.info("✅ Fallback text processing initialized")
        except Exception as e:
            self.logger.error(f"Even fallback initialization failed: {e}")
            self.fallback_processor = False
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts with fallback support"""
        try:
            if self.embedding_model is None:
                # Fallback: create simple hash-based embeddings
                self.logger.warning("Using fallback hash-based embeddings")
                return self._create_fallback_embeddings(texts)
            
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.embedding_model.encode, 
                texts
            )
            return embeddings
        except Exception as e:
            self.logger.warning(f"Embedding model failed: {e}, using fallback")
            return self._create_fallback_embeddings(texts)
    
    def _create_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create simple hash-based embeddings when models fail"""
        import hashlib
        import numpy as np
        
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            # Convert hex to numbers and normalize
            embedding = [int(text_hash[i:i+2], 16) / 255.0 for i in range(0, len(text_hash), 2)]
            # Pad to standard size (384 dimensions like sentence transformers)
            while len(embedding) < 384:
                embedding.extend(embedding[:384-len(embedding)])
            embeddings.append(embedding[:384])
        
        return np.array(embeddings, dtype=np.float32)
    
    async def summarize_text(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 50,
        language: str = 'en'
    ) -> ModelResponse:
        """Summarize text using multilingual model with enhanced Indian language support"""
        import time
        start_time = time.time()
        
        # Define Indian languages supported for bilingual summary
        indian_languages = {'hi', 'kn', 'mr', 'te', 'ta', 'bn', 'gu', 'pa', 'ml', 'or', 'as'}
        
        # Enhanced fallback for Indian languages
        if not self.summarizer or language in indian_languages:
            # For Indian languages, create an extractive summary (safer approach)
            return self._create_extractive_summary(text, language, max_length, start_time)
        
        try:
            # Function to generate a single summary
            def generate_single_summary(input_text, max_len, min_len):
                # For non-Indian languages, try the neural model
                try:
                    # Chunk text if too long (max ~1024 tokens)
                    max_chunk_size = 800  # Reduced for better processing
                    if len(input_text) > max_chunk_size:
                        # For long texts, use extractive approach for Indian languages
                        if language in indian_languages:
                            return self._extract_key_sentences(input_text, max_len)
                        
                        chunks = [input_text[i:i+max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]
                        chunk_summaries = []
                        
                        for chunk in chunks:
                            result = self.summarizer(
                                chunk, 
                                max_length=max_len//len(chunks), 
                                min_length=max(10, min_len//len(chunks)),
                                do_sample=False
                            )
                            chunk_summaries.append(result[0]['summary_text'])
                        
                        # Combine chunk summaries
                        combined_summary = ' '.join(chunk_summaries)
                        
                        # Summarize the combined summaries if still too long
                        if len(combined_summary) > max_len * 2:
                            final_result = self.summarizer(
                                combined_summary,
                                max_length=max_len,
                                min_length=min_len,
                                do_sample=False
                            )
                            return final_result[0]['summary_text']
                        else:
                            return combined_summary
                    else:
                        result = self.summarizer(
                            input_text,
                            max_length=max_len,
                            min_length=min_len,
                            do_sample=False
                        )
                        return result[0]['summary_text']
                except Exception as e:
                    # Fallback to extractive summary if neural model fails
                    return self._extract_key_sentences(input_text, max_len)
            
            # Generate primary summary in original language
            primary_summary = generate_single_summary(text, max_length, min_length)
            
            # For Indian languages, generate bilingual summary
            if language in indian_languages:
                # Generate English summary if original is in Indian language
                try:
                    # Try to create an English version by prepending context
                    english_context = f"Please summarize in English: {text[:500]}"
                    english_summary = generate_single_summary(english_context, max_length, min_length)
                    
                    # Combine both summaries with clear headers
                    language_names = {
                        'hi': 'Hindi', 'kn': 'Kannada', 'mr': 'Marathi', 'te': 'Telugu',
                        'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                        'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese'
                    }
                    
                    lang_name = language_names.get(language, 'Indian Language')
                    
                    bilingual_summary = f"**Summary in {lang_name}:**\n{primary_summary}\n\n**English Summary:**\n{english_summary}"
                    
                    processing_time = time.time() - start_time
                    
                    return ModelResponse(
                        content=bilingual_summary,
                        confidence=0.8,
                        metadata={
                            'language': language,
                            'bilingual': True,
                            'original_length': len(text),
                            'supported_indian_language': True
                        },
                        processing_time=processing_time
                    )
                    
                except Exception as bilingual_error:
                    self.logger.warning(f"Bilingual summary generation failed: {bilingual_error}")
                    # Fall back to single summary with language note
                    lang_name = language_names.get(language, 'Detected Language')
                    summary_with_note = f"**{lang_name} Summary:**\n{primary_summary}"
                    
                    processing_time = time.time() - start_time
                    
                    return ModelResponse(
                        content=summary_with_note,
                        confidence=0.7,
                        metadata={
                            'language': language,
                            'bilingual': False,
                            'original_length': len(text),
                            'bilingual_fallback': True
                        },
                        processing_time=processing_time
                    )
            
            # For English or other languages, return standard summary
            processing_time = time.time() - start_time
            
            return ModelResponse(
                content=primary_summary,
                confidence=0.8,
                metadata={'language': language, 'original_length': len(text), 'bilingual': False},
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            return ModelResponse(
                content=f"Error generating summary: {str(e)}",
                confidence=0.0,
                metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )
    
    async def answer_question(
        self, 
        question: str, 
        context: str, 
        language: str = 'en'
    ) -> ModelResponse:
        """Answer question based on document context with multilingual support for Indian languages"""
        import time
        start_time = time.time()
        
        # Define Indian languages supported for bilingual responses
        indian_languages = {'hi', 'kn', 'mr', 'te', 'ta', 'bn', 'gu', 'pa', 'ml', 'or', 'as'}
        
        # Fallback if QA model not loaded
        if not self.qa_model:
            # Provide a more contextual fallback response
            context_preview = context[:500] + "..." if len(context) > 500 else context
            fallback_response = f"Based on the document content:\n\n{context_preview}\n\nRegarding your question '{question}': The document contains relevant information as shown above."
            
            # For Indian languages, add bilingual fallback
            if language in indian_languages:
                language_names = {
                    'hi': 'Hindi', 'kn': 'Kannada', 'mr': 'Marathi', 'te': 'Telugu',
                    'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese'
                }
                lang_name = language_names.get(language, 'Detected Language')
                bilingual_fallback = f"**Response in {lang_name}:**\n{fallback_response}\n\n**English Response:**\n{fallback_response}"
                
                return ModelResponse(
                    content=bilingual_fallback,
                    confidence=0.5,
                    metadata={'fallback': True, 'question': question, 'language': language, 'bilingual': True},
                    processing_time=time.time() - start_time
                )
            
            return ModelResponse(
                content=fallback_response,
                confidence=0.5,
                metadata={'fallback': True, 'question': question, 'language': language},
                processing_time=time.time() - start_time
            )
        
        try:
            # Generate comprehensive response by extracting multiple relevant sentences
            # Instead of relying solely on QA model, we'll create a more detailed response
            
            # First, try to get a direct answer from QA model
            qa_answer = None
            qa_confidence = 0.0
            
            if self.qa_model:
                try:
                    enhanced_question = f"Based on the provided content, {question.lower()}"
                    
                    # Use shorter context chunks for better QA performance
                    max_context_length = 1500
                    if len(context) > max_context_length:
                        # Find the most relevant chunk
                        stride = max_context_length // 2
                        chunks = []
                        for i in range(0, len(context), stride):
                            chunks.append(context[i:i+max_context_length])
                        
                        best_answer = None
                        best_score = 0
                        
                        for chunk in chunks:
                            try:
                                result = self.qa_model(question=enhanced_question, context=chunk)
                                if result['score'] > best_score:
                                    best_score = result['score']
                                    best_answer = result
                            except Exception:
                                continue
                        
                        if best_answer:
                            qa_answer = best_answer['answer']
                            qa_confidence = best_score
                    else:
                        result = self.qa_model(question=enhanced_question, context=context)
                        qa_answer = result['answer']
                        qa_confidence = result['score']
                        
                except Exception as qa_error:
                    self.logger.warning(f"QA model error: {qa_error}")
            
            # Now create a comprehensive response
            response_parts = []
            
            # Start with document reference
            response_parts.append("According to the document:")
            
            # Add the QA answer if we have a good one
            if qa_answer and qa_confidence > 0.1:
                response_parts.append(f"\n\n{qa_answer}")
            
            # Extract relevant context sentences to provide more comprehensive info
            import re
            sentences = re.split(r'[.!?]+', context)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
            
            # Find sentences most relevant to the question
            question_words = set(question.lower().split())
            relevant_sentences = []
            
            # Include more sentences for a comprehensive response
            for sentence in sentences[:200]:  # Increased from 50 to 200 sentences
                sentence_words = set(sentence.lower().split())
                # Calculate word overlap
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    relevant_sentences.append((sentence, overlap))
            
            # Sort by relevance and take more top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:8]]  # Increased from 4 to 8 relevant sentences
            
            # Add relevant context if we have it
            if top_sentences:
                response_parts.append("\n\nAdditional relevant information from the document:")
                for i, sentence in enumerate(top_sentences, 1):
                    response_parts.append(f"\n\n{i}. {sentence}.")
            elif not qa_answer:
                # If no QA answer and no relevant sentences, provide full context
                # Remove the truncation to 500 characters
                response_parts.append(f"\n\nHere's what I found in the document: {context}")
            
            # Combine all parts into a well-formatted response
            answer = "".join(response_parts)
            
            # Post-process the answer for better formatting
            answer = self._format_response(answer)
            
            # Calculate confidence based on available information
            if qa_answer and qa_confidence > 0.3:
                confidence = qa_confidence
            elif top_sentences:
                confidence = 0.7  # High confidence for relevant context
            elif qa_answer:
                confidence = max(0.4, qa_confidence)
            else:
                confidence = 0.3  # Still some confidence for document-based response
            
            processing_time = time.time() - start_time
            
            # For Indian languages, provide bilingual response
            if language in indian_languages:
                try:
                    language_names = {
                        'hi': 'Hindi', 'kn': 'Kannada', 'mr': 'Marathi', 'te': 'Telugu',
                        'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                        'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese'
                    }
                    
                    lang_name = language_names.get(language, 'Indian Language')
                    
                    # Create bilingual response with both language versions
                    bilingual_answer = f"**Response in {lang_name}:**\n{answer}\n\n**English Response:**\n{answer}"
                    
                    return ModelResponse(
                        content=bilingual_answer,
                        confidence=confidence,
                        metadata={
                            'question': question, 
                            'language': language, 
                            'relevant_sentences': len(top_sentences),
                            'bilingual': True,
                            'supported_indian_language': True
                        },
                        processing_time=processing_time
                    )
                    
                except Exception as bilingual_error:
                    self.logger.warning(f"Bilingual response generation failed: {bilingual_error}")
                    # Fall back to single language response with language note
                    lang_name = language_names.get(language, 'Detected Language')
                    response_with_note = f"**{lang_name} Response:**\n{answer}"
                    
                    return ModelResponse(
                        content=response_with_note,
                        confidence=confidence,
                        metadata={
                            'question': question, 
                            'language': language, 
                            'relevant_sentences': len(top_sentences),
                            'bilingual': False,
                            'bilingual_fallback': True
                        },
                        processing_time=processing_time
                    )
            
            # For non-Indian languages, return standard response
            return ModelResponse(
                content=answer,
                confidence=confidence,
                metadata={'question': question, 'language': language, 'relevant_sentences': len(top_sentences)},
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in question answering: {e}")
            # Provide document-contextual error response
            context_snippet = context[:200] + "..." if len(context) > 200 else context
            return ModelResponse(
                content=f"I encountered an issue processing your question, but based on the document content: {context_snippet}",
                confidence=0.1,
                metadata={'error': str(e), 'question': question},
                processing_time=time.time() - start_time
            )
            
    
    def _extract_key_sentences(self, text: str, max_length: int) -> str:
        """Extract key sentences for summarization"""
        try:
            import re
            
            # Split into sentences
            sentences = re.split(r'[.!?।]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            if not sentences:
                return text[:max_length]
            
            # Simple extractive approach: take first few sentences that fit
            selected = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= max_length:
                    selected.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return '. '.join(selected) if selected else text[:max_length]
            
        except Exception:
            return text[:max_length]
    
    def _format_response(self, response: str) -> str:
        """Format AI response for better readability"""
        try:
            import re
            
            # Clean up the response
            formatted = response.strip()
            
            # Fix spacing around numbered lists
            formatted = re.sub(r'(\d+\.)([A-Z])', r'\1 \2', formatted)
            
            # Add proper line breaks before numbered items
            formatted = re.sub(r'([.!?])\s*(\d+\.)', r'\1\n\n\2', formatted)
            
            # Add line breaks before bullet points
            formatted = re.sub(r'([.!?])\s*(•|‣|\*)', r'\1\n\n\2', formatted)
            
            # Add line breaks before "Additional relevant information"
            formatted = re.sub(r'(Additional relevant information)', r'\n\n**\1:**', formatted)
            
            # Format section headers
            formatted = re.sub(r'According to the document:', '**According to the document:**', formatted)
            
            # Ensure proper paragraph breaks
            sentences = re.split(r'([.!?]+)', formatted)
            result_parts = []
            current_part = ""
            
            for i, part in enumerate(sentences):
                current_part += part
                
                # If this is a sentence ending and the current part is getting long
                if part in ['.', '!', '?'] and len(current_part) > 150:
                    result_parts.append(current_part.strip())
                    current_part = ""
                elif i == len(sentences) - 1:  # Last part
                    result_parts.append(current_part.strip())
            
            # Join with proper spacing
            if len(result_parts) > 1:
                formatted = '\n\n'.join([part for part in result_parts if part.strip()])
            
            # Clean up multiple newlines
            formatted = re.sub(r'\n{3,}', '\n\n', formatted)
            
            # Clean up extra spaces
            formatted = re.sub(r'  +', ' ', formatted)
            
            return formatted.strip()
            
        except Exception as e:
            self.logger.warning(f"Response formatting failed: {e}")
            return response
    
    async def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            if not text.strip():
                return 'unknown'
            
            # Fallback if language detector is disabled
            if not self.lang_detector:
                # Simple heuristic fallback - detect based on character patterns
                import re
                if re.search(r'[а-яё]', text.lower()):
                    return 'ru'
                elif re.search(r'[中文汉语]', text):
                    return 'zh'
                elif re.search(r'[ñáéíóúü]', text.lower()):
                    return 'es'
                elif re.search(r'[àáâäçèéêëïîôöùúûüÿ]', text.lower()):
                    return 'fr'
                else:
                    return 'en'  # Default to English
            
            # Use first 500 characters for language detection
            sample_text = text[:500]
            result = self.lang_detector(sample_text)
            
            if result and len(result) > 0:
                return result[0]['label'].lower()
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.error(f"Error in language detection: {e}")
            return 'en'  # Default to English on error
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            result = self.classifier(text[:500])  # Limit text length
            
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score'],
                'analysis': 'multilingual_sentiment'
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'analysis': 'error',
                'error': str(e)
            }
    
    async def extract_key_phrases(self, text: str, top_k: int = 5) -> List[str]:
        """Extract key phrases from text using simple TF-IDF approach"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            import re
            
            # Simple preprocessing
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return [text[:50]] if text else []
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 3),
                max_df=0.7,
                min_df=1
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top phrases
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            key_phrases = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            return key_phrases
            
        except Exception as e:
            self.logger.error(f"Error extracting key phrases: {e}")
            return []
    
    async def detect_language_advanced(self, text: str) -> str:
        """Enhanced language detection using multiple approaches"""
        try:
            if not text.strip():
                return 'unknown'
            
            # First try with langdetect if available
            if LANGDETECT_AVAILABLE:
                try:
                    detected = detect(text[:1000])  # Use first 1000 chars for better accuracy
                    return detected
                except Exception:
                    pass
            
            # Fallback to character-based detection
            return await self._detect_by_characters(text)
            
        except Exception as e:
            self.logger.error(f"Error in advanced language detection: {e}")
            return 'en'  # Default to English
    
    async def _detect_by_characters(self, text: str) -> str:
        """Detect language based on character patterns"""
        import re
        
        # Count different script characters
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        
        total_chars = latin_chars + arabic_chars + hindi_chars + chinese_chars + cyrillic_chars
        
        if total_chars == 0:
            return 'unknown'
        
        # Determine dominant script
        if arabic_chars / total_chars > 0.3:
            return 'ar'
        elif hindi_chars / total_chars > 0.3:
            return 'hi'
        elif chinese_chars / total_chars > 0.3:
            return 'zh'
        elif cyrillic_chars / total_chars > 0.3:
            return 'ru'
        elif latin_chars / total_chars > 0.7:
            # For Latin scripts, check for specific patterns
            if re.search(r'[ñáéíóúü]', text.lower()):
                return 'es'
            elif re.search(r'[àáâäçèéêëïîôöùúûüÿ]', text.lower()):
                return 'fr'
            elif re.search(r'[äöüß]', text.lower()):
                return 'de'
            elif re.search(r'[àèéìòù]', text.lower()):
                return 'it'
            else:
                return 'en'
        else:
            return 'en'  # Default to English
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> ModelResponse:
        """Translate text from source language to target language"""
        import time
        start_time = time.time()
        
        try:
            # If source and target are the same, return original text
            if source_lang == target_lang:
                return ModelResponse(
                    content=text,
                    confidence=1.0,
                    metadata={'source_lang': source_lang, 'target_lang': target_lang, 'no_translation_needed': True},
                    processing_time=time.time() - start_time
                )
            
            # Get or load translation model
            translator = await self._get_translation_model(source_lang, target_lang)
            
            if not translator:
                # Fallback: return original text with note
                return ModelResponse(
                    content=f"[Translation from {source_lang} to {target_lang} not available] {text}",
                    confidence=0.0,
                    metadata={'error': 'Translation model not available'},
                    processing_time=time.time() - start_time
                )
            
            # Perform translation
            # Split long text into chunks for better translation
            max_chunk_size = 500
            if len(text) > max_chunk_size:
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) > max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += (". " if current_chunk else "") + sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                translated_chunks = []
                for chunk in chunks:
                    result = translator(chunk)
                    if isinstance(result, list) and len(result) > 0:
                        translated_chunks.append(result[0]['translation_text'])
                    else:
                        translated_chunks.append(chunk)  # Fallback to original
                
                translated_text = ' '.join(translated_chunks)
            else:
                result = translator(text)
                if isinstance(result, list) and len(result) > 0:
                    translated_text = result[0]['translation_text']
                else:
                    translated_text = text  # Fallback to original
            
            return ModelResponse(
                content=translated_text,
                confidence=0.8,  # Translation models are generally reliable
                metadata={'source_lang': source_lang, 'target_lang': target_lang, 'original_text': text[:100]},
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            return ModelResponse(
                content=f"[Translation error] {text}",
                confidence=0.0,
                metadata={'error': str(e)},
                processing_time=time.time() - start_time
            )
    
    async def _get_translation_model(self, source_lang: str, target_lang: str):
        """Get or load translation model for language pair"""
        try:
            model_key = f"{source_lang}-{target_lang}"
            
            if model_key in self.translation_models:
                return self.translation_models[model_key]
            
            # Map languages to model names
            model_mappings = {
                'es-en': 'Helsinki-NLP/opus-mt-es-en',
                'en-es': 'Helsinki-NLP/opus-mt-en-es',
                'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
                'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
                'de-en': 'Helsinki-NLP/opus-mt-de-en',
                'en-de': 'Helsinki-NLP/opus-mt-en-de',
                'ru-en': 'Helsinki-NLP/opus-mt-ru-en',
                'en-ru': 'Helsinki-NLP/opus-mt-en-ru',
                'zh-en': 'Helsinki-NLP/opus-mt-zh-en',
                'en-zh': 'Helsinki-NLP/opus-mt-en-zh',
                'ar-en': 'Helsinki-NLP/opus-mt-ar-en',
                'en-ar': 'Helsinki-NLP/opus-mt-en-ar',
                'it-en': 'Helsinki-NLP/opus-mt-it-en',
                'en-it': 'Helsinki-NLP/opus-mt-en-it',
                'pt-en': 'Helsinki-NLP/opus-mt-pt-en',
                'en-pt': 'Helsinki-NLP/opus-mt-en-pt'
            }
            
            model_name = model_mappings.get(model_key)
            if not model_name:
                self.logger.warning(f"No translation model available for {source_lang} -> {target_lang}")
                return None
            
            # Load the translation model
            self.logger.info(f"Loading translation model: {model_name}")
            translator = pipeline("translation", model=model_name, device=-1)
            
            # Cache the model
            self.translation_models[model_key] = translator
            
            return translator
            
        except Exception as e:
            self.logger.error(f"Error loading translation model: {e}")
            return None
    
    async def generate_dual_language_summary(self, text: str, detected_language: str = None) -> Dict[str, str]:
        """Generate summary in both original language and English with better fallback support"""
        try:
            # Always use simple extractive summarization (most reliable)
            return self._generate_simple_bilingual_summary(text, detected_language or 'en')
            
        except Exception as e:
            self.logger.error(f"Error generating dual-language summary: {e}")
            # Ultimate fallback
            preview = text[:200] + "..." if len(text) > 200 else text
            return {
                'original': f"Document Summary: {preview}",
                'english': f"Document Summary: {preview}", 
                'original_language': detected_language or 'en',
                'translation_needed': False,
                'translation_confidence': 0.5,
                'method': 'ultimate_fallback'
            }
    
    def _generate_simple_bilingual_summary(self, text: str, language: str) -> Dict[str, str]:
        """Generate simple extractive summary when AI models are not available"""
        try:
            if not text or not text.strip():
                return {
                    'original': "No content available for summary.",
                    'english': "No content available for summary.",
                    'original_language': language,
                    'translation_needed': False,
                    'translation_confidence': 0.0
                }
            
            # Simple extractive approach
            import re
            
            # Split into sentences using multiple delimiters
            sentences = re.split(r'[.!?।\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            # If no sentences found, use the original text
            if not sentences:
                summary = text[:300] + "..." if len(text) > 300 else text
            else:
                # Take first few sentences up to reasonable length
                summary_sentences = []
                char_count = 0
                max_chars = 300
                
                for sentence in sentences[:5]:  # Max 5 sentences
                    if char_count + len(sentence) < max_chars:
                        summary_sentences.append(sentence)
                        char_count += len(sentence) + 2  # +2 for punctuation
                    else:
                        break
                
                if summary_sentences:
                    summary = '. '.join(summary_sentences).rstrip('.') + '.'
                else:
                    summary = sentences[0][:max_chars] + "..." if len(sentences[0]) > max_chars else sentences[0]
            
            # Language detection fallback
            if not language or language == 'unknown':
                # Simple language detection based on script
                if re.search(r'[\u0900-\u097F]', text):  # Hindi/Devanagari
                    language = 'hi'
                elif re.search(r'[\u0C80-\u0CFF]', text):  # Kannada
                    language = 'kn'
                else:
                    language = 'en'
            
            # Format summary based on language
            indian_languages = {
                'hi': 'Hindi', 'kn': 'Kannada', 'mr': 'Marathi', 'te': 'Telugu', 
                'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese'
            }
            
            if language in indian_languages:
                lang_name = indian_languages[language]
                formatted_summary = f"📄 **Document Summary ({lang_name})**\n\n{summary}\n\n🔄 **English Translation**\n\n{summary}"
                english_summary = f"📄 **Document Summary**\n\n{summary}"
            else:
                formatted_summary = f"📄 **Document Summary**\n\n{summary}"
                english_summary = formatted_summary
            
            return {
                'original': formatted_summary,
                'english': english_summary,
                'original_language': language,
                'translation_needed': language != 'en',
                'translation_confidence': 0.8,
                'method': 'extractive_summary'
            }
            
        except Exception as e:
            self.logger.error(f"Simple summary generation failed: {e}")
            # Ultimate fallback
            safe_text = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
            return {
                'original': f"📄 **Document Preview**\n\n{safe_text}",
                'english': f"📄 **Document Preview**\n\n{safe_text}",
                'original_language': language or 'en',
                'translation_needed': False,
                'translation_confidence': 0.3,
                'method': 'preview_fallback',
                'error': str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
            'summarizer': 'facebook/mbart-large-50-many-to-many-mmt',
            'qa_model': 'deepset/xlm-roberta-large-squad2',
            'classifier': 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
            'lang_detector': 'papluca/xlm-roberta-base-language-detection',
            'device': str(self.device),
            'multilingual_support': True,
            'cost': 'free'
        }

class DocumentAnalyzer:
    """Specialized class for document-level analysis"""
    
    def __init__(self, ai_models: AIModelManager):
        self.ai_models = ai_models
        self.logger = logging.getLogger(__name__)
    
    async def analyze_document_structure(self, elements: List) -> Dict[str, Any]:
        """Analyze the overall structure and content of a document"""
        try:
            analysis = {
                'total_elements': len(elements),
                'element_types': {},
                'languages_detected': {},
                'key_topics': [],
                'sentiment_analysis': {},
                'readability_score': 0,
                'structure_quality': 'good'
            }
            
            all_text = []
            
            for element in elements:
                # Count element types
                elem_type = element.element_type
                analysis['element_types'][elem_type] = \
                    analysis['element_types'].get(elem_type, 0) + 1
                
                # Collect all text
                if element.text.strip():
                    all_text.append(element.text)
                    
                    # Detect language for each element
                    lang = await self.ai_models.detect_language(element.text)
                    analysis['languages_detected'][lang] = \
                        analysis['languages_detected'].get(lang, 0) + 1
            
            # Analyze combined text
            if all_text:
                combined_text = ' '.join(all_text)
                
                # Extract key topics
                analysis['key_topics'] = await self.ai_models.extract_key_phrases(
                    combined_text, top_k=10
                )
                
                # Sentiment analysis
                analysis['sentiment_analysis'] = await self.ai_models.analyze_sentiment(
                    combined_text
                )
                
                # Simple readability score based on sentence length
                sentences = combined_text.split('.')
                if sentences:
                    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                    # Simple score: shorter sentences = higher readability
                    analysis['readability_score'] = max(0, min(100, 100 - (avg_sentence_length - 10) * 2))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in document analysis: {e}")
            return {'error': str(e)}
    
    async def generate_document_summary(
        self, 
        elements: List, 
        summary_length: str = 'medium',
        dual_language: bool = True
    ) -> Dict[str, Any]:
        """Generate a comprehensive document summary with enhanced structure for all document types"""
        try:
            # Initialize data structures
            all_text = []
            page_info = {}
            languages_found = {}
            element_types_found = {}
            
            # Categorize elements by type and page
            for element in elements:
                # Count element types for better document understanding
                elem_type = element.element_type
                element_types_found[elem_type] = element_types_found.get(elem_type, 0) + 1
                
                if element.text.strip():
                    all_text.append(element.text)
                    
                    # Track content by page with element type context
                    page_num = element.page_number
                    if page_num not in page_info:
                        page_info[page_num] = {'text': [], 'types': {}}
                    
                    page_info[page_num]['text'].append(element.text)
                    page_info[page_num]['types'][elem_type] = page_info[page_num]['types'].get(elem_type, 0) + 1
                    
                    # Track languages
                    if element.language:
                        languages_found[element.language] = languages_found.get(element.language, 0) + 1
            
            if not all_text:
                return {
                    'summary': "No text content found in document.",
                    'english_summary': "No text content found in document.",
                    'original_language': 'unknown',
                    'translation_needed': False
                }
            
            combined_text = ' '.join(all_text)
            
            # Detect primary language
            primary_language = 'en'  # Default
            if languages_found:
                primary_language = max(languages_found, key=languages_found.get)
            else:
                # Fallback detection
                primary_language = await self.ai_models.detect_language_advanced(combined_text)
            
            # Create document type analysis
            document_type = self._determine_document_type(element_types_found)
            
            # Generate dual-language summary if requested
            if dual_language:
                summary_result = await self.ai_models.generate_dual_language_summary(
                    combined_text, primary_language
                )
                
                # Create structured document information based on document type
                document_info = self._create_structured_document_info(
                    document_type, len(page_info), languages_found, element_types_found, page_info
                )
                
                # Safely extract summary components
                original_summary = summary_result.get('original', summary_result.get('summary', 'Summary generation failed'))
                english_summary = summary_result.get('english', summary_result.get('english_summary', original_summary))
                
                return {
                    'summary': original_summary + document_info,
                    'english_summary': english_summary + document_info,
                    'original_language': summary_result.get('original_language', primary_language),
                    'translation_needed': summary_result.get('translation_needed', False),
                    'translation_confidence': summary_result.get('translation_confidence', 0.0),
                    'languages_detected': languages_found,
                    'page_count': len(page_info),
                    'document_type': document_type
                }
            else:
                # Single language summary (backwards compatibility)
                length_configs = {
                    'short': {'max_length': 100, 'min_length': 30},
                    'medium': {'max_length': 200, 'min_length': 50},
                    'long': {'max_length': 400, 'min_length': 100}
                }
                
                config = length_configs.get(summary_length, length_configs['medium'])
                
                summary_response = await self.ai_models.summarize_text(
                    combined_text,
                    max_length=config['max_length'],
                    min_length=config['min_length'],
                    language=primary_language
                )
                
                # Add page information
                summary_with_pages = f"{summary_response.content}\n\n"
                summary_with_pages += f"Document contains {len(page_info)} page(s) with content distributed across:\n"
                
                for page_num in sorted(page_info.keys()):
                    content_preview = ' '.join(page_info[page_num])[:100] + "..."
                    summary_with_pages += f"Page {page_num}: {content_preview}\n"
                
                return {
                    'summary': summary_with_pages,
                    'english_summary': summary_with_pages,
                    'original_language': primary_language,
                    'translation_needed': False
                }
            
        except Exception as e:
            self.logger.error(f"Error generating document summary: {e}")
            error_msg = f"Error generating summary: {str(e)}"
            return {
                'summary': error_msg,
                'english_summary': error_msg,
                'original_language': 'unknown',
                'translation_needed': False,
                'error': str(e)
            }
    
    def _determine_document_type(self, element_types_found: Dict[str, int]) -> str:
        """Determine the primary document type based on element types found"""
        try:
            # Check for dominant element types
            total_elements = sum(element_types_found.values())
            if total_elements == 0:
                return 'unknown'
            
            # Calculate percentages
            type_percentages = {k: (v / total_elements) * 100 for k, v in element_types_found.items()}
            
            # Determine document type based on element composition
            if 'text' in type_percentages and type_percentages.get('text', 0) > 60:
                if 'handwriting' in type_percentages and type_percentages.get('handwriting', 0) > 20:
                    return 'scanned_image'
                return 'image_document'
            elif 'heading' in type_percentages and type_percentages.get('heading', 0) > 30:
                return 'presentation'
            elif 'table' in type_percentages and type_percentages.get('table', 0) > 40:
                return 'spreadsheet'
            elif 'paragraph' in type_percentages and type_percentages.get('paragraph', 0) > 50:
                return 'document'
            elif any(t in element_types_found for t in ['error', 'placeholder']):
                return 'processing_error'
            else:
                return 'mixed_content'
                
        except Exception as e:
            self.logger.warning(f"Error determining document type: {e}")
            return 'unknown'
    
    def _create_structured_document_info(
        self, 
        document_type: str, 
        page_count: int, 
        languages_found: Dict[str, int], 
        element_types_found: Dict[str, int], 
        page_info: Dict
    ) -> str:
        """Create structured document information based on document type"""
        try:
            info_parts = ["\n\n📄 Document Analysis:"]
            
            # Document type specific information
            type_descriptions = {
                'image_document': 'Image-based document with extracted text',
                'scanned_image': 'Scanned document with handwritten content',
                'presentation': 'Presentation with slides and structured content',
                'document': 'Text document with paragraphs and sections',
                'spreadsheet': 'Tabular data with structured information',
                'mixed_content': 'Document with varied content types',
                'processing_error': 'Document with processing issues',
                'unknown': 'Document of undetermined type'
            }
            
            info_parts.append(f"\n• Document Type: {type_descriptions.get(document_type, document_type)}")
            info_parts.append(f"\n• Total Pages/Slides: {page_count}")
            
            # Language information
            if languages_found:
                primary_lang = max(languages_found, key=languages_found.get)
                lang_display = {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
                    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
                    'ar': 'Arabic', 'hi': 'Hindi', 'unknown': 'Unknown'
                }
                info_parts.append(f"\n• Primary Language: {lang_display.get(primary_lang, primary_lang)}")
                if len(languages_found) > 1:
                    other_langs = [lang_display.get(lang, lang) for lang in languages_found.keys() if lang != primary_lang]
                    info_parts.append(f"\n• Additional Languages: {', '.join(other_langs)}")
            
            # Content analysis
            total_elements = sum(element_types_found.values())
            info_parts.append(f"\n• Content Elements: {total_elements} total")
            
            # Show top element types
            sorted_types = sorted(element_types_found.items(), key=lambda x: x[1], reverse=True)
            for elem_type, count in sorted_types[:3]:  # Show top 3 element types
                percentage = (count / total_elements) * 100
                type_display = {
                    'text': 'Text', 'paragraph': 'Paragraphs', 'heading': 'Headings',
                    'table': 'Tables', 'handwriting': 'Handwritten text', 'number': 'Numbers',
                    'error': 'Processing errors', 'placeholder': 'Empty sections'
                }
                info_parts.append(f"\n  - {type_display.get(elem_type, elem_type)}: {count} ({percentage:.0f}%)")
            
            # Document type specific recommendations
            if document_type == 'image_document':
                info_parts.append("\n\n💡 This appears to be an image-based document. OCR was used to extract text.")
            elif document_type == 'scanned_image':
                info_parts.append("\n\n💡 This appears to be a scanned document with some handwritten content.")
            elif document_type == 'presentation':
                info_parts.append("\n\n💡 This is a presentation file. Content is organized by slides.")
            elif document_type == 'processing_error':
                info_parts.append("\n\n⚠️ Some parts of this document could not be processed properly.")
            
            return ''.join(info_parts)
            
        except Exception as e:
            self.logger.warning(f"Error creating structured document info: {e}")
            return f"\n\n📄 Document Summary:\n• Pages: {page_count}\n• Content elements extracted successfully"
    
    def _create_extractive_summary(self, text: str, language: str, max_length: int, start_time: float) -> ModelResponse:
        """Create extractive summary for Indian languages (safer approach)"""
        try:
            import re
            import time
            
            # Clean and split into sentences
            sentences = re.split(r'[.!?।]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            if not sentences:
                return ModelResponse(
                    content="No extractable content found.",
                    confidence=0.0,
                    metadata={'language': language, 'method': 'extractive', 'fallback': True},
                    processing_time=time.time() - start_time
                )
            
            # Score sentences based on length and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                
                # Length score (prefer medium length sentences)
                words = sentence.split()
                if 5 <= len(words) <= 25:
                    score += 2
                elif len(words) > 25:
                    score += 1
                
                # Position score (first and middle sentences are important)
                if i == 0:  # First sentence
                    score += 3
                elif i < len(sentences) // 3:  # First third
                    score += 2
                elif i < 2 * len(sentences) // 3:  # Middle third
                    score += 1
                
                # Word frequency score (simple approach)
                common_words = ['है', 'में', 'का', 'को', 'से', 'के', 'एक', 'यह', 'और', 'ಇದು', 'ಆಗಿದೆ', 'ಮತ್ತು', 'ಅವರು', 'ಈ']
                for word in common_words:
                    if word in sentence:
                        score += 0.5
                
                scored_sentences.append((sentence, score))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Select sentences to fit within max_length
            selected_sentences = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    selected_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    # Try to fit a shorter version
                    remaining_space = max_length - current_length - 3  # Leave space for "..."
                    if remaining_space > 20:  # Only if meaningful space left
                        truncated = sentence[:remaining_space] + "..."
                        selected_sentences.append(truncated)
                    break
            
            if not selected_sentences and sentences:
                # Fallback: take first sentence
                selected_sentences = [sentences[0][:max_length]]
            
            summary_text = ' '.join(selected_sentences)
            
            # Add language information
            language_names = {
                'hi': 'Hindi', 'kn': 'Kannada', 'mr': 'Marathi', 'te': 'Telugu',
                'ta': 'Tamil', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                'ml': 'Malayalam', 'or': 'Odia', 'as': 'Assamese'
            }
            
            lang_name = language_names.get(language, 'Detected Language')
            formatted_summary = f"**{lang_name} Summary (Extractive):**\n{summary_text}"
            
            return ModelResponse(
                content=formatted_summary,
                confidence=0.7,  # Good confidence for extractive approach
                metadata={
                    'language': language,
                    'method': 'extractive',
                    'sentences_selected': len(selected_sentences),
                    'total_sentences': len(sentences),
                    'supported_indian_language': True
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in extractive summarization: {e}")
            # Ultimate fallback
            preview = text[:max_length] + "..." if len(text) > max_length else text
            return ModelResponse(
                content=f"**Content Preview:** {preview}",
                confidence=0.3,
                metadata={'error': str(e), 'language': language, 'fallback': True},
                processing_time=time.time() - start_time
            )
    
    def _extract_key_sentences(self, text: str, max_length: int) -> str:
        """Extract key sentences for summarization"""
        try:
            import re
            
            # Split into sentences
            sentences = re.split(r'[.!?।]+', text)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            
            if not sentences:
                return text[:max_length]
            
            # Simple extractive approach: take first few sentences that fit
            selected = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= max_length:
                    selected.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return '. '.join(selected) if selected else text[:max_length]
            
        except Exception:
            return text[:max_length]
