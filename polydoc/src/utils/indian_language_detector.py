"""
Indian Language Detection Module for PolyDoc AI
Specialized language detection for Indian languages with improved accuracy
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langdetect import detect, detect_langs, LangDetectException
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class LanguageDetection:
    """Language detection result with confidence and metadata"""
    language_code: str
    language_name: str
    native_name: str
    confidence: float
    script: str
    family: str

class IndianLanguageDetector:
    """
    Enhanced language detector focused on Indian languages with 
    script-based fallback detection
    """
    
    # Supported Indian languages with metadata
    INDIAN_LANGUAGES = {
        'hi': {
            'name': 'Hindi', 
            'native_name': 'हिन्दी',
            'script': 'Devanagari',
            'family': 'Indo-Aryan'
        },
        'kn': {
            'name': 'Kannada', 
            'native_name': 'ಕನ್ನಡ',
            'script': 'Kannada',
            'family': 'Dravidian'
        },
        'mr': {
            'name': 'Marathi', 
            'native_name': 'मराठी',
            'script': 'Devanagari',
            'family': 'Indo-Aryan'
        },
        'te': {
            'name': 'Telugu', 
            'native_name': 'తెలుగు',
            'script': 'Telugu',
            'family': 'Dravidian'
        },
        'ta': {
            'name': 'Tamil', 
            'native_name': 'தமிழ்',
            'script': 'Tamil',
            'family': 'Dravidian'
        },
        'bn': {
            'name': 'Bengali', 
            'native_name': 'বাংলা',
            'script': 'Bengali',
            'family': 'Indo-Aryan'
        },
        'gu': {
            'name': 'Gujarati', 
            'native_name': 'ગુજરાતી',
            'script': 'Gujarati',
            'family': 'Indo-Aryan'
        },
        'pa': {
            'name': 'Punjabi', 
            'native_name': 'ਪੰਜਾਬੀ',
            'script': 'Gurmukhi',
            'family': 'Indo-Aryan'
        },
        'ml': {
            'name': 'Malayalam', 
            'native_name': 'മലയാളം',
            'script': 'Malayalam',
            'family': 'Dravidian'
        },
        'or': {
            'name': 'Odia', 
            'native_name': 'ଓଡ଼ିଆ',
            'script': 'Odia',
            'family': 'Indo-Aryan'
        },
        'as': {
            'name': 'Assamese', 
            'native_name': 'অসমীয়া',
            'script': 'Bengali',
            'family': 'Indo-Aryan'
        },
        'en': {
            'name': 'English', 
            'native_name': 'English',
            'script': 'Latin',
            'family': 'Germanic'
        }
    }
    
    # Unicode script ranges for Indian languages
    SCRIPT_RANGES = {
        'Devanagari': [(0x0900, 0x097F)],
        'Bengali': [(0x0980, 0x09FF)],
        'Gurmukhi': [(0x0A00, 0x0A7F)],
        'Gujarati': [(0x0A80, 0x0AFF)],
        'Odia': [(0x0B00, 0x0B7F)],
        'Tamil': [(0x0B80, 0x0BFF)],
        'Telugu': [(0x0C00, 0x0C7F)],
        'Kannada': [(0x0C80, 0x0CFF)],
        'Malayalam': [(0x0D00, 0x0D7F)],
        'Latin': [(0x0041, 0x007A), (0x00C0, 0x024F)]
    }
    
    def __init__(self):
        """Initialize the Indian language detector"""
        logger.info("Initializing Indian Language Detector...")
        
    def _get_script_composition(self, text: str) -> Dict[str, float]:
        """Analyze the script composition of the text"""
        if not text:
            return {}
            
        script_counts = {}
        total_chars = 0
        
        for char in text:
            # Skip whitespace but include punctuation for better analysis
            if char.isspace():
                continue
                
            # Count all non-space characters
            total_chars += 1
            char_code = ord(char)
            
            # Find which script this character belongs to
            script_found = False
            for script, ranges in self.SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= char_code <= end:
                        script_counts[script] = script_counts.get(script, 0) + 1
                        script_found = True
                        break
                if script_found:
                    break
            
            # If character doesn't match any known script, count as 'Other'
            if not script_found and char.isalnum():
                script_counts['Other'] = script_counts.get('Other', 0) + 1
        
        if total_chars == 0:
            return {}
            
        # Convert counts to percentages
        script_percentages = {
            script: count / total_chars 
            for script, count in script_counts.items()
        }
        
        return script_percentages
    
    def _script_to_language(self, script: str, confidence: float) -> Optional[LanguageDetection]:
        """Map script to most likely language"""
        script_language_map = {
            'Devanagari': ['hi', 'mr'],  # Hindi more common
            'Bengali': ['bn', 'as'],     # Bengali more common
            'Kannada': ['kn'],
            'Telugu': ['te'],
            'Tamil': ['ta'],
            'Gujarati': ['gu'],
            'Gurmukhi': ['pa'],
            'Malayalam': ['ml'],
            'Odia': ['or'],
            'Latin': ['en']
        }
        
        languages = script_language_map.get(script, [])
        if not languages:
            return None
            
        # Return the first (most common) language for the script
        lang_code = languages[0]
        lang_info = self.INDIAN_LANGUAGES.get(lang_code)
        
        if lang_info:
            return LanguageDetection(
                language_code=lang_code,
                language_name=lang_info['name'],
                native_name=lang_info['native_name'],
                confidence=confidence,
                script=script,
                family=lang_info['family']
            )
        
        return None
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection"""
        if not text:
            return ""
            
        # Remove URLs, emails, numbers, and special characters
        cleaned = re.sub(r'http[s]?://\S+', '', text)
        cleaned = re.sub(r'\S+@\S+\.\S+', '', cleaned)
        cleaned = re.sub(r'\d+', '', cleaned)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    def detect_language(self, text: str) -> LanguageDetection:
        """
        Detect Indian language with fallback to script analysis
        Restricted to Indian languages and English only
        """
        if not text or not text.strip():
            return self._get_default_detection()
        
        # Clean text for better detection
        cleaned_text = self._clean_text_for_detection(text)
        
        if len(cleaned_text) < 3:
            return self._get_default_detection()
        
        # Method 1: Try langdetect first for supported languages
        try:
            detected_langs = detect_langs(cleaned_text)
            
            # Filter to only Indian languages and English
            indian_langs = [
                lang for lang in detected_langs 
                if lang.lang in self.INDIAN_LANGUAGES
            ]
            
            if indian_langs:
                # Get the most confident Indian/English language
                best_detection = max(indian_langs, key=lambda x: x.prob)
                
                lang_info = self.INDIAN_LANGUAGES[best_detection.lang]
                return LanguageDetection(
                    language_code=best_detection.lang,
                    language_name=lang_info['name'],
                    native_name=lang_info['native_name'],
                    confidence=best_detection.prob,
                    script=lang_info['script'],
                    family=lang_info['family']
                )
            else:
                # No Indian languages detected, check if it might be misclassified
                # Fall back to script-based detection
                pass
        
        except LangDetectException as e:
            logger.debug(f"LangDetect failed: {e}, falling back to script detection")
        
        # Method 2: Script-based detection fallback for mixed content
        script_composition = self._get_script_composition(text)
        
        if script_composition:
            # Check for Indian scripts even if they're not dominant
            indian_scripts = ['Devanagari', 'Kannada', 'Telugu', 'Tamil', 'Bengali', 'Gujarati', 'Gurmukhi', 'Malayalam', 'Odia']
            
            for script in indian_scripts:
                if script in script_composition and script_composition[script] > 0.1:  # Lowered threshold for mixed content
                    detection = self._script_to_language(script, script_composition[script])
                    if detection:
                        logger.info(f"Script-based detection (mixed content): {detection.language_name} ({script_composition[script]:.2%} {script} script)")
                        return detection
            
            # If no Indian scripts found, check for dominant script
            dominant_script = max(script_composition.items(), key=lambda x: x[1])
            script_name, percentage = dominant_script
            
            if percentage > 0.2:
                detection = self._script_to_language(script_name, percentage)
                if detection:
                    logger.info(f"Script-based detection: {detection.language_name} ({percentage:.2%} confidence)")
                    return detection
        
        # Method 3: Default to English if no Indian script detected
        logger.info("No Indian language detected, defaulting to English")
        return self._get_default_detection()
    
    def _get_default_detection(self) -> LanguageDetection:
        """Return default English detection"""
        return LanguageDetection(
            language_code='en',
            language_name='English',
            native_name='English',
            confidence=0.5,
            script='Latin',
            family='Germanic'
        )
    
    def detect_multiple_languages(self, text: str, threshold: float = 0.1) -> List[LanguageDetection]:
        """
        Detect multiple languages in text above threshold
        Restricted to Indian languages and English only
        """
        results = []
        
        try:
            # Get all language probabilities
            lang_probs = detect_langs(text)
            
            # Filter to only Indian languages and English, above threshold
            for prob in lang_probs:
                if prob.prob >= threshold and prob.lang in self.INDIAN_LANGUAGES:
                    lang_info = self.INDIAN_LANGUAGES[prob.lang]
                    results.append(LanguageDetection(
                        language_code=prob.lang,
                        language_name=lang_info['name'],
                        native_name=lang_info['native_name'],
                        confidence=prob.prob,
                        script=lang_info['script'],
                        family=lang_info['family']
                    ))
            
            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)
            
            # If no Indian languages found above threshold, try script-based detection
            if not results:
                script_detection = self.detect_language(text)
                if script_detection.language_code in self.INDIAN_LANGUAGES:
                    results = [script_detection]
            
        except LangDetectException:
            # Fallback to single detection
            single_detection = self.detect_language(text)
            results = [single_detection]
        
        return results if results else [self._get_default_detection()]
    
    def is_indian_language(self, language_code: str) -> bool:
        """Check if language code is an Indian language"""
        return language_code in self.INDIAN_LANGUAGES and language_code != 'en'
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get all supported languages with metadata"""
        return self.INDIAN_LANGUAGES.copy()
    
    def get_language_info(self, language_code: str) -> Optional[Dict[str, str]]:
        """Get detailed information about a language"""
        return self.INDIAN_LANGUAGES.get(language_code)

# Global instance
_detector_instance = None

def get_indian_language_detector() -> IndianLanguageDetector:
    """Get global Indian language detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IndianLanguageDetector()
    return _detector_instance

# Convenience functions
def detect_indian_language(text: str) -> LanguageDetection:
    """Detect Indian language in text"""
    detector = get_indian_language_detector()
    return detector.detect_language(text)

def is_indian_language(language_code: str) -> bool:
    """Check if language is Indian"""
    detector = get_indian_language_detector()
    return detector.is_indian_language(language_code)
