#!/usr/bin/env python3
"""
Test script to verify the _extract_key_sentences fix is working
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("🔍 Testing AIModelManager fix...")
    
    # Import the AIModelManager
    from src.models.ai_models import AIModelManager
    print("✅ Successfully imported AIModelManager")
    
    # Check if the method exists
    if hasattr(AIModelManager, '_extract_key_sentences'):
        print("✅ _extract_key_sentences method exists in AIModelManager")
        
        # Test the method
        manager = AIModelManager()
        print("✅ AIModelManager instantiated successfully")
        
        # Test the extract method
        test_text = "This is a test sentence. This is another test sentence for summarization. This is the third sentence."
        result = manager._extract_key_sentences(test_text, 100)
        print(f"✅ _extract_key_sentences method works: {result[:50]}...")
        
        print("\n🎉 FIX VERIFICATION SUCCESSFUL!")
        print("The _extract_key_sentences method is now properly implemented.")
        print("Your chat model should work correctly now.")
        
    else:
        print("❌ _extract_key_sentences method NOT found in AIModelManager")
        print("The fix was not applied correctly.")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("There might be an import or dependency issue.")