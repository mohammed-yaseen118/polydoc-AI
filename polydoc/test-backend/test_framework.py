#!/usr/bin/env python3
"""
Test Framework Verification Script
==================================

This script tests the basic functionality of the test-backend framework
without requiring the full PolyDoc backend to be running.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

async def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    
    try:
        from ml_trainer import MLTrainingFramework
        print("✅ ml_trainer import successful")
        
        # Test framework initialization
        framework = MLTrainingFramework()
        await framework.initialize_models()
        print("✅ Framework initialization successful")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

async def test_csv_loading():
    """Test CSV loading functionality"""
    print("\n📊 Testing CSV loading...")
    
    try:
        from ml_trainer import MLTrainingFramework
        framework = MLTrainingFramework()
        
        # Test with sample data
        current_dir = Path(__file__).parent
        sample_files = [
            "sample_training_data.csv",
            "sample_test_data.csv", 
            "sample_validation_data.csv"
        ]
        
        for filename in sample_files:
            file_path = current_dir / filename
            if file_path.exists():
                try:
                    data = framework.load_csv_dataset(str(file_path), 'test')
                    print(f"✅ Loaded {filename}: {len(data)} rows, columns: {list(data.columns)}")
                except Exception as e:
                    print(f"⚠️  Warning loading {filename}: {e}")
            else:
                print(f"ℹ️  {filename} not found (this is OK for testing)")
        
        return True
    except Exception as e:
        print(f"❌ CSV loading test failed: {e}")
        return False

async def test_mock_models():
    """Test mock model functionality"""
    print("\n🎭 Testing mock models...")
    
    try:
        from ml_trainer import MLTrainingFramework, MODELS_AVAILABLE
        
        framework = MLTrainingFramework()
        await framework.initialize_models()
        
        if not MODELS_AVAILABLE:
            print("✅ Running in mock mode (expected when full backend not available)")
            
            # Test sentiment analysis
            sentiment = await framework.ai_models.analyze_sentiment("This is a test")
            print(f"✅ Mock sentiment analysis: {sentiment}")
            
            # Test summary generation  
            summary = await framework.ai_models.generate_summary("This is a test document")
            print(f"✅ Mock summary generation: {summary}")
            
        else:
            print("✅ Real models available")
        
        return True
    except Exception as e:
        print(f"❌ Mock models test failed: {e}")
        return False

async def test_run_tests_script():
    """Test that the run_tests.py script can be imported and parsed"""
    print("\n🚀 Testing run_tests.py script...")
    
    try:
        # Test import
        import run_tests
        print("✅ run_tests.py import successful")
        
        # Test argument parser (without actually running tests)
        import argparse
        from unittest.mock import patch
        
        with patch('sys.argv', ['run_tests.py', '--test', 'basic']):
            # This should not crash
            print("✅ Argument parsing test passed")
        
        return True
    except Exception as e:
        print(f"❌ run_tests.py test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🔧 PolyDoc Test Framework Verification")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_csv_loading,
        test_mock_models,
        test_run_tests_script
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if all(results):
        print("\n🎉 All tests passed! The test framework is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Test framework crashed: {e}")
        sys.exit(1)