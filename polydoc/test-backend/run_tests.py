#!/usr/bin/env python3
"""
PolyDoc Test Runner
==================

Simple script to run various ML training and testing pipelines with sample data.
"""

import asyncio
import os
import sys
import argparse
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from ml_trainer import MLTrainingFramework, run_ml_training

async def run_basic_test():
    """Run basic training pipeline with sample data"""
    print("🚀 Starting Basic ML Pipeline Test...")
    
    # Get file paths
    current_dir = Path(__file__).parent
    training_csv = current_dir / "sample_training_data.csv"
    test_csv = current_dir / "sample_test_data.csv"
    validation_csv = current_dir / "sample_validation_data.csv"
    
    # Check if files exist
    for file_path, name in [(training_csv, "training"), (test_csv, "test"), (validation_csv, "validation")]:
        if not file_path.exists():
            print(f"❌ Error: {name} file not found at {file_path}")
            return False
    
    try:
        # Run the training pipeline
        framework = await run_ml_training(
            training_csv=str(training_csv),
            test_csv=str(test_csv),
            validation_csv=str(validation_csv)
        )
        
        print("✅ Basic ML Pipeline Test Completed Successfully!")
        
        # Print summary results
        if framework.training_results:
            print("\n📊 Training Results Summary:")
            for task, results in framework.training_results.items():
                if task == 'qa':
                    similarity = results.get('average_similarity', 0)
                    confidence = results.get('average_confidence', 0)
                    print(f"  {task}: {similarity:.3f} similarity, {confidence:.3f} confidence")
                elif task == 'sentiment':
                    accuracy = results.get('accuracy', 'N/A')
                    confidence = results.get('average_confidence', 0)
                    if accuracy == 'N/A':
                        print(f"  {task}: {confidence:.3f} average confidence")
                    else:
                        print(f"  {task}: {accuracy:.3f} accuracy, {confidence:.3f} confidence")
                else:
                    print(f"  {task}: {results.get('accuracy', 'N/A')} accuracy")
        
        if framework.test_results:
            print("\n🧪 Test Results Summary:")
            for task, results in framework.test_results.items():
                print(f"  {task}: {results.get('success_rate', 'N/A')} success rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during basic test: {e}")
        return False

async def run_classification_only_test():
    """Run classification-focused test"""
    print("🎯 Starting Classification-Only Test...")
    
    current_dir = Path(__file__).parent
    training_csv = current_dir / "sample_training_data.csv"
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        training_data = framework.load_csv_dataset(str(training_csv), 'training')
        
        if 'label' in training_data.columns:
            results = await framework.train_classification_model(training_data)
            print(f"✅ Classification Test Completed!")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   F1 Score: {results['f1_score']:.4f}")
            print(f"   Classes: {results['class_names']}")
        else:
            print("❌ No 'label' column found for classification")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during classification test: {e}")
        return False

async def run_qa_only_test():
    """Run Question-Answering focused test"""
    print("❓ Starting Question-Answering Test...")
    
    current_dir = Path(__file__).parent
    training_csv = current_dir / "sample_training_data.csv"
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        training_data = framework.load_csv_dataset(str(training_csv), 'training')
        
        required_cols = ['question', 'answer']
        if all(col in training_data.columns for col in required_cols):
            results = await framework.train_qa_model(training_data)
            print(f"✅ QA Test Completed!")
            print(f"   Average Similarity: {results['average_similarity']:.4f}")
            print(f"   Average Confidence: {results['average_confidence']:.4f}")
            print(f"   Sample predictions: {len(results['sample_predictions'])}")
        else:
            print(f"❌ Missing required columns for QA: {required_cols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during QA test: {e}")
        return False

async def run_robustness_test():
    """Run robustness testing only"""
    print("🛡️ Starting Robustness Test...")
    
    current_dir = Path(__file__).parent
    test_csv = current_dir / "sample_test_data.csv"
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        test_data = framework.load_csv_dataset(str(test_csv), 'testing')
        
        results = await framework.test_model_robustness(test_data)
        print(f"✅ Robustness Test Completed!")
        print(f"   Success Rate: {results['success_rate']:.2%}")
        print(f"   Avg Processing Time: {results['average_processing_time']:.4f}s")
        print(f"   Total Tests: {results['total_tests']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during robustness test: {e}")
        return False

async def run_custom_csv_test(csv_path: str):
    """Run test with custom CSV file"""
    print(f"📄 Starting Custom CSV Test with {csv_path}...")
    
    if not Path(csv_path).exists():
        print(f"❌ Error: File not found at {csv_path}")
        return False
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        data = framework.load_csv_dataset(csv_path, 'custom')
        
        print(f"✅ Loaded {len(data)} samples from custom CSV")
        print(f"   Columns: {list(data.columns)}")
        
        # Try different tests based on available columns
        results_summary = {}
        
        if 'label' in data.columns:
            print("   Running classification test...")
            cls_results = await framework.train_classification_model(data)
            results_summary['classification'] = cls_results['accuracy']
        
        if all(col in data.columns for col in ['question', 'answer']):
            print("   Running QA test...")
            qa_results = await framework.train_qa_model(data)
            results_summary['qa'] = qa_results['average_similarity']
        
        if 'text' in data.columns:
            print("   Running sentiment analysis...")
            sentiment_results = await framework.validate_sentiment_analysis(data)
            results_summary['sentiment'] = sentiment_results.get('accuracy', 'N/A')
        
        print(f"✅ Custom CSV Test Completed!")
        for test_type, score in results_summary.items():
            print(f"   {test_type}: {score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during custom CSV test: {e}")
        return False

async def run_multilingual_test():
    """Run multilingual summary generation and QA tests"""
    print("🌐 Starting Multilingual Features Test...")
    
    current_dir = Path(__file__).parent
    training_csv = current_dir / "sample_training_data.csv"
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        data = framework.load_csv_dataset(str(training_csv), 'multilingual')
        
        print(f"✅ Loaded {len(data)} samples for multilingual testing")
        
        # Test multilingual summary generation
        if 'text' in data.columns:
            print("   Running multilingual summary generation test...")
            summary_results = await framework.test_multilingual_summary_generation(data)
            print(f"   ✅ Summary Test - Success Rate: {summary_results['success_rate']:.2%}, Bilingual Rate: {summary_results['bilingual_rate']:.2%}")
            print(f"   📊 Indian Languages Detected: {summary_results['indian_language_rate']:.2%}")
        
        # Test multilingual QA if we have QA columns
        if all(col in data.columns for col in ['question', 'answer']):
            print("   Running multilingual QA test...")
            qa_results = await framework.test_multilingual_qa(data)
            print(f"   ✅ QA Test - Success Rate: {qa_results['success_rate']:.2%}, Bilingual Rate: {qa_results['bilingual_rate']:.2%}")
            print(f"   📊 Average Similarity: {qa_results['average_similarity']:.3f}")
        
        # Test Indian language detection
        if 'text' in data.columns:
            print("   Running Indian language detection test...")
            lang_results = await framework.test_indian_language_detection(data)
            print(f"   ✅ Language Detection - Success Rate: {lang_results['success_rate']:.2%}")
            if lang_results['accuracy']:
                print(f"   📊 Detection Accuracy: {lang_results['accuracy']:.2%}")
        
        # Save results
        framework.save_results()
        
        print(f"✅ Multilingual Features Test Completed!")
        print(f"   Results saved in results/ directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during multilingual test: {e}")
        return False

async def run_indian_language_test():
    """Run specific Indian language detection test"""
    print("🇮🇳 Starting Indian Language Detection Test...")
    
    current_dir = Path(__file__).parent
    training_csv = current_dir / "sample_training_data.csv"
    
    framework = MLTrainingFramework()
    
    try:
        await framework.initialize_models()
        data = framework.load_csv_dataset(str(training_csv), 'language')
        
        results = await framework.test_indian_language_detection(data)
        print(f"✅ Indian Language Detection Test Completed!")
        print(f"   Success Rate: {results['success_rate']:.2%}")
        print(f"   Average Confidence: {results['average_confidence']:.3f}")
        print(f"   Languages Detected: {list(results['language_distribution'].keys())}")
        
        if results['accuracy']:
            print(f"   Detection Accuracy: {results['accuracy']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during Indian language test: {e}")
        return False

async def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="PolyDoc ML Test Runner")
    # Support both old and new argument formats for backward compatibility
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test', choices=['basic', 'classification', 'qa', 'robustness', 'custom', 'multilingual', 'indian-language'], 
                           help='Type of test to run (deprecated, use --test-type)')
    test_group.add_argument('--test-type', choices=['basic', 'classification', 'qa', 'robustness', 'custom', 'multilingual', 'indian-language'], 
                           help='Type of test to run')
    parser.add_argument('--csv-path', help='Path to custom CSV file (for custom test type)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    test_type = args.test_type or args.test or 'basic'
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("="*60)
    print("🤖 PolyDoc ML Training Framework Test Runner")
    print("="*60)
    
    success = False
    
    try:
        if test_type == 'basic':
            success = await run_basic_test()
        elif test_type == 'classification':
            success = await run_classification_only_test()
        elif test_type == 'qa':
            success = await run_qa_only_test()
        elif test_type == 'robustness':
            success = await run_robustness_test()
        elif test_type == 'custom':
            if not args.csv_path:
                print("❌ Error: --csv-path is required for custom test type")
                sys.exit(1)
            success = await run_custom_csv_test(args.csv_path)
        elif test_type == 'multilingual':
            success = await run_multilingual_test()
        elif test_type == 'indian-language':
            success = await run_indian_language_test()
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logging.exception("Unexpected error during test execution")
        sys.exit(1)
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests completed successfully!")
    else:
        print("💥 Some tests failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Failed to start program: {e}")
        sys.exit(1)
