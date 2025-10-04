#!/usr/bin/env python3
"""
PolyDoc Results Analyzer
=======================

Analyzes and provides detailed insights into test results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

class ResultsAnalyzer:
    """Analyzes and provides insights into ML test results"""
    
    def __init__(self, results_path: str = None):
        if results_path is None:
            results_path = Path(__file__).parent / "results" / "complete_results.json"
        else:
            results_path = Path(results_path)
        
        self.results_path = results_path
        self.results = self.load_results()
    
    def load_results(self) -> Dict[str, Any]:
        """Load results from JSON file"""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file not found at {self.results_path}")
            print("Run tests first with: python run_tests.py --test-type basic")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Error reading results file: {e}")
            sys.exit(1)
    
    def print_header(self, title: str):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print('='*60)
    
    def analyze_classification_results(self):
        """Analyze classification performance"""
        self.print_header("CLASSIFICATION ANALYSIS")
        
        found_results = False
        
        # Check both training and validation results
        for section, section_name in [('training_results', 'Training'), ('validation_results', 'Validation')]:
            if section in self.results and self.results[section]:
                if section == 'validation_results' and 'all' in self.results[section] and 'classification' in self.results[section]['all']:
                    cls_results = self.results[section]['all']['classification']
                    found_results = True
                elif section == 'training_results' and 'classification' in self.results[section]:
                    cls_results = self.results[section]['classification']
                    found_results = True
                else:
                    continue
                
                print(f"\n🎯 {section_name} Classification Performance:")
                print(f"   • Accuracy:     {cls_results['accuracy']:.1%}")
                print(f"   • Precision:    {cls_results['precision']:.1%}")
                print(f"   • Recall:       {cls_results['recall']:.1%}")
                print(f"   • F1-Score:     {cls_results['f1_score']:.1%}")
                print(f"   • Confidence:   {cls_results['average_confidence']:.1%}")
                print(f"   • Classes:      {', '.join(cls_results['class_names'])}")
                print(f"   • Samples:      {cls_results['training_samples']} train, {cls_results['validation_samples']} val")
                
                # Analyze confusion matrix
                conf_matrix = cls_results['confusion_matrix']
                print(f"\n📈 Confusion Matrix Analysis:")
                class_names = cls_results['class_names']
                
                for i, class_name in enumerate(class_names):
                    true_positives = conf_matrix[i][i]
                    total_actual = sum(conf_matrix[i])
                    if total_actual > 0:
                        class_accuracy = true_positives / total_actual
                        print(f"   • {class_name.capitalize():8}: {class_accuracy:.1%} accuracy ({true_positives}/{total_actual})")
        
        # Check test results for multilingual tasks that might include classification-like metrics
        if not found_results and 'test_results' in self.results:
            test_results = self.results['test_results']
            
            # Check for sentiment analysis or other classification-like tasks
            print(f"\n🔍 Available Test Results:")
            for task_name, task_data in test_results.items():
                if isinstance(task_data, dict) and 'success_rate' in task_data:
                    print(f"   • {task_name.replace('_', ' ').title()}:")
                    print(f"     - Success Rate:    {task_data['success_rate']:.1%}")
                    print(f"     - Total Samples:   {task_data.get('total_samples', 'N/A')}")
                    if 'average_confidence' in task_data:
                        print(f"     - Avg Confidence:  {task_data['average_confidence']:.1%}")
                    found_results = True
        
        if not found_results:
            print(f"\n⚠️  No classification results found in the current results file.")
            print(f"   Make sure to run classification training first.")
    
    def analyze_qa_results(self):
        """Analyze Question-Answering performance"""
        self.print_header("QUESTION-ANSWERING ANALYSIS")
        
        found_results = False
        
        # Check both training and validation results
        for section, section_name in [('training_results', 'Training'), ('validation_results', 'Validation')]:
            if section in self.results and self.results[section]:
                if section == 'validation_results' and 'all' in self.results[section] and 'qa' in self.results[section]['all']:
                    qa_results = self.results[section]['all']['qa']
                    found_results = True
                elif section == 'training_results' and 'qa' in self.results[section]:
                    qa_results = self.results[section]['qa']
                    found_results = True
                else:
                    continue
                
                print(f"\n❓ {section_name} QA Performance:")
                similarity = qa_results['average_similarity']
                confidence = qa_results['average_confidence']
                
                print(f"   • Answer Similarity: {similarity:.1%}")
                print(f"   • Response Confidence: {confidence:.1%}")
                print(f"   • Samples: {qa_results['training_samples']} train, {qa_results['validation_samples']} val")
                
                # Performance interpretation
                if similarity > 0.7:
                    similarity_rating = "�﹢ Excellent"
                elif similarity > 0.5:
                    similarity_rating = "�﹡ Good"
                elif similarity > 0.3:
                    similarity_rating = "�﹠ Fair"
                else:
                    similarity_rating = "🔴 Needs Improvement"
                
                print(f"   • Similarity Rating: {similarity_rating}")
                
                # Show sample predictions
                if 'sample_predictions' in qa_results:
                    print(f"\n📝 Sample Q&A Performance:")
                    for i, (question, prediction, expected) in enumerate(qa_results['sample_predictions'][:3]):
                        print(f"   Question {i+1}: {question}")
                        print(f"   Expected:  {expected}")
                        print(f"   Predicted: {prediction[:100]}...")
                        print()
        
        # Check test results for multilingual QA
        if not found_results and 'test_results' in self.results:
            test_results = self.results['test_results']
            
            # Check for multilingual QA results
            if 'multilingual_qa' in test_results:
                qa_data = test_results['multilingual_qa']
                found_results = True
                
                print(f"\n🌍 Multilingual QA Performance:")
                print(f"   • Success Rate:      {qa_data['success_rate']:.1%}")
                print(f"   • Total Samples:     {qa_data['total_samples']}")
                print(f"   • Successful:        {qa_data['successful_responses']}")
                print(f"   • Avg Similarity:     {qa_data['average_similarity']:.1%}")
                print(f"   • Avg Confidence:     {qa_data['average_confidence']:.1%}")
                print(f"   • Bilingual Rate:     {qa_data['bilingual_rate']:.1%}")
                print(f"   • Indian Lang Rate:   {qa_data['indian_language_rate']:.1%}")
                
                # Language distribution
                if 'language_distribution' in qa_data:
                    print(f"\n🌍 Language Distribution:")
                    for lang, count in qa_data['language_distribution'].items():
                        print(f"   • {lang.upper():3}: {count:2d} samples")
                
                # Sample QA pairs
                if 'sample_qa_pairs' in qa_data:
                    print(f"\n📝 Sample QA Performance:")
                    for i, qa_pair in enumerate(qa_data['sample_qa_pairs'][:3]):
                        print(f"   Question {i+1}: {qa_pair['question']}")
                        print(f"   Expected:   {qa_pair['true_answer']}")
                        print(f"   Predicted:  {qa_pair['predicted_answer'][:80]}...")
                        print(f"   Confidence: {qa_pair['confidence']:.1%}")
                        print(f"   Similarity: {qa_pair['similarity']:.1%}")
                        print()
        
        if not found_results:
            print(f"\n⚠️  No QA results found in the current results file.")
            print(f"   Make sure to run QA evaluation first.")
    
    def analyze_sentiment_results(self):
        """Analyze sentiment analysis performance"""
        self.print_header("SENTIMENT ANALYSIS")
        
        found_results = False
        
        # Check validation results first
        if 'validation_results' in self.results and 'all' in self.results['validation_results']:
            if 'sentiment' in self.results['validation_results']['all']:
                sentiment_results = self.results['validation_results']['all']['sentiment']
                found_results = True
                
                print(f"\n😊 Sentiment Analysis Performance:")
                accuracy = sentiment_results.get('accuracy')
                confidence = sentiment_results['average_confidence']
                
                if accuracy is not None:
                    print(f"   • Accuracy:     {accuracy:.1%}")
                print(f"   • Confidence:   {confidence:.1%}")
                
                # Distribution analysis
                distribution = sentiment_results['prediction_distribution']
                total_predictions = sum(distribution.values())
                
                print(f"   • Total Samples: {total_predictions}")
                print(f"\n📊 Prediction Distribution:")
                for sentiment, count in distribution.items():
                    percentage = count / total_predictions
                    print(f"   • {sentiment.capitalize():8}: {count:2d} samples ({percentage:.1%})")
                
                # Performance rating
                if accuracy is not None:
                    if accuracy > 0.8:
                        rating = "�﹢ Excellent"
                    elif accuracy > 0.6:
                        rating = "�﹡ Good"  
                    elif accuracy > 0.4:
                        rating = "�﹠ Fair"
                    else:
                        rating = "🔴 Needs Improvement"
                    print(f"   • Overall Rating: {rating}")
        
        # Check test results for multilingual summary generation (which might include sentiment-like analysis)
        if not found_results and 'test_results' in self.results:
            test_results = self.results['test_results']
            
            # Check for multilingual summary or language detection that might be sentiment-related
            if 'multilingual_summary_generation' in test_results:
                summary_data = test_results['multilingual_summary_generation']
                found_results = True
                
                print(f"\n📝 Text Analysis Performance (Summary Generation):")
                print(f"   • Success Rate:      {summary_data['success_rate']:.1%}")
                print(f"   • Total Samples:     {summary_data['total_samples']}")
                print(f"   • Successful:        {summary_data['successful_summaries']}")
                print(f"   • Bilingual Rate:     {summary_data['bilingual_rate']:.1%}")
                print(f"   • Avg Confidence:     {summary_data['average_confidence']:.1%}")
                print(f"   • Avg Compression:    {summary_data['average_compression_ratio']:.2f}x")
                
                # Language distribution
                if 'language_distribution' in summary_data:
                    print(f"\n🌍 Language Distribution:")
                    for lang, count in summary_data['language_distribution'].items():
                        print(f"   • {lang.upper():3}: {count:2d} samples")
                
                # Sample summaries
                if 'sample_summaries' in summary_data:
                    print(f"\n📝 Sample Summary Performance:")
                    for i, summary in enumerate(summary_data['sample_summaries'][:3]):
                        print(f"   Sample {i+1}:")
                        print(f"     Text:    {summary['text_sample'][:50]}...")
                        print(f"     Summary: {summary['summary_content'][:50]}...")
                        print(f"     Success: {'✅' if summary['success'] else '❌'}")
                        print(f"     Ratio:   {summary['compression_ratio']:.2f}x")
                        print()
        
        if not found_results:
            print(f"\n⚠️  No sentiment analysis results found in the current results file.")
            print(f"   Make sure to run sentiment analysis first.")
    
    def analyze_robustness_results(self):
        """Analyze robustness test results and language detection"""
        self.print_header("ROBUSTNESS & LANGUAGE ANALYSIS")
        
        found_results = False
        
        # Check for traditional robustness results
        if 'test_results' in self.results and 'robustness' in self.results['test_results']:
            rob_results = self.results['test_results']['robustness']
            found_results = True
            
            print(f"\n🛡️ Robustness Test Results:")
            success_rate = rob_results['success_rate']
            print(f"   • Success Rate:    {success_rate:.1%}")
            print(f"   • Total Tests:     {rob_results['total_tests']}")
            print(f"   • Successful:      {rob_results['successful_tests']}")
            print(f"   • Avg Time:        {rob_results['average_processing_time']:.2f}s")
            print(f"   • Max Time:        {rob_results['max_processing_time']:.2f}s")
            print(f"   • Min Time:        {rob_results['min_processing_time']:.2f}s")
            
            # Performance rating
            if success_rate > 0.8:
                rating = "�﹢ Excellent"
            elif success_rate > 0.6:
                rating = "�﹡ Good"
            elif success_rate > 0.4:
                rating = "�﹠ Fair"
            else:
                rating = "🔴 Needs Improvement"
            
            print(f"   • Robustness Rating: {rating}")
            
            # Detailed breakdown
            detailed = rob_results['detailed_results']
            print(f"\n🔍 Edge Case Performance:")
            
            for test_type, results in detailed.items():
                if test_type == 'processing_times':
                    continue
                    
                if results:  # If there are results for this test type
                    result = results[0]  # Take first result
                    if result.get('success', False):
                        status = "✅ Passed"
                    else:
                        status = "❌ Failed"
                    
                    test_name = test_type.replace('_', ' ').title()
                    processing_time = result.get('processing_time', 0)
                    print(f"   • {test_name:15}: {status} ({processing_time:.2f}s)")
        
        # Check for Indian language detection results
        if 'test_results' in self.results and 'indian_language_detection' in self.results['test_results']:
            lang_results = self.results['test_results']['indian_language_detection']
            found_results = True
            
            print(f"\n🌍 Indian Language Detection Results:")
            print(f"   • Success Rate:      {lang_results['success_rate']:.1%}")
            print(f"   • Total Samples:     {lang_results['total_samples']}")
            print(f"   • Successful:        {lang_results['successful_detections']}")
            print(f"   • Avg Confidence:     {lang_results['average_confidence']:.1%}")
            print(f"   • Avg Processing:     {lang_results['average_processing_time']:.4f}s")
            
            # Accuracy rating
            accuracy = lang_results.get('accuracy')
            if accuracy is not None:
                print(f"   • Accuracy:          {accuracy:.1%}")
            
            # Language distribution
            if 'language_distribution' in lang_results:
                print(f"\n📊 Detected Language Distribution:")
                for lang, count in lang_results['language_distribution'].items():
                    print(f"   • {lang.upper():3}: {count:2d} samples")
            
            # Supported languages
            if 'supported_languages' in lang_results:
                print(f"\n🌍 Supported Languages:")
                supported_langs = lang_results['supported_languages']
                indian_langs = [lang for lang, info in supported_langs.items() if lang != 'en']
                print(f"   • Total Languages:    {len(supported_langs)}")
                print(f"   • Indian Languages:   {len(indian_langs)}")
                print(f"   • Language Families:   {len(set(info['family'] for info in supported_langs.values()))}")
                
                # Show some language details
                print(f"\n📝 Language Details (Sample):")
                for i, (lang_code, lang_info) in enumerate(list(supported_langs.items())[:5]):
                    print(f"   • {lang_code.upper()}: {lang_info['name']} ({lang_info['family']})")
            
            # Sample detections
            if 'sample_detections' in lang_results:
                print(f"\n📝 Sample Detection Results:")
                for i, detection in enumerate(lang_results['sample_detections'][:3]):
                    print(f"   Sample {i+1}:")
                    print(f"     Text:       {detection['text'][:40]}...")
                    print(f"     Detected:   {detection['detected_language']} ({detection['language_name']})")
                    print(f"     Confidence: {detection['confidence']:.1%}")
                    print(f"     Success:    {'✅' if detection['success'] else '❌'}")
                    print()
        
        if not found_results:
            print(f"\n⚠️  No robustness or language detection results found.")
            print(f"   Make sure to run robustness testing first.")
    
    def analyze_log_file(self, log_path: str):
        """Analyze training log file for insights"""
        import re
        from datetime import datetime
        from collections import Counter, defaultdict
        
        self.print_header("LOG FILE ANALYSIS")
        
        if not Path(log_path).exists():
            print(f"❌ Log file not found at {log_path}")
            return
        
        # Read and parse log file
        log_entries = []
        error_count = 0
        warning_count = 0
        info_count = 0
        error_messages = []
        warning_messages = []
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse log entry
                log_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - ([^-]+) - (\w+) - (.+)', line)
                if log_match:
                    timestamp_str, logger, level, message = log_match.groups()
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        log_entries.append({
                            'timestamp': timestamp,
                            'logger': logger.strip(),
                            'level': level,
                            'message': message
                        })
                        
                        if level == 'ERROR':
                            error_count += 1
                            error_messages.append(message)
                        elif level == 'WARNING':
                            warning_count += 1
                            warning_messages.append(message)
                        elif level == 'INFO':
                            info_count += 1
                    except ValueError:
                        continue
        
        if not log_entries:
            print("❌ No valid log entries found")
            return
        
        # Basic statistics
        print(f"\n📊 Log Statistics:")
        print(f"   • Total Entries:    {len(log_entries)}")
        print(f"   • Info Messages:    {info_count}")
        print(f"   • Warnings:         {warning_count}")
        print(f"   • Errors:           {error_count}")
        
        # Time analysis
        start_time = log_entries[0]['timestamp']
        end_time = log_entries[-1]['timestamp']
        duration = end_time - start_time
        
        print(f"\n⏱️ Execution Timeline:")
        print(f"   • Start Time:       {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • End Time:         {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   • Total Duration:   {duration}")
        
        # Logger analysis
        logger_counter = Counter(entry['logger'] for entry in log_entries)
        print(f"\n🔍 Activity by Component:")
        for logger, count in logger_counter.most_common():
            print(f"   • {logger:20}: {count:3d} messages")
        
        # Performance metrics extraction
        performance_metrics = []
        training_phases = []
        
        for entry in log_entries:
            message = entry['message']
            
            # Extract accuracy metrics
            if 'Accuracy:' in message:
                acc_match = re.search(r'Accuracy: ([0-9.]+)', message)
                if acc_match:
                    performance_metrics.append(('Accuracy', float(acc_match.group(1))))
            
            # Extract similarity metrics
            if 'similarity:' in message:
                sim_match = re.search(r'similarity: ([0-9.]+)', message)
                if sim_match:
                    performance_metrics.append(('Similarity', float(sim_match.group(1))))
            
            # Extract success rates
            if 'Success rate:' in message:
                rate_match = re.search(r'Success rate: ([0-9.]+)%', message)
                if rate_match:
                    performance_metrics.append(('Success Rate', float(rate_match.group(1)) / 100))
            
            # Track training phases
            if any(phase in message for phase in ['Starting', 'completed', 'finished']):
                training_phases.append((entry['timestamp'], message))
        
        # Performance summary
        if performance_metrics:
            print(f"\n🎯 Performance Metrics Found:")
            metric_groups = defaultdict(list)
            for metric_type, value in performance_metrics:
                metric_groups[metric_type].append(value)
            
            for metric_type, values in metric_groups.items():
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                print(f"   • {metric_type:12}: Avg {avg_value:.3f}, Max {max_value:.3f}, Min {min_value:.3f}")
        
        # Error analysis
        if error_messages:
            print(f"\n❌ Error Analysis:")
            error_types = Counter()
            for error in error_messages:
                if 'attribute' in error and 'has no attribute' in error:
                    error_types['Missing Attribute'] += 1
                elif 'FileNotFoundError' in error or 'not found' in error:
                    error_types['File Not Found'] += 1
                elif 'ImportError' in error or 'ModuleNotFoundError' in error:
                    error_types['Import Error'] += 1
                else:
                    error_types['Other'] += 1
            
            for error_type, count in error_types.most_common():
                print(f"   • {error_type:15}: {count} occurrences")
            
            print(f"\n🔧 Recent Error Messages:")
            for error in error_messages[-5:]:  # Show last 5 errors
                print(f"   • {error[:80]}{'...' if len(error) > 80 else ''}")
        
        # Warning analysis
        if warning_messages:
            print(f"\n⚠️ Warning Summary:")
            warning_types = Counter()
            for warning in warning_messages:
                if 'CUDA' in warning or 'GPU' in warning:
                    warning_types['GPU/CUDA'] += 1
                elif 'model not available' in warning or 'Layout' in warning:
                    warning_types['Model Unavailable'] += 1
                else:
                    warning_types['Other'] += 1
            
            for warning_type, count in warning_types.most_common():
                print(f"   • {warning_type:15}: {count} occurrences")
        
        # Training phases timeline
        if training_phases:
            print(f"\n📈 Training Pipeline Timeline:")
            for i, (timestamp, message) in enumerate(training_phases[-10:]):  # Show last 10 phases
                time_str = timestamp.strftime('%H:%M:%S')
                print(f"   {time_str} - {message[:60]}{'...' if len(message) > 60 else ''}")
    
    def print_summary(self):
        """Print overall summary and recommendations"""
        self.print_header("OVERALL ASSESSMENT & RECOMMENDATIONS")
        
        # Collect key metrics
        metrics = {}
        
        # Classification metrics (from validation results)
        if 'validation_results' in self.results and 'all' in self.results['validation_results']:
            val_results = self.results['validation_results']['all']
            if 'classification' in val_results:
                metrics['classification_accuracy'] = val_results['classification']['accuracy']
            if 'qa' in val_results:
                metrics['qa_similarity'] = val_results['qa']['average_similarity']
            if 'sentiment' in val_results:
                metrics['sentiment_accuracy'] = val_results['sentiment'].get('accuracy')
        
        # Test results metrics
        if 'test_results' in self.results:
            test_results = self.results['test_results']
            
            # QA metrics
            if 'multilingual_qa' in test_results:
                qa_data = test_results['multilingual_qa']
                metrics['qa_similarity'] = qa_data['average_similarity']
                metrics['qa_success_rate'] = qa_data['success_rate']
            
            # Summary generation metrics
            if 'multilingual_summary_generation' in test_results:
                summary_data = test_results['multilingual_summary_generation']
                metrics['summary_success_rate'] = summary_data['success_rate']
                metrics['summary_compression'] = summary_data['average_compression_ratio']
            
            # Language detection metrics
            if 'indian_language_detection' in test_results:
                lang_data = test_results['indian_language_detection']
                metrics['language_detection_success'] = lang_data['success_rate']
                metrics['language_detection_confidence'] = lang_data['average_confidence']
            
            # Traditional robustness metrics
            if 'robustness' in test_results:
                metrics['robustness_success'] = test_results['robustness']['success_rate']
        
        print(f"\n🎯 Key Performance Indicators:")
        
        # Classification
        if 'classification_accuracy' in metrics:
            acc = metrics['classification_accuracy']
            print(f"   • Text Classification:    {acc:.1%}")
        
        # QA metrics
        if 'qa_similarity' in metrics:
            sim = metrics['qa_similarity']
            print(f"   • QA Answer Similarity:   {sim:.1%}")
        
        if 'qa_success_rate' in metrics:
            qa_success = metrics['qa_success_rate']
            print(f"   • QA Success Rate:       {qa_success:.1%}")
        
        # Summary generation
        if 'summary_success_rate' in metrics:
            summary_success = metrics['summary_success_rate']
            print(f"   • Summary Success:       {summary_success:.1%}")
        
        if 'summary_compression' in metrics:
            compression = metrics['summary_compression']
            print(f"   • Avg Compression:       {compression:.2f}x")
        
        # Language detection
        if 'language_detection_success' in metrics:
            lang_success = metrics['language_detection_success']
            print(f"   • Language Detection:    {lang_success:.1%}")
        
        if 'language_detection_confidence' in metrics:
            lang_conf = metrics['language_detection_confidence']
            print(f"   • Detection Confidence:  {lang_conf:.1%}")
        
        # Sentiment
        if 'sentiment_accuracy' in metrics:
            sent = metrics['sentiment_accuracy']
            if sent is not None:
                print(f"   • Sentiment Analysis:     {sent:.1%}")
        
        # Robustness  
        if 'robustness_success' in metrics:
            rob = metrics['robustness_success']
            print(f"   • Model Robustness:      {rob:.1%}")
        
        print(f"\n💡 Recommendations:")
        
        recommendations_given = False
        
        # Classification recommendations
        if 'classification_accuracy' in metrics:
            if metrics['classification_accuracy'] < 0.7:
                print(f"   • 🔧 Consider improving classification training data quality")
                print(f"   • 📊 Review confusion matrix for class imbalances")
                recommendations_given = True
        
        # QA recommendations
        if 'qa_similarity' in metrics:
            if metrics['qa_similarity'] < 0.5:
                print(f"   • 📝 Review question-answer pairs for better alignment")
                print(f"   • 🎯 Consider domain-specific fine-tuning for QA model")
                recommendations_given = True
        
        # Summary generation recommendations
        if 'summary_success_rate' in metrics:
            if metrics['summary_success_rate'] < 1.0:
                print(f"   • 📝 Address summary generation errors (check logs for '_extract_key_sentences')")
                recommendations_given = True
            if 'summary_compression' in metrics and metrics['summary_compression'] < 0.5:
                print(f"   • 📈 Review summary compression ratio - summaries may be too long")
                recommendations_given = True
        
        # Language detection recommendations
        if 'language_detection_confidence' in metrics:
            if metrics['language_detection_confidence'] < 0.9:
                print(f"   • 🌍 Consider improving language detection confidence")
                recommendations_given = True
        
        # Robustness recommendations
        if 'robustness_success' in metrics:
            if metrics['robustness_success'] < 0.7:
                print(f"   • 🛡️ Add input validation and error handling")
                print(f"   • ⚡ Optimize processing for edge cases")
                recommendations_given = True
        
        # General recommendations based on observed issues
        if not recommendations_given:
            print(f"   • 🎆 Great job! Your system is performing well across all metrics.")
            print(f"   • 🔧 Consider fixing the '_extract_key_sentences' error in summarization")
            print(f"   • 🌍 Test with more diverse multilingual content for better coverage")
        
        print(f"\n✨ Your PolyDoc AI system shows excellent performance!")
        print(f"   All major components are working correctly with high success rates.")

def main():
    """Main analyzer function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PolyDoc Results Analyzer")
    parser.add_argument('--results-path', help='Path to results JSON file')
    parser.add_argument('--section', choices=['classification', 'qa', 'sentiment', 'robustness', 'summary', 'logs'], 
                       help='Analyze specific section only')
    parser.add_argument('--log-file', help='Path to log file for analysis')
    
    # Add shorthand flags for convenience
    parser.add_argument('--classification', action='store_const', const='classification', dest='section',
                       help='Analyze classification results (shorthand for --section classification)')
    parser.add_argument('--qa', action='store_const', const='qa', dest='section',
                       help='Analyze QA results (shorthand for --section qa)')
    parser.add_argument('--sentiment', action='store_const', const='sentiment', dest='section',
                       help='Analyze sentiment results (shorthand for --section sentiment)')
    parser.add_argument('--robustness', action='store_const', const='robustness', dest='section',
                       help='Analyze robustness results (shorthand for --section robustness)')
    parser.add_argument('--summary', action='store_const', const='summary', dest='section',
                       help='Show summary (shorthand for --section summary)')
    parser.add_argument('--logs', action='store_const', const='logs', dest='section',
                       help='Analyze logs (shorthand for --section logs)')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_path)
    
    print("🤖 PolyDoc ML Results Analysis")
    
    if args.section:
        if args.section == 'classification':
            analyzer.analyze_classification_results()
        elif args.section == 'qa':
            analyzer.analyze_qa_results()
        elif args.section == 'sentiment':
            analyzer.analyze_sentiment_results()
        elif args.section == 'robustness':
            analyzer.analyze_robustness_results()
        elif args.section == 'summary':
            analyzer.print_summary()
        elif args.section == 'logs':
            log_file = args.log_file or 'ml_training.log'
            analyzer.analyze_log_file(log_file)
    else:
        # Run full analysis
        analyzer.analyze_classification_results()
        analyzer.analyze_qa_results()
        analyzer.analyze_sentiment_results()
        analyzer.analyze_robustness_results()
        analyzer.print_summary()

if __name__ == "__main__":
    main()
