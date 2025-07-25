"""
Example usage and demo script for the Query Classifier

This script demonstrates various ways to use the production-level query classifier
with different configurations and scenarios.
"""

import sys
import os
from pathlib import Path
import json
import time

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_classifier import QueryClassifier, ClassificationResult
from config import DEFAULT_INTENTS, ECOMMERCE_INTENTS, DOMAIN_CONFIGS


def demo_basic_usage():
    """Demonstrate basic usage of the query classifier"""
    print("=" * 60)
    print("BASIC USAGE DEMO")
    print("=" * 60)
    
    # Initialize classifier
    classifier = QueryClassifier(
        vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
        vocab_type="frequency",
        min_confidence_threshold=0.1
    )
    
    # Test queries
    test_queries = [
        "Where is my order?",
        "How much does this item cost?",
        "Is this product available in stock?",
        "My delivery failed to arrive",
        "I didn't receive a greeting card with my purchase",
        "I want to return this item",
        "Can I speak to customer service?"
    ]
    
    print("Classifying queries...")
    print("-" * 40)
    
    for query in test_queries:
        result = classifier.classify_query(query)
        print(f"Query: '{query}'")
        print(f"Intent: {result.intent}")
        print(f"Confidence: {result.confidence:.3f}")
        
        # Show top 3 scores
        sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 intents:")
        for intent, score in sorted_scores:
            print(f"  {intent}: {score:.3f}")
        print("-" * 40)


def demo_custom_intents():
    """Demonstrate using custom intent configurations"""
    print("\n" + "=" * 60)
    print("CUSTOM INTENTS DEMO")
    print("=" * 60)
    
    # Custom intents for a restaurant
    restaurant_intents = {
        "reservation": ["book", "table", "reservation", "reserve", "seat"],
        "menu_inquiry": ["menu", "food", "dish", "meal", "what", "serve"],
        "hours": ["hours", "open", "close", "time", "when"],
        "location": ["where", "address", "location", "directions", "find"],
        "takeout": ["takeout", "pickup", "order", "delivery", "call"]
    }
    
    classifier = QueryClassifier(
        vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
        intents_config=restaurant_intents,
        vocab_type="frequency"
    )
    
    restaurant_queries = [
        "I want to book a table for tonight",
        "What food do you serve?",
        "What time do you close?",
        "Where are you located?",
        "Can I order takeout?"
    ]
    
    print("Restaurant queries classification:")
    print("-" * 40)
    
    for query in restaurant_queries:
        result = classifier.classify_query(query)
        print(f"Query: '{query}'")
        print(f"Intent: {result.intent} (confidence: {result.confidence:.3f})")
        print()


def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)
    
    classifier = QueryClassifier(
        vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
        intents_config=ECOMMERCE_INTENTS,
        vocab_type="frequency"
    )
    
    # Large batch of queries
    batch_queries = [
        "track my order",
        "what's the price",
        "is it in stock",
        "delivery problem",
        "missing card",
        "return policy",
        "need help",
        "product details",
        "login issues",
        "payment failed"
    ] * 5  # Simulate 50 queries
    
    print(f"Processing {len(batch_queries)} queries in batch...")
    
    start_time = time.time()
    results = classifier.batch_classify(batch_queries)
    end_time = time.time()
    
    processing_time = end_time - start_time
    queries_per_second = len(batch_queries) / processing_time
    
    print(f"Batch processing completed in {processing_time:.3f} seconds")
    print(f"Processing rate: {queries_per_second:.1f} queries/second")
    
    # Analyze results
    intent_counts = {}
    total_confidence = 0
    
    for result in results:
        intent_counts[result.intent] = intent_counts.get(result.intent, 0) + 1
        total_confidence += result.confidence
    
    avg_confidence = total_confidence / len(results)
    
    print(f"Average confidence: {avg_confidence:.3f}")
    print("Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {intent}: {count} ({percentage:.1f}%)")


def demo_configuration_management():
    """Demonstrate configuration save/load functionality"""
    print("\n" + "=" * 60)
    print("CONFIGURATION MANAGEMENT DEMO")
    print("=" * 60)
    
    # Create classifier with custom configuration
    classifier = QueryClassifier(
        vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
        intents_config=ECOMMERCE_INTENTS,
        vocab_type="frequency",
        min_confidence_threshold=0.2
    )
    
    # Add a new intent
    classifier.add_intent("warranty", ["warranty", "guarantee", "coverage", "protection"])
    
    # Save configuration
    config_file = "/home/vaibhav/Qery Classifer/demo_config.json"
    classifier.save_config(config_file)
    print(f"Configuration saved to: {config_file}")
    
    # Load configuration
    new_classifier = QueryClassifier.load_config(
        config_file, 
        "/home/vaibhav/Qery Classifer/vocab.txt"
    )
    
    print("Configuration loaded successfully")
    print(f"Available intents: {new_classifier.get_intents()}")
    
    # Test the loaded classifier
    test_query = "What warranty do you offer?"
    result = new_classifier.classify_query(test_query)
    print(f"\nTest query: '{test_query}'")
    print(f"Classified as: {result.intent} (confidence: {result.confidence:.3f})")


def demo_confidence_thresholding():
    """Demonstrate confidence threshold functionality"""
    print("\n" + "=" * 60)
    print("CONFIDENCE THRESHOLDING DEMO")
    print("=" * 60)
    
    # Test with different confidence thresholds
    thresholds = [0.0, 0.1, 0.3, 0.5]
    test_query = "some random unrelated text"
    
    for threshold in thresholds:
        classifier = QueryClassifier(
            vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
            vocab_type="frequency",
            min_confidence_threshold=threshold
        )
        
        result = classifier.classify_query(test_query)
        
        print(f"Threshold: {threshold}")
        print(f"Query: '{test_query}'")
        print(f"Result: {result.intent} (confidence: {result.confidence:.3f})")
        print(f"Status: {'ACCEPTED' if result.intent != 'unknown' else 'REJECTED'}")
        print("-" * 30)


def demo_domain_specific():
    """Demonstrate domain-specific configurations"""
    print("\n" + "=" * 60)
    print("DOMAIN-SPECIFIC DEMO")
    print("=" * 60)
    
    domains = ["banking", "healthcare"]
    
    for domain in domains:
        print(f"\n{domain.upper()} DOMAIN:")
        print("-" * 30)
        
        classifier = QueryClassifier(
            vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
            intents_config=DOMAIN_CONFIGS[domain],
            vocab_type="frequency"
        )
        
        if domain == "banking":
            test_queries = [
                "What's my account balance?",
                "I want to transfer money",
                "I think there's fraud on my account",
                "My card is blocked"
            ]
        else:  # healthcare
            test_queries = [
                "I need to book an appointment",
                "Can you refill my prescription?",
                "I have severe chest pain",
                "What does my insurance cover?"
            ]
        
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"'{query}' → {result.intent} ({result.confidence:.3f})")


def performance_benchmark():
    """Run performance benchmark"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    classifier = QueryClassifier(
        vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
        vocab_type="frequency"
    )
    
    # Single query performance
    query = "Where is my order?"
    iterations = 1000
    
    start_time = time.time()
    for _ in range(iterations):
        classifier.classify_query(query)
    end_time = time.time()
    
    single_query_time = (end_time - start_time) / iterations
    
    print(f"Single query processing time: {single_query_time*1000:.2f} ms")
    print(f"Theoretical max throughput: {1/single_query_time:.0f} queries/second")
    
    # Memory usage estimation
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Approximate memory usage: {memory_usage:.1f} MB")
    except ImportError:
        print("psutil not available - install with: pip install psutil")


def main():
    """Run all demonstrations"""
    print("QUERY CLASSIFIER - PRODUCTION DEMO")
    print("=====================================")
    
    try:
        demo_basic_usage()
        demo_custom_intents()
        demo_batch_processing()
        demo_configuration_management()
        demo_confidence_thresholding()
        demo_domain_specific()
        performance_benchmark()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
