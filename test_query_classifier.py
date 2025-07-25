"""
Test suite for the Query Classifier

Comprehensive tests for all components of the production-level query classifier.
"""

import unittest
import tempfile
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to import the classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_classifier import (
    QueryClassifier, VocabularyLoader, TextPreprocessor, 
    TFIDFVectorizer, EmbeddingVectorizer, SimilarityCalculator,
    ClassificationResult
)


class TestVocabularyLoader(unittest.TestCase):
    """Test cases for VocabularyLoader class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_load_frequency_vocab(self):
        """Test loading frequency-based vocabulary"""
        vocab_content = "the 1000\nand 500\nof 300\n"
        vocab_file = Path(self.temp_dir) / "test_vocab.txt"
        
        with open(vocab_file, 'w') as f:
            f.write(vocab_content)
        
        vocab = VocabularyLoader.load_frequency_vocab(vocab_file)
        
        self.assertEqual(len(vocab), 3)
        self.assertEqual(vocab['the'], 1000)
        self.assertEqual(vocab['and'], 500)
        self.assertEqual(vocab['of'], 300)
    
    def test_load_embedding_vocab(self):
        """Test loading embedding vocabulary"""
        embed_content = "the 0.1 0.2 0.3\nand 0.4 0.5 0.6\nof 0.7 0.8 0.9\n"
        embed_file = Path(self.temp_dir) / "test_embed.txt"
        
        with open(embed_file, 'w') as f:
            f.write(embed_content)
        
        embeddings = VocabularyLoader.load_embedding_vocab(embed_file)
        
        self.assertEqual(len(embeddings), 3)
        self.assertTrue(np.allclose(embeddings['the'], [0.1, 0.2, 0.3]))
        self.assertTrue(np.allclose(embeddings['and'], [0.4, 0.5, 0.6]))
    
    def test_file_not_found(self):
        """Test handling of non-existent files"""
        with self.assertRaises(FileNotFoundError):
            VocabularyLoader.load_frequency_vocab("nonexistent.txt")


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for TextPreprocessor class"""
    
    def setUp(self):
        self.preprocessor = TextPreprocessor()
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing"""
        text = "Hello, World! How are you?"
        tokens = self.preprocessor.preprocess(text)
        expected = ["hello", "world", "how", "are", "you"]
        self.assertEqual(tokens, expected)
    
    def test_empty_string(self):
        """Test preprocessing of empty string"""
        tokens = self.preprocessor.preprocess("")
        self.assertEqual(tokens, [])
    
    def test_punctuation_removal(self):
        """Test punctuation removal"""
        text = "test@email.com, phone: 123-456-7890"
        tokens = self.preprocessor.preprocess(text)
        self.assertIn("test", tokens)
        self.assertIn("email", tokens)
        self.assertIn("com", tokens)


class TestTFIDFVectorizer(unittest.TestCase):
    """Test cases for TFIDFVectorizer class"""
    
    def setUp(self):
        self.vocab_freq = {"the": 1000, "and": 500, "order": 100, "price": 50}
        self.vectorizer = TFIDFVectorizer(self.vocab_freq)
    
    def test_compute_tfidf_vector(self):
        """Test TF-IDF vector computation"""
        words = ["the", "order"]
        tfidf_scores = self.vectorizer.compute_tfidf_vector(words)
        
        self.assertIsInstance(tfidf_scores, dict)
        self.assertIn("the", tfidf_scores)
        self.assertIn("order", tfidf_scores)
    
    def test_empty_words(self):
        """Test TF-IDF computation with empty word list"""
        tfidf_scores = self.vectorizer.compute_tfidf_vector([])
        self.assertIsInstance(tfidf_scores, dict)
        self.assertEqual(len(tfidf_scores), 0)


class TestSimilarityCalculator(unittest.TestCase):
    """Test cases for SimilarityCalculator class"""
    
    def test_cosine_similarity_dense(self):
        """Test cosine similarity with dense vectors"""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        
        vec3 = np.array([0, 1, 0])
        similarity2 = SimilarityCalculator.cosine_similarity(vec1, vec3)
        self.assertAlmostEqual(similarity2, 0.0, places=5)
    
    def test_cosine_similarity_sparse(self):
        """Test cosine similarity with sparse vectors"""
        vec1 = {"word1": 1.0, "word2": 0.5}
        vec2 = {"word1": 1.0, "word2": 0.5}
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_zero_vectors(self):
        """Test cosine similarity with zero vectors"""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 0, 0])
        similarity = SimilarityCalculator.cosine_similarity(vec1, vec2)
        self.assertEqual(similarity, 0.0)


class TestQueryClassifier(unittest.TestCase):
    """Test cases for QueryClassifier class"""
    
    def setUp(self):
        # Create a temporary vocabulary file
        self.temp_dir = tempfile.mkdtemp()
        vocab_content = """the 1000
and 500
order 100
track 80
status 70
price 60
cost 50
available 40
stock 35
delivery 30
failed 25
card 20
note 15"""
        
        self.vocab_file = Path(self.temp_dir) / "test_vocab.txt"
        with open(self.vocab_file, 'w') as f:
            f.write(vocab_content)
        
        self.classifier = QueryClassifier(
            vocab_file=self.vocab_file,
            vocab_type="frequency",
            min_confidence_threshold=0.0
        )
    
    def test_classify_order_status_query(self):
        """Test classification of order status query"""
        query = "where is my order"
        result = self.classifier.classify_query(query)
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.intent, "order_status")
        self.assertGreater(result.confidence, 0)
    
    def test_classify_pricing_query(self):
        """Test classification of pricing query"""
        query = "how much does this cost"
        result = self.classifier.classify_query(query)
        
        self.assertEqual(result.intent, "pricing")
        self.assertGreater(result.confidence, 0)
    
    def test_empty_query(self):
        """Test handling of empty query"""
        with self.assertRaises(ValueError):
            self.classifier.classify_query("")
    
    def test_batch_classification(self):
        """Test batch classification"""
        queries = ["track my order", "what's the price"]
        results = self.classifier.batch_classify(queries)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ClassificationResult)
        self.assertIsInstance(results[1], ClassificationResult)
    
    def test_add_new_intent(self):
        """Test adding a new intent"""
        original_intents = len(self.classifier.get_intents())
        self.classifier.add_intent("refund", ["refund", "return", "money", "back"])
        
        new_intents = len(self.classifier.get_intents())
        self.assertEqual(new_intents, original_intents + 1)
        self.assertIn("refund", self.classifier.get_intents())
    
    def test_config_save_and_load(self):
        """Test saving and loading configuration"""
        config_file = Path(self.temp_dir) / "test_config.json"
        
        # Save configuration
        self.classifier.save_config(config_file)
        self.assertTrue(config_file.exists())
        
        # Load configuration
        new_classifier = QueryClassifier.load_config(config_file, self.vocab_file)
        self.assertEqual(
            new_classifier.get_intents(), 
            self.classifier.get_intents()
        )


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a more comprehensive vocabulary file
        vocab_content = """the 5000
and 3000
of 2500
order 1000
track 800
status 700
tracking 650
shipment 600
delivery 550
price 500
cost 450
charge 400
fee 350
expensive 300
cheap 250
available 200
stock 180
have 160
supply 140
inventory 120
sold 100
failed 90
recipient 80
address 70
shipping 60
delayed 50
card 40
note 35
message 30
missing 25
forgot 20
include 15"""
        
        self.vocab_file = Path(self.temp_dir) / "integration_vocab.txt"
        with open(self.vocab_file, 'w') as f:
            f.write(vocab_content)
    
    def test_end_to_end_classification(self):
        """Test complete classification workflow"""
        classifier = QueryClassifier(
            vocab_file=self.vocab_file,
            vocab_type="frequency",
            min_confidence_threshold=0.1
        )
        
        test_cases = [
            ("where is my order", "order_status"),
            ("track my shipment", "order_status"),
            ("how much does this cost", "pricing"),
            ("is this available", "availability"),
            ("delivery failed", "delivery_issue"),
            ("missing card", "card_missing")
        ]
        
        for query, expected_intent in test_cases:
            with self.subTest(query=query):
                result = classifier.classify_query(query)
                self.assertEqual(
                    result.intent, expected_intent,
                    f"Query '{query}' classified as '{result.intent}' instead of '{expected_intent}'"
                )
                self.assertGreater(result.confidence, 0)
    
    def test_confidence_threshold(self):
        """Test confidence threshold functionality"""
        classifier = QueryClassifier(
            vocab_file=self.vocab_file,
            vocab_type="frequency",
            min_confidence_threshold=0.8  # High threshold
        )
        
        # Test with a query that might have low confidence
        result = classifier.classify_query("random unrelated words")
        
        # Should return unknown if confidence is below threshold
        if result.confidence < 0.8:
            self.assertEqual(result.intent, "unknown")


def create_test_data():
    """Create additional test data files"""
    test_dir = Path("/home/vaibhav/Qery Classifer/tests")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample embedding file for testing
    embedding_content = """the 0.1 0.2 0.3 0.4 0.5
and 0.2 0.3 0.4 0.5 0.6
order 0.8 0.1 0.2 0.3 0.4
track 0.7 0.2 0.1 0.4 0.3
price 0.1 0.8 0.2 0.3 0.4
cost 0.2 0.7 0.3 0.1 0.4"""
    
    embedding_file = test_dir / "sample_embeddings.txt"
    with open(embedding_file, 'w') as f:
        f.write(embedding_content)
    
    print(f"Test data created in {test_dir}")


if __name__ == "__main__":
    # Create test data
    create_test_data()
    
    # Run tests
    unittest.main(verbosity=2)
