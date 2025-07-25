"""
Production-level Query Classifier

A modular, robust query classification system that supports both frequency-based
vocabularies and pre-trained embeddings for intent classification.

Author: AI Assistant
Date: 2025-07-25
"""

import logging
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of query classification"""
    intent: str
    confidence: float
    all_scores: Dict[str, float]


class VocabularyLoader:
    """Handles loading different types of vocabulary files"""
    
    @staticmethod
    def load_frequency_vocab(filepath: Union[str, Path]) -> Dict[str, int]:
        """
        Load frequency-based vocabulary file
        Format: word frequency
        """
        vocab = {}
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 2:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
                        continue
                    
                    word, freq_str = parts
                    try:
                        frequency = int(freq_str)
                        vocab[word.lower()] = frequency
                    except ValueError:
                        logger.warning(f"Invalid frequency at line {line_num}: {freq_str}")
                        continue
            
            logger.info(f"Loaded {len(vocab)} words from frequency vocabulary")
            return vocab
            
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise
    
    @staticmethod
    def load_embedding_vocab(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load pre-trained embeddings file
        Format: word val1 val2 ... valN
        """
        embeddings = {}
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 2:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
                        continue
                    
                    word = parts[0].lower()
                    try:
                        vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        embeddings[word] = vector
                    except ValueError:
                        logger.warning(f"Invalid vector at line {line_num}")
                        continue
            
            logger.info(f"Loaded {len(embeddings)} word embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise


class TextPreprocessor:
    """Handles text preprocessing for queries"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
    def preprocess(self, text: str) -> List[str]:
        """Preprocess text into list of tokens"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation and split
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split and filter empty strings
        tokens = [token for token in text.split() if token.strip()]
        
        return tokens


class TFIDFVectorizer:
    """TF-IDF vectorizer for frequency-based vocabularies"""
    
    def __init__(self, vocab_frequencies: Dict[str, int]):
        self.vocab_frequencies = vocab_frequencies
        self.total_documents = sum(vocab_frequencies.values())
        
    def compute_tfidf_vector(self, words: List[str]) -> np.ndarray:
        """Compute TF-IDF vector for a list of words"""
        if not words:
            return np.zeros(len(self.vocab_frequencies))
        
        # Calculate term frequencies
        tf = defaultdict(int)
        for word in words:
            if word in self.vocab_frequencies:
                tf[word] += 1
        
        # Normalize by document length
        doc_length = len(words)
        for word in tf:
            tf[word] = tf[word] / doc_length
        
        # Calculate TF-IDF scores
        tfidf_scores = {}
        for word, tf_score in tf.items():
            # Use frequency as inverse document frequency proxy
            idf = np.log(self.total_documents / (self.vocab_frequencies[word] + 1))
            tfidf_scores[word] = tf_score * idf
        
        return tfidf_scores


class EmbeddingVectorizer:
    """Vectorizer for pre-trained embeddings"""
    
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self.embeddings = embeddings
        self.embedding_dim = len(next(iter(embeddings.values()))) if embeddings else 0
        
    def compute_sentence_vector(self, words: List[str]) -> np.ndarray:
        """Compute sentence vector by averaging word embeddings"""
        vectors = []
        for word in words:
            if word in self.embeddings:
                vectors.append(self.embeddings[word])
        
        if not vectors:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        return np.mean(vectors, axis=0)


class SimilarityCalculator:
    """Handles similarity calculations"""
    
    @staticmethod
    def cosine_similarity(vec1: Union[np.ndarray, Dict], vec2: Union[np.ndarray, Dict]) -> float:
        """Calculate cosine similarity between two vectors"""
        if isinstance(vec1, dict) and isinstance(vec2, dict):
            return SimilarityCalculator._cosine_similarity_sparse(vec1, vec2)
        elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
            return SimilarityCalculator._cosine_similarity_dense(vec1, vec2)
        else:
            raise ValueError("Vectors must be both dict or both numpy arrays")
    
    @staticmethod
    def _cosine_similarity_dense(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Cosine similarity for dense vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    @staticmethod
    def _cosine_similarity_sparse(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Cosine similarity for sparse vectors (dictionaries)"""
        # Calculate dot product
        dot_product = 0.0
        for word in vec1:
            if word in vec2:
                dot_product += vec1[word] * vec2[word]
        
        # Calculate norms
        norm1 = np.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = np.sqrt(sum(val**2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class QueryClassifier:
    """Main query classifier class"""
    
    def __init__(self, 
                 vocab_file: Union[str, Path],
                 intents_config: Optional[Dict[str, List[str]]] = None,
                 vocab_type: str = "frequency",
                 min_confidence_threshold: float = 0.0):
        """
        Initialize the query classifier
        
        Args:
            vocab_file: Path to vocabulary file
            intents_config: Dictionary mapping intent names to keyword lists
            vocab_type: Type of vocabulary file ("frequency" or "embedding")
            min_confidence_threshold: Minimum confidence threshold for classification
        """
        self.vocab_file = Path(vocab_file)
        self.vocab_type = vocab_type.lower()
        self.min_confidence_threshold = min_confidence_threshold
        
        # Default intents configuration
        self.intents_config = intents_config or {
            "order_status": ["order", "track", "status", "tracking", "shipment", "delivery"],
            "pricing": ["price", "cost", "charge", "fee", "expensive", "cheap", "money"],
            "availability": ["available", "stock", "have", "supply", "inventory", "sold"],
            "delivery_issue": ["delivery", "failed", "recipient", "address", "shipping", "delayed"],
            "card_missing": ["card", "note", "message", "missing", "forgot", "include"]
        }
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.similarity_calculator = SimilarityCalculator()
        
        # Load vocabulary and initialize vectorizer
        self._load_vocabulary()
        self._compute_intent_vectors()
        
        logger.info(f"Query classifier initialized with {len(self.intents_config)} intents")
    
    def _load_vocabulary(self):
        """Load vocabulary based on type"""
        if self.vocab_type == "frequency":
            self.vocab = VocabularyLoader.load_frequency_vocab(self.vocab_file)
            self.vectorizer = TFIDFVectorizer(self.vocab)
        elif self.vocab_type == "embedding":
            self.vocab = VocabularyLoader.load_embedding_vocab(self.vocab_file)
            self.vectorizer = EmbeddingVectorizer(self.vocab)
        else:
            raise ValueError(f"Unsupported vocabulary type: {self.vocab_type}")
    
    def _compute_intent_vectors(self):
        """Precompute vectors for all intents"""
        self.intent_vectors = {}
        
        for intent, keywords in self.intents_config.items():
            processed_keywords = self.preprocessor.preprocess(" ".join(keywords))
            
            if self.vocab_type == "frequency":
                intent_vector = self.vectorizer.compute_tfidf_vector(processed_keywords)
            else:  # embedding
                intent_vector = self.vectorizer.compute_sentence_vector(processed_keywords)
            
            self.intent_vectors[intent] = intent_vector
            logger.debug(f"Computed vector for intent '{intent}' with {len(processed_keywords)} keywords")
    
    def classify_query(self, query: str) -> ClassificationResult:
        """
        Classify a query and return the most likely intent
        
        Args:
            query: Input query string
            
        Returns:
            ClassificationResult object with intent, confidence, and all scores
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        # Preprocess query
        query_words = self.preprocessor.preprocess(query)
        
        if not query_words:
            logger.warning("No valid words found in query after preprocessing")
            return ClassificationResult(
                intent="unknown",
                confidence=0.0,
                all_scores={}
            )
        
        # Compute query vector
        if self.vocab_type == "frequency":
            query_vector = self.vectorizer.compute_tfidf_vector(query_words)
        else:  # embedding
            query_vector = self.vectorizer.compute_sentence_vector(query_words)
        
        # Calculate similarities with all intents
        similarities = {}
        for intent, intent_vector in self.intent_vectors.items():
            similarity = self.similarity_calculator.cosine_similarity(query_vector, intent_vector)
            similarities[intent] = similarity
        
        # Find best match
        if not similarities:
            return ClassificationResult(
                intent="unknown",
                confidence=0.0,
                all_scores={}
            )
        
        best_intent = max(similarities, key=similarities.get)
        best_confidence = similarities[best_intent]
        
        # Check confidence threshold
        if best_confidence < self.min_confidence_threshold:
            return ClassificationResult(
                intent="unknown",
                confidence=best_confidence,
                all_scores=similarities
            )
        
        return ClassificationResult(
            intent=best_intent,
            confidence=best_confidence,
            all_scores=similarities
        )
    
    def batch_classify(self, queries: List[str]) -> List[ClassificationResult]:
        """Classify multiple queries in batch"""
        results = []
        for query in queries:
            try:
                result = self.classify_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying query '{query}': {e}")
                results.append(ClassificationResult(
                    intent="error",
                    confidence=0.0,
                    all_scores={}
                ))
        return results
    
    def add_intent(self, intent_name: str, keywords: List[str]):
        """Add a new intent to the classifier"""
        self.intents_config[intent_name] = keywords
        self._compute_intent_vectors()
        logger.info(f"Added new intent: {intent_name}")
    
    def update_intent(self, intent_name: str, keywords: List[str]):
        """Update keywords for an existing intent"""
        if intent_name not in self.intents_config:
            raise ValueError(f"Intent '{intent_name}' does not exist")
        
        self.intents_config[intent_name] = keywords
        self._compute_intent_vectors()
        logger.info(f"Updated intent: {intent_name}")
    
    def get_intents(self) -> List[str]:
        """Get list of all available intents"""
        return list(self.intents_config.keys())
    
    def save_config(self, filepath: Union[str, Path]):
        """Save current configuration to JSON file"""
        config = {
            "intents": self.intents_config,
            "vocab_type": self.vocab_type,
            "min_confidence_threshold": self.min_confidence_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, config_filepath: Union[str, Path], vocab_file: Union[str, Path]):
        """Load classifier from configuration file"""
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        
        return cls(
            vocab_file=vocab_file,
            intents_config=config.get("intents"),
            vocab_type=config.get("vocab_type", "frequency"),
            min_confidence_threshold=config.get("min_confidence_threshold", 0.0)
        )


def main():
    """Example usage of the query classifier"""
    try:
        # Initialize classifier with frequency-based vocabulary
        classifier = QueryClassifier(
            vocab_file="/home/vaibhav/Qery Classifer/vocab.txt",
            vocab_type="frequency",
            min_confidence_threshold=0.1
        )
        
        # Test queries
        test_queries = [
            "Where is my order?",
            "How much does this cost?",
            "Is this item available?",
            "My delivery failed",
            "I didn't receive a card with my order"
        ]
        
        print("Query Classification Results:")
        print("=" * 50)
        
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"Query: '{query}'")
            print(f"Intent: {result.intent}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"All scores: {result.all_scores}")
            print("-" * 30)
        
        # Batch classification
        batch_results = classifier.batch_classify(test_queries)
        print(f"\nBatch processing completed: {len(batch_results)} queries processed")
        
        # Save configuration
        classifier.save_config("classifier_config.json")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
