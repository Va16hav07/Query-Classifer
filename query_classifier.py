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
import time

# Sentence-BERT imports (optional)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of query classification"""
    intent: str
    confidence: float
    all_scores: Dict[str, float]
    processing_time_ms: Optional[float] = None


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


class SentenceBERTVectorizer:
    """Vectorizer using Sentence-BERT for semantic embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence-BERT vectorizer
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence-Transformers not available. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Sentence-BERT model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into embedding vector"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts into embedding vectors"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return np.zeros((len(texts), self.embedding_dim))
    
    def compute_sentence_vector(self, words: List[str]) -> np.ndarray:
        """Compute sentence vector from list of words (for compatibility)"""
        text = " ".join(words)
        return self.encode_text(text)


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
                 vocab_file: Union[str, Path] = None,
                 intents_config: Optional[Dict[str, List[str]]] = None,
                 vocab_type: str = "frequency",
                 min_confidence_threshold: float = 0.0,
                 use_sentence_bert: bool = False,
                 sentence_bert_model: str = "all-MiniLM-L6-v2",
                 sentence_bert_labels: List[str] = None,
                 sentence_bert_descriptions: Dict[str, str] = None):
        """
        Initialize the query classifier
        
        Args:
            vocab_file: Path to vocabulary file (not used with Sentence-BERT)
            intents_config: Dictionary mapping intent names to keyword lists
            vocab_type: Type of vocabulary file ("frequency", "embedding", or "sentence_bert")
            min_confidence_threshold: Minimum confidence threshold for classification
            use_sentence_bert: Whether to use Sentence-BERT instead of TF-IDF
            sentence_bert_model: Model name for Sentence-BERT
            sentence_bert_labels: List of labels for Sentence-BERT classification
            sentence_bert_descriptions: Label descriptions for better semantic matching
        """
        self.vocab_file = Path(vocab_file) if vocab_file else None
        self.vocab_type = vocab_type.lower()
        self.min_confidence_threshold = min_confidence_threshold
        self.use_sentence_bert = use_sentence_bert or vocab_type == "sentence_bert"
        
        # Sentence-BERT configuration
        if self.use_sentence_bert:
            from config import SENTENCE_BERT_CONFIG
            self.sentence_bert_labels = sentence_bert_labels or SENTENCE_BERT_CONFIG["labels"]
            self.sentence_bert_descriptions = sentence_bert_descriptions or SENTENCE_BERT_CONFIG["label_descriptions"]
            self.sentence_bert_model = sentence_bert_model or SENTENCE_BERT_CONFIG["model_name"]
        
        # Default intents configuration (for TF-IDF mode)
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
        
        if self.use_sentence_bert:
            self._init_sentence_bert()
        else:
            # Load vocabulary and initialize vectorizer for TF-IDF mode
            self._load_vocabulary()
            self._compute_intent_vectors()
        
        logger.info(f"Query classifier initialized in {'Sentence-BERT' if self.use_sentence_bert else 'TF-IDF'} mode")
    
    def _load_vocabulary(self):
        """Load vocabulary based on type"""
        if self.use_sentence_bert:
            # Skip vocabulary loading for Sentence-BERT mode
            return
            
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
        start_time = time.time()
        
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if self.use_sentence_bert:
            return self._classify_with_sentence_bert(query, start_time)
        else:
            return self._classify_with_tfidf(query, start_time)
    
    def _classify_with_sentence_bert(self, query: str, start_time: float) -> ClassificationResult:
        """Classify using Sentence-BERT"""
        logger.debug(f"Classifying with Sentence-BERT: '{query}'")
        
        try:
            # Encode the query using Sentence-BERT
            query_embedding = self.vectorizer.encode_text(query)
            
            # Calculate cosine similarity with each label embedding
            similarities = {}
            for label, label_embedding in self.label_embeddings.items():
                similarity = self.similarity_calculator.cosine_similarity(query_embedding, label_embedding)
                similarities[label] = float(similarity)
                logger.debug(f"Similarity with '{label}': {similarity:.4f}")
            
            # Find the best match
            if not similarities:
                logger.warning("No similarities computed")
                return ClassificationResult(
                    intent="unknown",
                    confidence=0.0,
                    all_scores={},
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            best_label = max(similarities, key=similarities.get)
            best_confidence = similarities[best_label]
            
            # Always return the label with highest confidence (never "unknown")
            predicted_label = best_label
            
            if best_confidence < self.min_confidence_threshold:
                logger.info(f"Best confidence {best_confidence:.4f} below threshold {self.min_confidence_threshold}, but returning best match: {predicted_label}")
            else:
                logger.info(f"Best confidence {best_confidence:.4f} above threshold {self.min_confidence_threshold}")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Query classified as '{predicted_label}' with confidence {best_confidence:.4f}")
            
            return ClassificationResult(
                intent=predicted_label,
                confidence=best_confidence,
                all_scores=similarities,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error during Sentence-BERT classification: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                intent="error",
                confidence=0.0,
                all_scores={},
                processing_time_ms=processing_time_ms
            )
    
    def _classify_with_tfidf(self, query: str, start_time: float) -> ClassificationResult:
        """Classify using TF-IDF (original method)"""
        # Preprocess query
        query_words = self.preprocessor.preprocess(query)
        
        if not query_words:
            logger.warning("No valid words found in query after preprocessing")
            return ClassificationResult(
                intent="unknown",
                confidence=0.0,
                all_scores={},
                processing_time_ms=(time.time() - start_time) * 1000
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
                all_scores={},
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        best_intent = max(similarities, key=similarities.get)
        best_confidence = similarities[best_intent]
        
        # Always return the intent with highest confidence (never "unknown")
        predicted_intent = best_intent
        
        if best_confidence < self.min_confidence_threshold:
            logger.info(f"Best confidence {best_confidence:.4f} below threshold {self.min_confidence_threshold}, but returning best match: {predicted_intent}")
        
        return ClassificationResult(
            intent=predicted_intent,
            confidence=best_confidence,
            all_scores=similarities,
            processing_time_ms=(time.time() - start_time) * 1000
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
    
    def _init_sentence_bert(self):
        """Initialize Sentence-BERT components"""
        logger.info("Initializing Sentence-BERT mode")
        
        # Initialize Sentence-BERT vectorizer
        self.vectorizer = SentenceBERTVectorizer(self.sentence_bert_model)
        
        # Precompute label embeddings
        self.label_embeddings = {}
        
        for label in self.sentence_bert_labels:
            if label in self.sentence_bert_descriptions:
                # Use descriptive text for better semantic matching
                text_to_encode = self.sentence_bert_descriptions[label]
                logger.info(f"Encoding label '{label}' with description: '{text_to_encode}'")
            else:
                # Use label name directly
                text_to_encode = label
                logger.info(f"Encoding label '{label}' directly")
            
            embedding = self.vectorizer.encode_text(text_to_encode)
            self.label_embeddings[label] = embedding
            logger.debug(f"✓ Computed embedding for label '{label}'")


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
