"""
BERT-based Query Classifier

Enhanced version of the query classifier using BERT-style embeddings
for superior semantic understanding and intent classification.

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

# BERT and transformer imports
try:
    from sentence_transformers import SentenceTransformer
    import torch
    BERT_AVAILABLE = True
    print("✓ BERT dependencies available")
except ImportError as e:
    BERT_AVAILABLE = False
    print(f"⚠ BERT dependencies not available: {e}")
    print("Install with: pip install sentence-transformers torch")

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


class BERTVectorizer:
    """BERT-based sentence vectorizer using sentence-transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize BERT vectorizer
        
        Args:
            model_name: Name of the sentence-transformer model to use
                       'all-MiniLM-L6-v2' - Fast, good performance (default)
                       'all-mpnet-base-v2' - Best performance, slower
                       'paraphrase-MiniLM-L6-v2' - Good for paraphrase detection
        """
        if not BERT_AVAILABLE:
            raise ImportError("BERT dependencies not available. Install with: pip install sentence-transformers torch")
        
        self.model_name = model_name
        logger.info(f"Loading BERT model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ BERT model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences into BERT embeddings
        
        Args:
            sentences: List of sentences to encode
            
        Returns:
            numpy array of shape (num_sentences, embedding_dim)
        """
        if not sentences:
            return np.empty((0, self.embedding_dim))
        
        try:
            # Convert to embeddings
            embeddings = self.model.encode(sentences, convert_to_tensor=False)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding sentences: {e}")
            return np.zeros((len(sentences), self.embedding_dim))
    
    def encode_single(self, sentence: str) -> np.ndarray:
        """Encode a single sentence"""
        return self.encode_sentences([sentence])[0]


class TextPreprocessor:
    """Handles text preprocessing for queries"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        """
        Initialize preprocessor
        
        Note: For BERT, we typically keep punctuation as it helps with understanding
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
    def preprocess(self, text: str) -> str:
        """Preprocess text - for BERT we do minimal preprocessing"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Basic cleaning
        text = text.strip()
        
        # Lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation only if explicitly requested (not recommended for BERT)
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class SimilarityCalculator:
    """Handles similarity calculations for BERT embeddings"""
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1.shape) == 1:
            vec1 = vec1.reshape(1, -1)
        if len(vec2.shape) == 1:
            vec2 = vec2.reshape(1, -1)
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm.T)
        
        return float(similarity[0, 0])
    
    @staticmethod
    def euclidean_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean similarity (1 / (1 + distance))"""
        distance = np.linalg.norm(vec1 - vec2)
        return 1.0 / (1.0 + distance)


class BERTQueryClassifier:
    """Main BERT-based query classifier class"""
    
    def __init__(self, 
                 intents_config: Optional[Dict[str, List[str]]] = None,
                 model_name: str = 'all-MiniLM-L6-v2',
                 min_confidence_threshold: float = 0.0,
                 similarity_metric: str = 'cosine'):
        """
        Initialize the BERT query classifier
        
        Args:
            intents_config: Dictionary mapping intent names to example phrases
            model_name: Name of the sentence-transformer model to use
            min_confidence_threshold: Minimum confidence threshold for classification
            similarity_metric: Similarity metric to use ('cosine' or 'euclidean')
        """
        if not BERT_AVAILABLE:
            raise ImportError("BERT dependencies not available. Install with: pip install sentence-transformers torch")
        
        # Default intents configuration
        self.intents_config = intents_config or {
            "order_status": ["order", "track", "status", "tracking", "shipment", "delivery", "where is my order"],
            "pricing": ["price", "cost", "charge", "fee", "expensive", "cheap", "money", "how much does this cost"],
            "availability": ["available", "stock", "have", "supply", "inventory", "sold", "is this available"],
            "delivery_issue": ["delivery", "failed", "recipient", "address", "shipping", "delayed", "delivery problem"],
            "card_missing": ["card", "note", "message", "missing", "forgot", "include", "greeting card missing"]
        }
        
        self.model_name = model_name
        self.min_confidence_threshold = min_confidence_threshold
        self.similarity_metric = similarity_metric
        
        # Initialize components
        self.preprocessor = TextPreprocessor(lowercase=True, remove_punctuation=False)
        self.vectorizer = BERTVectorizer(model_name)
        self.similarity_calculator = SimilarityCalculator()
        
        # Compute intent embeddings
        self._compute_intent_embeddings()
        
        logger.info(f"BERT Query classifier initialized with {len(self.intents_config)} intents")
    
    def _compute_intent_embeddings(self):
        """Precompute BERT embeddings for all intents"""
        self.intent_embeddings = {}
        
        for intent, examples in self.intents_config.items():
            # Create training sentences from examples
            training_sentences = []
            
            # Add individual keywords
            training_sentences.extend(examples)
            
            # Add some synthetic sentences for better understanding
            for example in examples[:3]:  # Use first 3 examples
                training_sentences.extend([
                    f"I need help with {example}",
                    f"Question about {example}",
                    f"Issue with {example}"
                ])
            
            # Preprocess sentences
            processed_sentences = [
                self.preprocessor.preprocess(sentence) 
                for sentence in training_sentences
            ]
            
            # Get BERT embeddings
            embeddings = self.vectorizer.encode_sentences(processed_sentences)
            
            # Use mean embedding as intent representation
            intent_embedding = np.mean(embeddings, axis=0)
            self.intent_embeddings[intent] = intent_embedding
            
            logger.debug(f"Computed embedding for intent '{intent}' from {len(training_sentences)} examples")
    
    def classify_query(self, query: str) -> ClassificationResult:
        """
        Classify a query and return the most likely intent
        
        Args:
            query: Input query string
            
        Returns:
            ClassificationResult object with intent, confidence, and all scores
        """
        import time
        start_time = time.time()
        
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess(query)
        
        if not processed_query:
            logger.warning("No valid content found in query after preprocessing")
            return ClassificationResult(
                intent="unknown",
                confidence=0.0,
                all_scores={},
                processing_time_ms=0.0
            )
        
        # Get BERT embedding for query
        query_embedding = self.vectorizer.encode_single(processed_query)
        
        # Calculate similarities with all intents
        similarities = {}
        for intent, intent_embedding in self.intent_embeddings.items():
            if self.similarity_metric == 'cosine':
                similarity = self.similarity_calculator.cosine_similarity(
                    query_embedding, intent_embedding
                )
            else:  # euclidean
                similarity = self.similarity_calculator.euclidean_similarity(
                    query_embedding, intent_embedding
                )
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
        
        # Check confidence threshold
        if best_confidence < self.min_confidence_threshold:
            final_intent = "unknown"
        else:
            final_intent = best_intent
        
        processing_time = (time.time() - start_time) * 1000
        
        return ClassificationResult(
            intent=final_intent,
            confidence=best_confidence,
            all_scores=similarities,
            processing_time_ms=processing_time
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
                    all_scores={},
                    processing_time_ms=0.0
                ))
        return results
    
    def add_intent(self, intent_name: str, examples: List[str]):
        """Add a new intent to the classifier"""
        self.intents_config[intent_name] = examples
        self._compute_intent_embeddings()
        logger.info(f"Added new intent: {intent_name}")
    
    def update_intent(self, intent_name: str, examples: List[str]):
        """Update examples for an existing intent"""
        if intent_name not in self.intents_config:
            raise ValueError(f"Intent '{intent_name}' does not exist")
        
        self.intents_config[intent_name] = examples
        self._compute_intent_embeddings()
        logger.info(f"Updated intent: {intent_name}")
    
    def get_intents(self) -> List[str]:
        """Get list of all available intents"""
        return list(self.intents_config.keys())
    
    def save_config(self, filepath: Union[str, Path]):
        """Save current configuration to JSON file"""
        config = {
            "intents": self.intents_config,
            "model_name": self.model_name,
            "min_confidence_threshold": self.min_confidence_threshold,
            "similarity_metric": self.similarity_metric
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, config_filepath: Union[str, Path]):
        """Load classifier from configuration file"""
        with open(config_filepath, 'r') as f:
            config = json.load(f)
        
        return cls(
            intents_config=config.get("intents"),
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            min_confidence_threshold=config.get("min_confidence_threshold", 0.0),
            similarity_metric=config.get("similarity_metric", "cosine")
        )
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.vectorizer.embedding_dim,
            "similarity_metric": self.similarity_metric,
            "min_confidence_threshold": self.min_confidence_threshold,
            "total_intents": len(self.intents_config),
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }


def main():
    """Example usage of the BERT query classifier"""
    if not BERT_AVAILABLE:
        print("Error: BERT dependencies not available.")
        print("Install with: pip install sentence-transformers torch")
        return
    
    try:
        # Initialize classifier
        classifier = BERTQueryClassifier(
            model_name='all-MiniLM-L6-v2',  # Fast model
            min_confidence_threshold=0.3
        )
        
        # Test queries
        test_queries = [
            "Where is my order?",
            "How much does this product cost?",
            "Is this item still available in stock?",
            "My delivery failed to arrive yesterday",
            "I didn't receive a greeting card with my purchase",
            "What's the status of my shipment?",
            "Can you tell me the price?",
            "Do you have this in stock?",
            "Delivery issue with my package",
            "Missing thank you note"
        ]
        
        print("BERT Query Classification Results:")
        print("=" * 60)
        
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"Query: '{query}'")
            print(f"Intent: {result.intent}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Processing time: {result.processing_time_ms:.1f}ms")
            
            # Show top 3 scores
            sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top 3 intents:")
            for intent, score in sorted_scores:
                print(f"  {intent}: {score:.3f}")
            print("-" * 40)
        
        # Model info
        model_info = classifier.get_model_info()
        print(f"\nModel Information:")
        print(f"Model: {model_info['model_name']}")
        print(f"Embedding dimension: {model_info['embedding_dimension']}")
        print(f"Device: {model_info['device']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
