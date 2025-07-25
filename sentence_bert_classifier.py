"""
Sentence-BERT Query Classifier

A modern query classification system using Sentence-BERT embeddings
with cosine similarity for binary classification between BACKEND_QUERY 
and PRODUCT_QUERY categories.

Author: AI Assistant
Date: 2025-07-25
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Sentence-BERT imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("✓ Sentence-Transformers available")
except ImportError as e:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print(f"⚠ Sentence-Transformers not available: {e}")
    print("Install with: pip install sentence-transformers")

from config import SENTENCE_BERT_CONFIG

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


class SentenceBERTClassifier:
    """
    Sentence-BERT based query classifier for binary classification
    
    Uses the 'all-MiniLM-L6-v2' model to encode queries and label descriptions,
    then uses cosine similarity to determine the best matching category.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 labels: List[str] = None,
                 label_descriptions: Dict[str, str] = None,
                 similarity_threshold: float = 0.1,
                 use_label_descriptions: bool = True):
        """
        Initialize the Sentence-BERT classifier
        
        Args:
            model_name: Sentence-BERT model name (default: 'all-MiniLM-L6-v2')
            labels: List of labels for classification
            label_descriptions: Mapping of labels to descriptive text
            similarity_threshold: Minimum similarity threshold for classification
            use_label_descriptions: Whether to use descriptions instead of label names
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence-Transformers not available. Install with: pip install sentence-transformers")
        
        # Use config defaults if not provided
        self.labels = labels or SENTENCE_BERT_CONFIG["labels"]
        self.label_descriptions = label_descriptions or SENTENCE_BERT_CONFIG["label_descriptions"]
        self.similarity_threshold = similarity_threshold or SENTENCE_BERT_CONFIG["similarity_threshold"]
        self.use_label_descriptions = use_label_descriptions
        
        logger.info(f"Initializing Sentence-BERT classifier with model: {model_name}")
        logger.info(f"Labels: {self.labels}")
        
        # Load the Sentence-BERT model
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"✓ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            raise
        
        # Precompute label embeddings
        self._compute_label_embeddings()
    
    def _compute_label_embeddings(self):
        """Precompute embeddings for all labels"""
        self.label_embeddings = {}
        
        for label in self.labels:
            if self.use_label_descriptions and label in self.label_descriptions:
                # Use descriptive text for better semantic matching
                text_to_encode = self.label_descriptions[label]
                logger.info(f"Encoding label '{label}' with description: '{text_to_encode}'")
            else:
                # Use label name directly
                text_to_encode = label
                logger.info(f"Encoding label '{label}' directly")
            
            try:
                embedding = self.model.encode(text_to_encode, convert_to_tensor=False)
                self.label_embeddings[label] = np.array(embedding)
                logger.debug(f"✓ Computed embedding for label '{label}' (shape: {embedding.shape})")
            except Exception as e:
                logger.error(f"Failed to compute embedding for label '{label}': {e}")
                # Use zero vector as fallback
                self.label_embeddings[label] = np.zeros(self.embedding_dim)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Ensure vectors are 1D
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Calculate dot product
        dot_product = np.dot(vec1, vec2)
        
        # Calculate norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure result is in [-1, 1] range
        return np.clip(similarity, -1.0, 1.0)
    
    def classify_query(self, query: str) -> ClassificationResult:
        """
        Classify a query using Sentence-BERT embeddings and cosine similarity
        
        Args:
            query: Input query string
            
        Returns:
            ClassificationResult with predicted label and confidence scores
        """
        start_time = time.time()
        
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        logger.debug(f"Classifying query: '{query}'")
        
        try:
            # Encode the query using Sentence-BERT
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            query_embedding = np.array(query_embedding)
            
            # Calculate cosine similarity with each label embedding
            similarities = {}
            for label, label_embedding in self.label_embeddings.items():
                similarity = self._cosine_similarity(query_embedding, label_embedding)
                similarities[label] = float(similarity)
                logger.debug(f"Similarity with '{label}': {similarity:.4f}")
            
            # Find the best match
            if not similarities:
                logger.warning("No similarities computed")
                return ClassificationResult(
                    intent="unknown",
                    confidence=0.0,
                    all_scores={},
                    processing_time_ms=0.0
                )
            
            best_label = max(similarities, key=similarities.get)
            best_confidence = similarities[best_label]
            
            # Always return the label with highest confidence (never "unknown")
            predicted_label = best_label
            
            if best_confidence < self.similarity_threshold:
                logger.info(f"Best confidence {best_confidence:.4f} below threshold {self.similarity_threshold}, but returning best match: {predicted_label}")
            else:
                logger.info(f"Best confidence {best_confidence:.4f} above threshold {self.similarity_threshold}")
            
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
            logger.error(f"Error during classification: {e}")
            processing_time_ms = (time.time() - start_time) * 1000
            return ClassificationResult(
                intent="error",
                confidence=0.0,
                all_scores={},
                processing_time_ms=processing_time_ms
            )
    
    def batch_classify(self, queries: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple queries in batch
        
        Args:
            queries: List of query strings
            
        Returns:
            List of ClassificationResult objects
        """
        logger.info(f"Batch classifying {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            try:
                result = self.classify_query(query)
                results.append(result)
                logger.debug(f"Batch item {i+1}/{len(queries)}: '{query}' → {result.intent}")
            except Exception as e:
                logger.error(f"Error classifying query {i+1}: {e}")
                results.append(ClassificationResult(
                    intent="error",
                    confidence=0.0,
                    all_scores={},
                    processing_time_ms=0.0
                ))
        
        logger.info(f"Batch classification completed: {len(results)} results")
        return results
    
    def add_label(self, label: str, description: str = None):
        """
        Add a new label to the classifier
        
        Args:
            label: Label name
            description: Optional descriptive text for the label
        """
        if label not in self.labels:
            self.labels.append(label)
            
        if description:
            self.label_descriptions[label] = description
            
        # Recompute embeddings to include new label
        self._compute_label_embeddings()
        logger.info(f"Added new label: '{label}'")
    
    def update_label_description(self, label: str, description: str):
        """
        Update the description for an existing label
        
        Args:
            label: Label name
            description: New descriptive text
        """
        if label not in self.labels:
            raise ValueError(f"Label '{label}' not found")
            
        self.label_descriptions[label] = description
        
        # Recompute embeddings
        self._compute_label_embeddings()
        logger.info(f"Updated description for label: '{label}'")
    
    def get_labels(self) -> List[str]:
        """Get list of available labels"""
        return self.labels.copy()
    
    def get_model_info(self) -> Dict[str, Union[str, int]]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model._modules['0'].auto_model.name_or_path,
            "embedding_dimension": self.embedding_dim,
            "labels": self.labels,
            "similarity_threshold": self.similarity_threshold,
            "use_label_descriptions": self.use_label_descriptions
        }


def create_default_classifier() -> SentenceBERTClassifier:
    """Create a classifier with default configuration"""
    return SentenceBERTClassifier(
        model_name=SENTENCE_BERT_CONFIG["model_name"],
        labels=SENTENCE_BERT_CONFIG["labels"],
        label_descriptions=SENTENCE_BERT_CONFIG["label_descriptions"],
        similarity_threshold=SENTENCE_BERT_CONFIG["similarity_threshold"],
        use_label_descriptions=SENTENCE_BERT_CONFIG["use_label_descriptions"]
    )


def main():
    """Example usage of the Sentence-BERT classifier"""
    print("=" * 60)
    print("SENTENCE-BERT QUERY CLASSIFIER DEMO")
    print("=" * 60)
    
    try:
        # Create classifier with default configuration
        classifier = create_default_classifier()
        
        print(f"Model info: {classifier.get_model_info()}")
        print()
        
        # Test queries
        test_queries = [
            "Where is my order?",
            "Track my shipment status",
            "Change my delivery address", 
            "What payment methods do you accept?",
            "What types of flowers do you have?",
            "Do you have roses available?",
            "What occasions are suitable for this product?",
            "Show me your product catalog",
            "I need flowers for a wedding",
            "Can I see different bouquet options?"
        ]
        
        print("Classifying test queries...")
        print("-" * 40)
        
        for query in test_queries:
            result = classifier.classify_query(query)
            print(f"Query: '{query}'")
            print(f"Classification: {result.intent}")
            print(f"Confidence: {result.confidence:.4f}")
            print(f"Processing time: {result.processing_time_ms:.2f}ms")
            print(f"All scores: {result.all_scores}")
            print("-" * 40)
        
        # Batch processing demo
        print("\nBatch processing demo...")
        batch_results = classifier.batch_classify(test_queries[:5])
        print(f"Processed {len(batch_results)} queries in batch")
        
        # Calculate average processing time
        avg_time = np.mean([r.processing_time_ms for r in batch_results if r.processing_time_ms])
        print(f"Average processing time: {avg_time:.2f}ms")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install sentence-transformers: pip install sentence-transformers")


if __name__ == "__main__":
    main()
