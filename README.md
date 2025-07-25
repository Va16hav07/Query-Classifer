# Query Classifier - Production Level

A robust, production-ready query classification system that uses pre-trained vocabulary files to classify user intents. Supports both frequency-based vocabularies and embedding-based approaches.

## Features

- **Multiple Vocabulary Formats**: Supports frequency-based vocabularies and pre-trained embeddings
- **TF-IDF Vectorization**: Advanced text vectorization for frequency-based vocabularies
- **Configurable Intents**: Easily customizable intent definitions
- **Batch Processing**: Efficient processing of multiple queries
- **Confidence Thresholding**: Configurable minimum confidence for classifications
- **Domain-Specific Configurations**: Pre-built configurations for different domains
- **Comprehensive Testing**: Full test suite with unit and integration tests
- **Production-Ready**: Robust error handling, logging, and performance optimization

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the basic example:
```python
from query_classifier import QueryClassifier

# Initialize classifier
classifier = QueryClassifier(
    vocab_file="vocab.txt",
    vocab_type="frequency"
)

# Classify a query
result = classifier.classify_query("Where is my order?")
print(f"Intent: {result.intent}, Confidence: {result.confidence:.3f}")
```

### Basic Usage

```python
from query_classifier import QueryClassifier

# Initialize with default e-commerce intents
classifier = QueryClassifier(
    vocab_file="/path/to/vocab.txt",
    vocab_type="frequency",  # or "embedding" for pre-trained embeddings
    min_confidence_threshold=0.1
)

# Single query classification
result = classifier.classify_query("Track my shipment")
print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence:.3f}")
print(f"All scores: {result.all_scores}")

# Batch processing
queries = ["Where is my order?", "How much does this cost?"]
results = classifier.batch_classify(queries)
```

## Configuration

### Default Intents

The classifier comes with pre-configured intents for common use cases:

- **order_status**: Order tracking and status inquiries
- **pricing**: Price and cost-related queries
- **availability**: Stock and availability questions
- **delivery_issue**: Delivery problems and issues
- **card_missing**: Missing cards or notes with orders

### Custom Intents

```python
custom_intents = {
    "refund": ["refund", "return", "money back"],
    "technical_support": ["broken", "not working", "error"]
}

classifier = QueryClassifier(
    vocab_file="vocab.txt",
    intents_config=custom_intents
)
```

### Domain-Specific Configurations

```python
from config import DOMAIN_CONFIGS

# Banking domain
banking_classifier = QueryClassifier(
    vocab_file="vocab.txt",
    intents_config=DOMAIN_CONFIGS["banking"]
)

# Healthcare domain
healthcare_classifier = QueryClassifier(
    vocab_file="vocab.txt",
    intents_config=DOMAIN_CONFIGS["healthcare"]
)
```

## Vocabulary File Formats

### Frequency-Based Format
```
the 1061396
of 593677
and 416629
order 12345
track 8765
```

### Embedding Format
```
the 0.1 0.2 0.3 0.4 0.5
of 0.2 0.1 0.4 0.3 0.5
order 0.8 0.1 0.2 0.3 0.4
```

## Advanced Features

### Confidence Thresholding

```python
classifier = QueryClassifier(
    vocab_file="vocab.txt",
    min_confidence_threshold=0.3
)

result = classifier.classify_query("ambiguous query")
if result.intent == "unknown":
    print("Low confidence classification rejected")
```

### Configuration Management

```python
# Save configuration
classifier.save_config("my_config.json")

# Load configuration
new_classifier = QueryClassifier.load_config(
    "my_config.json", 
    "vocab.txt"
)
```

### Adding New Intents

```python
# Add new intent
classifier.add_intent("warranty", ["warranty", "guarantee", "coverage"])

# Update existing intent
classifier.update_intent("pricing", ["price", "cost", "fee", "rate"])
```

## API Reference

### QueryClassifier

Main classifier class for intent classification.

#### Constructor Parameters

- `vocab_file` (str|Path): Path to vocabulary file
- `intents_config` (dict, optional): Intent definitions
- `vocab_type` (str): "frequency" or "embedding"
- `min_confidence_threshold` (float): Minimum confidence for classification

#### Methods

- `classify_query(query: str) -> ClassificationResult`: Classify single query
- `batch_classify(queries: List[str]) -> List[ClassificationResult]`: Batch classification
- `add_intent(name: str, keywords: List[str])`: Add new intent
- `update_intent(name: str, keywords: List[str])`: Update intent keywords
- `get_intents() -> List[str]`: Get available intents
- `save_config(filepath: str)`: Save configuration
- `load_config(config_file: str, vocab_file: str)`: Load from configuration

### ClassificationResult

Result object containing classification information.

#### Properties

- `intent` (str): Predicted intent name
- `confidence` (float): Confidence score (0-1)
- `all_scores` (dict): Scores for all intents

## Testing

Run the test suite:

```bash
python test_query_classifier.py
```

Run specific test categories:

```bash
# Unit tests only
python -m unittest test_query_classifier.TestVocabularyLoader

# Integration tests
python -m unittest test_query_classifier.TestIntegration
```

## Performance

### Benchmarks

On a typical modern system:
- Single query processing: ~0.5-2ms
- Batch processing: 500-2000 queries/second
- Memory usage: ~10-50MB (depending on vocabulary size)

### Optimization Tips

1. Use frequency-based vocabularies for faster processing
2. Enable batch processing for multiple queries
3. Set appropriate confidence thresholds to filter low-quality predictions
4. Cache the classifier instance for repeated use

## Examples

### E-commerce Customer Service

```python
from query_classifier import QueryClassifier
from config import ECOMMERCE_INTENTS

classifier = QueryClassifier(
    vocab_file="vocab.txt",
    intents_config=ECOMMERCE_INTENTS
)

# Customer queries
queries = [
    "I want to track my order",
    "What's the return policy?",
    "Is this item in stock?",
    "My payment was declined"
]

for query in queries:
    result = classifier.classify_query(query)
    print(f"'{query}' → {result.intent}")
```

### Restaurant Booking System

```python
restaurant_intents = {
    "reservation": ["book", "table", "reservation"],
    "menu": ["menu", "food", "dish", "specials"],
    "hours": ["hours", "open", "close", "time"],
    "location": ["where", "address", "directions"]
}

classifier = QueryClassifier(
    vocab_file="vocab.txt",
    intents_config=restaurant_intents
)
```

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure vocab.txt exists and path is correct
2. **Low accuracy**: Check intent keyword relevance to vocabulary
3. **Memory issues**: Use smaller vocabulary or frequency-based approach
4. **Slow performance**: Enable batch processing and check vocabulary size

### Error Handling

The classifier includes comprehensive error handling:

```python
try:
    result = classifier.classify_query("test query")
except ValueError as e:
    print(f"Invalid query: {e}")
except FileNotFoundError as e:
    print(f"Vocabulary file not found: {e}")
```

## 🆕 Sentence-BERT Implementation

**NEW**: Advanced semantic classification using Sentence-BERT for superior accuracy!

### Quick Start with Sentence-BERT

```python
from query_classifier import QueryClassifier

# Create Sentence-BERT classifier
classifier = QueryClassifier(use_sentence_bert=True)

# Classify with semantic understanding
result = classifier.classify_query("Update my payment method")
print(f"Intent: {result.intent}")  # BACKEND_QUERY
print(f"Confidence: {result.confidence:.3f}")  # Higher accuracy
```

### Sentence-BERT vs TF-IDF

| Feature | Sentence-BERT | TF-IDF |
|---------|---------------|--------|
| **Semantic Understanding** | ✅ Excellent | ❌ Limited |
| **Complex Queries** | ✅ Handles well | ❌ May struggle |
| **Speed** | ⚡ ~15ms | ⚡⚡ ~1ms |
| **Accuracy** | 🎯 Superior | 📊 Good for keywords |

### Binary Classification Labels

- **BACKEND_QUERY**: Order related questions like delivery, payment, address changes
- **PRODUCT_QUERY**: Product related queries like availability, types, occasions

## 🌐 Web Interface

Interactive web interface for testing and demonstration:

```bash
# Start with Sentence-BERT
export USE_SENTENCE_BERT=true
python3 mock_api_server.py

# Open browser to: http://localhost:5000/web
```

**Features:**
- 🔍 Single query classification
- 📦 Batch processing
- ⚙️ Mode switching (TF-IDF ↔ Sentence-BERT)
- 📊 Real-time results and metrics
- 🎯 Visual confidence indicators

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Changelog

### v1.0.0
- Initial production release
- Support for frequency-based and embedding vocabularies
- Comprehensive test suite
- Domain-specific configurations
- Batch processing capabilities
- Configuration management
