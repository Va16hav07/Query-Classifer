"""
Configuration file for Query Classifier

This file contains default configurations and examples for customizing
the query classifier behavior.
"""

# Default intent configurations
DEFAULT_INTENTS = {
    "order_status": [
        "order", "track", "status", "tracking", "shipment", "delivery",
        "package", "shipping", "dispatch", "sent", "progress", "location"
    ],
    "pricing": [
        "price", "cost", "charge", "fee", "expensive", "cheap", "money",
        "dollar", "payment", "bill", "amount", "rate", "quote"
    ],
    "availability": [
        "available", "stock", "have", "supply", "inventory", "sold",
        "out", "shortage", "quantity", "left", "remaining", "in stock"
    ],
    "delivery_issue": [
        "delivery", "failed", "recipient", "address", "shipping", "delayed",
        "problem", "issue", "wrong", "missing", "damaged", "lost"
    ],
    "card_missing": [
        "card", "note", "message", "missing", "forgot", "include",
        "greeting", "gift", "thank", "letter", "envelope"
    ],
    "return_refund": [
        "return", "refund", "exchange", "replace", "defective", "warranty",
        "money back", "cancel", "unwanted", "wrong item"
    ],
    "customer_service": [
        "help", "support", "contact", "representative", "agent", "human",
        "speak", "talk", "call", "phone", "email", "chat"
    ]
}

# Extended intent configurations for e-commerce
ECOMMERCE_INTENTS = {
    **DEFAULT_INTENTS,
    "product_info": [
        "specifications", "details", "features", "description", "manual",
        "dimensions", "weight", "color", "size", "material"
    ],
    "account_issues": [
        "login", "password", "account", "profile", "settings", "email",
        "username", "forgot", "reset", "access", "locked"
    ],
    "payment_issues": [
        "payment", "credit card", "billing", "invoice", "charge", "transaction",
        "declined", "failed", "method", "paypal", "bank"
    ],
    "technical_support": [
        "broken", "not working", "error", "bug", "malfunction", "repair",
        "fix", "troubleshoot", "installation", "setup", "configure"
    ]
}

# Configuration for different domains
DOMAIN_CONFIGS = {
    "general": DEFAULT_INTENTS,
    "ecommerce": ECOMMERCE_INTENTS,
    "banking": {
        "account_balance": ["balance", "amount", "funds", "money", "account"],
        "transfer": ["transfer", "send", "move", "pay", "wire"],
        "loan": ["loan", "mortgage", "credit", "borrow", "finance"],
        "fraud": ["fraud", "suspicious", "unauthorized", "stolen", "hack"],
        "card_issues": ["card", "blocked", "lost", "stolen", "pin", "activate"]
    },
    "healthcare": {
        "appointment": ["appointment", "schedule", "book", "doctor", "visit"],
        "prescription": ["prescription", "medication", "pharmacy", "refill", "drug"],
        "insurance": ["insurance", "coverage", "claim", "copay", "deductible"],
        "symptoms": ["symptoms", "pain", "sick", "fever", "cough", "headache"],
        "emergency": ["emergency", "urgent", "severe", "immediate", "critical"]
    }
}

# Preprocessing configurations
PREPROCESSING_CONFIG = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": False,  # Keep stopwords for now as they might be important
    "stemming": False,  # Can be enabled if needed
    "min_word_length": 1
}

# Model configurations
MODEL_CONFIG = {
    "vocab_type": "frequency",  # or "embedding"
    "min_confidence_threshold": 0.1,
    "similarity_metric": "cosine",  # cosine, euclidean, manhattan
    "normalization": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": "query_classifier.log"
}

# Performance settings
PERFORMANCE_CONFIG = {
    "batch_size": 100,
    "cache_vectors": True,
    "parallel_processing": False,
    "max_workers": 4
}

# Validation settings
VALIDATION_CONFIG = {
    "cross_validation_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# Sentence-BERT Configuration
SENTENCE_BERT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "labels": ["BACKEND_QUERY", "PRODUCT_QUERY"],
    "label_descriptions": {
        "BACKEND_QUERY": "Order related questions like delivery, payment, address changes",
        "PRODUCT_QUERY": "Product related queries like availability, types, occasions"
    },
    "similarity_threshold": 0.1,
    "use_label_descriptions": True  # Use descriptive text instead of just label names
}
