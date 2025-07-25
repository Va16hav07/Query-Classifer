"""
REST API for Query Classifier

A Flask-based REST API that provides intent classification services.
Supports real-time query classification with configurable models.

Author: AI Assistant
Date: 2025-07-25
"""

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("Warning: flask-cors not available. Install with: pip install flask-cors")

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path

# Import our query classifier
from query_classifier import QueryClassifier, ClassificationResult
from config import DEFAULT_INTENTS, ECOMMERCE_INTENTS, DOMAIN_CONFIGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })  # Enable CORS for all routes

# Global classifier instance
classifier = None
classifier_config = {}

def initialize_classifier():
    """Initialize the global classifier instance"""
    global classifier, classifier_config
    
    try:
        # Default configuration
        config = {
            "vocab_file": "/home/vaibhav/Qery Classifer/vocab.txt",
            "vocab_type": "frequency",
            "intents_config": DEFAULT_INTENTS,
            "min_confidence_threshold": 0.1
        }
        
        # Load from config file if exists
        config_file = "api_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
                config.update(saved_config)
        
        classifier = QueryClassifier(
            vocab_file=config["vocab_file"],
            intents_config=config["intents_config"],
            vocab_type=config["vocab_type"],
            min_confidence_threshold=config["min_confidence_threshold"]
        )
        
        classifier_config = config
        logger.info("Query classifier initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize classifier: {e}")
        return False

@app.before_request
def before_first_request():
    """Initialize classifier on first request"""
    global classifier
    if classifier is None:
        if not initialize_classifier():
            logger.error("Failed to initialize classifier - API may not work properly")

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Query Classifier API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "classifier_loaded": classifier is not None
    })

@app.route('/classify', methods=['POST'])
def classify_query():
    """
    Classify a single query
    
    Expected JSON payload:
    {
        "query": "Where is my order?",
        "include_all_scores": false  // optional
    }
    
    Returns:
    {
        "query": "Where is my order?",
        "intent": "order_status",
        "confidence": 0.85,
        "all_scores": {...},  // if include_all_scores is true
        "processing_time_ms": 12.5,
        "timestamp": "2025-07-25T10:30:00"
    }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request body"
            }), 400
        
        query = data['query']
        include_all_scores = data.get('include_all_scores', False)
        
        if not isinstance(query, str) or not query.strip():
            return jsonify({
                "error": "Query must be a non-empty string"
            }), 400
        
        # Check if classifier is available
        if classifier is None:
            return jsonify({
                "error": "Classifier not initialized"
            }), 503
        
        # Classify the query
        result = classifier.classify_query(query)
        
        # Prepare response
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            "query": query,
            "intent": result.intent,
            "confidence": round(result.confidence, 4),
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if include_all_scores:
            response["all_scores"] = {
                intent: round(score, 4) 
                for intent, score in result.all_scores.items()
            }
        
        logger.info(f"Classified query: '{query}' -> {result.intent} ({result.confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in classify_query: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """
    Classify multiple queries in batch
    
    Expected JSON payload:
    {
        "queries": ["Where is my order?", "How much does this cost?"],
        "include_all_scores": false  // optional
    }
    
    Returns:
    {
        "results": [
            {
                "query": "Where is my order?",
                "intent": "order_status",
                "confidence": 0.85
            },
            ...
        ],
        "total_queries": 2,
        "processing_time_ms": 25.3,
        "timestamp": "2025-07-25T10:30:00"
    }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        if not data or 'queries' not in data:
            return jsonify({
                "error": "Missing 'queries' field in request body"
            }), 400
        
        queries = data['queries']
        include_all_scores = data.get('include_all_scores', False)
        
        if not isinstance(queries, list):
            return jsonify({
                "error": "Queries must be a list of strings"
            }), 400
        
        if len(queries) > 100:  # Limit batch size
            return jsonify({
                "error": "Maximum batch size is 100 queries"
            }), 400
        
        # Check if classifier is available
        if classifier is None:
            return jsonify({
                "error": "Classifier not initialized"
            }), 503
        
        # Classify all queries
        results = classifier.batch_classify(queries)
        
        # Prepare response
        processing_time = (time.time() - start_time) * 1000
        
        response_results = []
        for i, (query, result) in enumerate(zip(queries, results)):
            result_data = {
                "query": query,
                "intent": result.intent,
                "confidence": round(result.confidence, 4)
            }
            
            if include_all_scores:
                result_data["all_scores"] = {
                    intent: round(score, 4) 
                    for intent, score in result.all_scores.items()
                }
            
            response_results.append(result_data)
        
        response = {
            "results": response_results,
            "total_queries": len(queries),
            "processing_time_ms": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Classified {len(queries)} queries in batch")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in classify_batch: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/intents', methods=['GET'])
def get_intents():
    """
    Get all available intents
    
    Returns:
    {
        "intents": ["order_status", "pricing", "availability", ...],
        "total_intents": 5,
        "configurations": ["default", "ecommerce", "banking", "healthcare"]
    }
    """
    try:
        if classifier is None:
            return jsonify({
                "error": "Classifier not initialized"
            }), 503
        
        intents = classifier.get_intents()
        
        return jsonify({
            "intents": intents,
            "total_intents": len(intents),
            "configurations": list(DOMAIN_CONFIGS.keys()),
            "current_config": classifier_config.get("domain", "default")
        })
        
    except Exception as e:
        logger.error(f"Error in get_intents: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/config', methods=['GET'])
def get_config():
    """
    Get current classifier configuration
    
    Returns:
    {
        "vocab_type": "frequency",
        "min_confidence_threshold": 0.1,
        "vocab_file": "/path/to/vocab.txt",
        "total_vocabulary_size": 71290
    }
    """
    try:
        if classifier is None:
            return jsonify({
                "error": "Classifier not initialized"
            }), 503
        
        config_info = {
            "vocab_type": classifier_config.get("vocab_type", "frequency"),
            "min_confidence_threshold": classifier_config.get("min_confidence_threshold", 0.1),
            "vocab_file": classifier_config.get("vocab_file", "unknown"),
            "total_vocabulary_size": len(classifier.vocab) if hasattr(classifier, 'vocab') else 0,
            "intents_count": len(classifier.get_intents())
        }
        
        return jsonify(config_info)
        
    except Exception as e:
        logger.error(f"Error in get_config: {e}")
        return jsonify({
            "error": "Internal server error", 
            "message": str(e)
        }), 500

@app.route('/config/domain', methods=['POST'])
def change_domain():
    """
    Change the domain configuration
    
    Expected JSON payload:
    {
        "domain": "ecommerce"  // one of: default, ecommerce, banking, healthcare
    }
    
    Returns:
    {
        "message": "Domain changed successfully",
        "domain": "ecommerce",
        "intents": [...]
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        domain = data.get('domain')
        
        if domain not in DOMAIN_CONFIGS:
            return jsonify({
                "error": f"Invalid domain. Available domains: {list(DOMAIN_CONFIGS.keys())}"
            }), 400
        
        # Update classifier configuration
        global classifier, classifier_config
        
        classifier_config['intents_config'] = DOMAIN_CONFIGS[domain]
        classifier_config['domain'] = domain
        
        # Reinitialize classifier with new domain
        classifier = QueryClassifier(
            vocab_file=classifier_config["vocab_file"],
            intents_config=classifier_config["intents_config"],
            vocab_type=classifier_config["vocab_type"],
            min_confidence_threshold=classifier_config["min_confidence_threshold"]
        )
        
        # Save configuration
        with open("api_config.json", 'w') as f:
            json.dump(classifier_config, f, indent=2)
        
        logger.info(f"Changed domain to: {domain}")
        
        return jsonify({
            "message": "Domain changed successfully",
            "domain": domain,
            "intents": classifier.get_intents()
        })
        
    except Exception as e:
        logger.error(f"Error in change_domain: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get API usage statistics
    
    Returns:
    {
        "uptime_seconds": 3600,
        "total_requests": 150,
        "classifier_status": "healthy"
    }
    """
    # Simple stats - in production you'd use proper metrics storage
    return jsonify({
        "classifier_status": "healthy" if classifier is not None else "unhealthy",
        "available_endpoints": [
            "/classify",
            "/classify/batch", 
            "/intents",
            "/config",
            "/config/domain",
            "/stats"
        ],
        "api_version": "1.0.0"
    })

@app.route('/web')
def web_interface():
    """Serve the web interface"""
    try:
        with open('web_interface.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({'error': 'Web interface not found'}), 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "GET /",
            "POST /classify",
            "POST /classify/batch",
            "GET /intents", 
            "GET /config",
            "POST /config/domain",
            "GET /stats"
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The HTTP method is not allowed for this endpoint"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    # Development server
    print("Starting Query Classifier API...")
    print("API Documentation:")
    print("==================")
    print("POST /classify - Classify a single query")
    print("POST /classify/batch - Classify multiple queries")
    print("GET /intents - Get available intents")
    print("GET /config - Get current configuration")
    print("POST /config/domain - Change domain configuration")
    print("GET /stats - Get API statistics")
    print("GET / - Health check")
    print()
    print("Starting server on http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
