#!/bin/bash

# Query Classifier API Startup Script
# This script installs dependencies and starts the API server

echo "🚀 Starting Query Classifier API Setup..."
echo "========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✅ Python 3 found"

# Install Flask if not available
echo "📦 Installing required packages..."
pip install flask requests || {
    echo "⚠️  Failed to install Flask via pip. Trying with user install..."
    pip install --user flask requests
}

# Check if vocab.txt exists
if [ ! -f "vocab.txt" ]; then
    echo "❌ vocab.txt not found. Please ensure the vocabulary file exists."
    exit 1
fi

echo "✅ Vocabulary file found"

# Check if the main classifier file exists
if [ ! -f "query_classifier.py" ]; then
    echo "❌ query_classifier.py not found. Please ensure all files are present."
    exit 1
fi

echo "✅ Query classifier module found"

# Start the API server
echo ""
echo "🌟 Starting Query Classifier API Server..."
echo "=========================================="
echo ""
echo "📍 API will be available at: http://localhost:5000"
echo "🌐 Web interface will be available at: web_interface.html"
echo ""
echo "API Endpoints:"
echo "  GET  /                    - Health check"
echo "  POST /classify            - Classify single query"
echo "  POST /classify/batch      - Classify multiple queries"
echo "  GET  /intents             - Get available intents"
echo "  GET  /config              - Get configuration"
echo "  POST /config/domain       - Change domain"
echo ""
echo "📋 Example curl command:"
echo 'curl -X POST http://localhost:5000/classify -H "Content-Type: application/json" -d '"'"'{"query": "Where is my order?"}'"'"''
echo ""
echo "🔄 Press Ctrl+C to stop the server"
echo ""

# Run the API
python3 api.py
