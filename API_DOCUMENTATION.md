# Query Classifier API Documentation

## Overview

The Query Classifier API provides intelligent intent classification for text queries using pre-trained vocabulary files. It supports both frequency-based and embedding-based approaches with configurable domain-specific intent sets.

## Quick Start

### 1. Start the API Server

```bash
# Option 1: Use the startup script (recommended)
./start_api.sh

# Option 2: Manual start
pip install flask requests
python api.py
```

The API will be available at `http://localhost:5000`

### 2. Test the API

```bash
# Health check
curl http://localhost:5000/

# Classify a query
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my order?"}'
```

### 3. Use the Web Interface

Open `web_interface.html` in your browser for a visual interface to interact with the API.

## API Endpoints

### GET / - Health Check

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "service": "Query Classifier API",
  "version": "1.0.0",
  "timestamp": "2025-07-25T10:30:00",
  "classifier_loaded": true
}
```

### POST /classify - Classify Single Query

Classify a single text query and return the predicted intent.

**Request Body:**
```json
{
  "query": "Where is my order?",
  "include_all_scores": false
}
```

**Parameters:**
- `query` (string, required): The text query to classify
- `include_all_scores` (boolean, optional): Include confidence scores for all intents

**Response:**
```json
{
  "query": "Where is my order?",
  "intent": "order_status",
  "confidence": 0.8542,
  "processing_time_ms": 12.5,
  "timestamp": "2025-07-25T10:30:00",
  "all_scores": {
    "order_status": 0.8542,
    "pricing": 0.0234,
    "availability": 0.0156,
    "delivery_issue": 0.0789,
    "card_missing": 0.0279
  }
}
```

### POST /classify/batch - Batch Classification

Classify multiple queries in a single request for improved performance.

**Request Body:**
```json
{
  "queries": [
    "Where is my order?",
    "How much does this cost?",
    "Is this item available?"
  ],
  "include_all_scores": false
}
```

**Parameters:**
- `queries` (array of strings, required): List of queries to classify (max 100)
- `include_all_scores` (boolean, optional): Include confidence scores for all intents

**Response:**
```json
{
  "results": [
    {
      "query": "Where is my order?",
      "intent": "order_status",
      "confidence": 0.8542
    },
    {
      "query": "How much does this cost?",
      "intent": "pricing", 
      "confidence": 0.9234
    },
    {
      "query": "Is this item available?",
      "intent": "availability",
      "confidence": 0.7891
    }
  ],
  "total_queries": 3,
  "processing_time_ms": 28.7,
  "timestamp": "2025-07-25T10:30:00"
}
```

### GET /intents - Get Available Intents

Retrieve all currently configured intents.

**Response:**
```json
{
  "intents": [
    "order_status",
    "pricing", 
    "availability",
    "delivery_issue",
    "card_missing"
  ],
  "total_intents": 5,
  "configurations": ["default", "ecommerce", "banking", "healthcare"],
  "current_config": "default"
}
```

### GET /config - Get Configuration

Get current classifier configuration details.

**Response:**
```json
{
  "vocab_type": "frequency",
  "min_confidence_threshold": 0.1,
  "vocab_file": "/path/to/vocab.txt",
  "total_vocabulary_size": 71290,
  "intents_count": 5
}
```

### POST /config/domain - Change Domain

Switch between different domain-specific intent configurations.

**Request Body:**
```json
{
  "domain": "ecommerce"
}
```

**Available Domains:**
- `default`: Basic intent set
- `ecommerce`: E-commerce specific intents
- `banking`: Banking and financial intents  
- `healthcare`: Healthcare and medical intents

**Response:**
```json
{
  "message": "Domain changed successfully",
  "domain": "ecommerce",
  "intents": [
    "order_status",
    "pricing",
    "availability", 
    "delivery_issue",
    "card_missing",
    "return_refund",
    "customer_service",
    "product_info",
    "account_issues",
    "payment_issues",
    "technical_support"
  ]
}
```

### GET /stats - API Statistics

Get basic API statistics and information.

**Response:**
```json
{
  "classifier_status": "healthy",
  "available_endpoints": [
    "/classify",
    "/classify/batch",
    "/intents", 
    "/config",
    "/config/domain",
    "/stats"
  ],
  "api_version": "1.0.0"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

### 400 Bad Request
```json
{
  "error": "Missing 'query' field in request body"
}
```

### 503 Service Unavailable
```json
{
  "error": "Classifier not initialized"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Detailed error description"
}
```

## Client Examples

### Python Client

```python
import requests

class QueryClassifierClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def classify(self, query):
        response = requests.post(
            f"{self.base_url}/classify",
            json={"query": query}
        )
        return response.json()

# Usage
client = QueryClassifierClient()
result = client.classify("Where is my order?")
print(f"Intent: {result['intent']}, Confidence: {result['confidence']}")
```

### JavaScript/Node.js Client

```javascript
const fetch = require('node-fetch');

class QueryClassifierClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async classify(query) {
        const response = await fetch(`${this.baseUrl}/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        return response.json();
    }
}

// Usage
const client = new QueryClassifierClient();
client.classify("Where is my order?")
    .then(result => {
        console.log(`Intent: ${result.intent}, Confidence: ${result.confidence}`);
    });
```

### cURL Examples

```bash
# Single query classification
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my order?", "include_all_scores": true}'

# Batch classification
curl -X POST http://localhost:5000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "Where is my order?",
      "How much does this cost?"
    ]
  }'

# Change domain
curl -X POST http://localhost:5000/config/domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "banking"}'

# Get available intents
curl http://localhost:5000/intents
```

## Domain Configurations

### Default Domain
Basic intent classification for general queries:
- `order_status`: Order tracking and status
- `pricing`: Price and cost inquiries
- `availability`: Stock and availability
- `delivery_issue`: Delivery problems
- `card_missing`: Missing cards or notes

### E-commerce Domain
Extended intents for online retail:
- All default intents plus:
- `return_refund`: Returns and refunds
- `customer_service`: Customer support
- `product_info`: Product specifications
- `account_issues`: Account problems
- `payment_issues`: Payment failures
- `technical_support`: Technical problems

### Banking Domain
Financial and banking specific intents:
- `account_balance`: Balance inquiries
- `transfer`: Money transfers
- `loan`: Loan applications
- `fraud`: Fraud reports
- `card_issues`: Card problems

### Healthcare Domain
Medical and healthcare intents:
- `appointment`: Appointment booking
- `prescription`: Prescription refills
- `insurance`: Insurance coverage
- `symptoms`: Symptom descriptions
- `emergency`: Emergency situations

## Performance Considerations

### Response Times
- Single query: ~5-15ms
- Batch queries: ~20-50ms for 10 queries
- Domain switching: ~100-200ms

### Rate Limits
- No built-in rate limiting (implement as needed)
- Batch endpoint limited to 100 queries per request

### Scaling
- Stateless design enables horizontal scaling
- Consider load balancing for production use
- Database caching for improved performance

## Production Deployment

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=false
export API_HOST=0.0.0.0
export API_PORT=5000
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "api.py"]
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Security Considerations

### Authentication
- No built-in authentication (implement JWT/API keys as needed)
- Consider rate limiting for production use

### Input Validation
- Query length limits (max 1000 characters)
- Batch size limits (max 100 queries)
- Content type validation

### HTTPS
- Use HTTPS in production
- Configure proper SSL/TLS certificates

## Monitoring and Logging

### Logging
The API logs all requests and errors:
```
2025-07-25 10:30:00 - INFO - Classified query: 'Where is my order?' -> order_status (0.854)
2025-07-25 10:30:05 - ERROR - Error in classify_query: Invalid input
```

### Health Monitoring
- Use the `/` endpoint for health checks
- Monitor response times and error rates
- Set up alerts for API downtime

## Troubleshooting

### Common Issues

1. **API not starting**
   - Check if port 5000 is available
   - Verify Python and Flask installation
   - Ensure vocab.txt file exists

2. **Classifier not loaded**
   - Check vocab.txt file format
   - Verify file permissions
   - Review server logs for errors

3. **Low accuracy**
   - Verify vocabulary relevance to your domain
   - Consider using domain-specific configurations
   - Adjust confidence thresholds

4. **Slow response times**
   - Use batch endpoints for multiple queries
   - Consider vocabulary size optimization
   - Monitor server resources

### Debug Mode
Run the API in debug mode for detailed error information:
```bash
export FLASK_DEBUG=true
python api.py
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review server logs
3. Test with the web interface
4. Verify API endpoints with curl

## Changelog

### v1.0.0 (2025-07-25)
- Initial API release
- Single and batch query classification
- Domain switching support
- Web interface
- Comprehensive documentation
