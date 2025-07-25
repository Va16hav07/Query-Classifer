"""
API Client Example

This script demonstrates how to use the Query Classifier API
from different programming languages and scenarios.

Author: AI Assistant
Date: 2025-07-25
"""

import requests
import json
import time

class QueryClassifierClient:
    """Python client for the Query Classifier API"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def classify_query(self, query, include_all_scores=False):
        """Classify a single query"""
        payload = {
            "query": query,
            "include_all_scores": include_all_scores
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def classify_batch(self, queries, include_all_scores=False):
        """Classify multiple queries"""
        payload = {
            "queries": queries,
            "include_all_scores": include_all_scores
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/classify/batch",
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def get_intents(self):
        """Get all available intents"""
        try:
            response = self.session.get(f"{self.base_url}/intents")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def get_config(self):
        """Get current configuration"""
        try:
            response = self.session.get(f"{self.base_url}/config")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Request failed: {e}")
            return None
    
    def change_domain(self, domain):
        """Change the domain configuration"""
        payload = {"domain": domain}
        
        try:
            response = self.session.post(
                f"{self.base_url}/config/domain",
                data=json.dumps(payload)
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Request failed: {e}")
            return None

def demo_basic_usage():
    """Demonstrate basic API usage"""
    print("=" * 60)
    print("BASIC API USAGE DEMO")
    print("=" * 60)
    
    # Initialize client
    client = QueryClassifierClient()
    
    # Health check
    health = client.health_check()
    if not health:
        print("❌ API is not accessible. Make sure the server is running on http://localhost:5000")
        return
    
    print(f"✅ API Status: {health['status']}")
    print(f"📊 Classifier Loaded: {health['classifier_loaded']}")
    print()
    
    # Single query classification
    print("🔍 Single Query Classification:")
    print("-" * 40)
    
    query = "Where is my order?"
    result = client.classify_query(query, include_all_scores=True)
    
    if result:
        print(f"Query: '{result['query']}'")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Processing Time: {result['processing_time_ms']} ms")
        print("All Scores:")
        for intent, score in result['all_scores'].items():
            print(f"  {intent}: {score}")
    print()
    
    # Batch classification
    print("📦 Batch Classification:")
    print("-" * 40)
    
    queries = [
        "How much does this cost?",
        "Is this item available?", 
        "My delivery failed",
        "I need customer support"
    ]
    
    batch_result = client.classify_batch(queries)
    
    if batch_result:
        print(f"Processed {batch_result['total_queries']} queries in {batch_result['processing_time_ms']} ms")
        for result in batch_result['results']:
            print(f"'{result['query']}' → {result['intent']} ({result['confidence']})")
    print()

def demo_domain_switching():
    """Demonstrate domain switching"""
    print("=" * 60)
    print("DOMAIN SWITCHING DEMO")
    print("=" * 60)
    
    client = QueryClassifierClient()
    
    # Get current intents
    intents = client.get_intents()
    if intents:
        print(f"Current Domain: {intents.get('current_config', 'default')}")
        print(f"Available Intents: {intents['intents']}")
        print()
    
    # Switch to banking domain
    print("🏦 Switching to Banking Domain:")
    banking_result = client.change_domain("banking")
    if banking_result:
        print(f"✅ {banking_result['message']}")
        print(f"New Intents: {banking_result['intents']}")
        print()
        
        # Test with banking query
        banking_query = "What's my account balance?"
        result = client.classify_query(banking_query)
        if result:
            print(f"Banking Query: '{result['query']}'")
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']}")
    print()
    
    # Switch to healthcare domain  
    print("🏥 Switching to Healthcare Domain:")
    healthcare_result = client.change_domain("healthcare")
    if healthcare_result:
        print(f"✅ {healthcare_result['message']}")
        print(f"New Intents: {healthcare_result['intents']}")
        print()
        
        # Test with healthcare query
        healthcare_query = "I need to book an appointment"
        result = client.classify_query(healthcare_query)
        if result:
            print(f"Healthcare Query: '{result['query']}'")
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']}")
    print()

def demo_performance_testing():
    """Demonstrate performance testing"""
    print("=" * 60)
    print("PERFORMANCE TESTING DEMO")
    print("=" * 60)
    
    client = QueryClassifierClient()
    
    # Single query performance
    query = "Where is my order?"
    iterations = 10
    
    print(f"🚀 Testing single query performance ({iterations} iterations):")
    start_time = time.time()
    
    for i in range(iterations):
        result = client.classify_query(query)
        if not result:
            print(f"Request {i+1} failed")
            break
    
    total_time = time.time() - start_time
    avg_time = (total_time / iterations) * 1000
    
    print(f"Average response time: {avg_time:.2f} ms")
    print(f"Requests per second: {iterations / total_time:.1f}")
    print()
    
    # Batch performance
    batch_queries = [
        "track my order",
        "what's the price", 
        "is it available",
        "delivery problem",
        "need help"
    ] * 4  # 20 queries total
    
    print(f"📦 Testing batch performance ({len(batch_queries)} queries):")
    start_time = time.time()
    
    batch_result = client.classify_batch(batch_queries)
    
    if batch_result:
        api_time = time.time() - start_time
        processing_time = batch_result['processing_time_ms']
        
        print(f"Total API time: {api_time * 1000:.2f} ms")
        print(f"Processing time: {processing_time:.2f} ms")
        print(f"Network overhead: {(api_time * 1000) - processing_time:.2f} ms")
        print(f"Queries per second: {len(batch_queries) / api_time:.1f}")

def generate_curl_examples():
    """Generate curl command examples"""
    print("=" * 60)
    print("CURL COMMAND EXAMPLES")
    print("=" * 60)
    
    examples = [
        {
            "title": "Health Check",
            "command": """curl -X GET http://localhost:5000/"""
        },
        {
            "title": "Classify Single Query",
            "command": """curl -X POST http://localhost:5000/classify \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Where is my order?", "include_all_scores": true}'"""
        },
        {
            "title": "Batch Classification",
            "command": """curl -X POST http://localhost:5000/classify/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "queries": [
      "How much does this cost?", 
      "Is this available?"
    ],
    "include_all_scores": false
  }'"""
        },
        {
            "title": "Get Available Intents",
            "command": """curl -X GET http://localhost:5000/intents"""
        },
        {
            "title": "Change Domain to E-commerce",
            "command": """curl -X POST http://localhost:5000/config/domain \\
  -H "Content-Type: application/json" \\
  -d '{"domain": "ecommerce"}'"""
        },
        {
            "title": "Get Configuration",
            "command": """curl -X GET http://localhost:5000/config"""
        }
    ]
    
    for example in examples:
        print(f"📋 {example['title']}:")
        print(f"```bash")
        print(f"{example['command']}")
        print(f"```")
        print()

def generate_javascript_example():
    """Generate JavaScript fetch examples"""
    print("=" * 60)
    print("JAVASCRIPT EXAMPLES")
    print("=" * 60)
    
    js_code = """
// JavaScript example using fetch API

const API_BASE_URL = 'http://localhost:5000';

// Classify a single query
async function classifyQuery(query) {
    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                include_all_scores: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Classification result:', result);
        return result;
        
    } catch (error) {
        console.error('Error classifying query:', error);
        return null;
    }
}

// Batch classification
async function classifyBatch(queries) {
    try {
        const response = await fetch(`${API_BASE_URL}/classify/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                queries: queries,
                include_all_scores: false
            })
        });
        
        const result = await response.json();
        console.log('Batch results:', result);
        return result;
        
    } catch (error) {
        console.error('Error in batch classification:', error);
        return null;
    }
}

// Example usage
(async () => {
    // Single query
    await classifyQuery("Where is my order?");
    
    // Batch queries
    await classifyBatch([
        "How much does this cost?",
        "Is this item available?",
        "My delivery failed"
    ]);
})();
"""
    
    print("```javascript")
    print(js_code.strip())
    print("```")
    print()

def main():
    """Run all demonstrations"""
    print("QUERY CLASSIFIER API CLIENT DEMO")
    print("==================================")
    print()
    print("⚠️  Make sure the API server is running: python api.py")
    print()
    
    try:
        demo_basic_usage()
        demo_domain_switching()
        demo_performance_testing()
        generate_curl_examples()
        generate_javascript_example()
        
        print("=" * 60)
        print("🎉 ALL DEMOS COMPLETED!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Start the API: python api.py")
        print("2. Use the examples above to integrate with your application")
        print("3. Check the API documentation at http://localhost:5000")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()
