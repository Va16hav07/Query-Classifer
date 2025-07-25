#!/usr/bin/env python3
"""
Simple startup script for the Query Classifier API
"""

import os
import sys
from pathlib import Path

def main():
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 Starting Query Classifier API...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    try:
        # Import and start the API
        from api import app, initialize_classifier
        
        print("📚 Initializing classifier...")
        initialize_classifier()
        
        print("🌐 Starting Flask server...")
        print("📡 API will be available at: http://localhost:5000")
        print("🖥️  Web interface at: http://localhost:5000/web")
        print("📋 API documentation at: http://localhost:5000/docs")
        print("\n✨ Ready to classify queries!")
        print("🛑 Press Ctrl+C to stop\n")
        
        # Start the server
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to True for development
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
