#!/usr/bin/env python3
"""
Setup and installation script for Query Classifier

This script handles the initial setup and validation of the query classifier system.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def validate_files():
    """Validate that all necessary files exist"""
    required_files = [
        "vocab.txt",
        "query_classifier.py",
        "config.py",
        "requirements.txt"
    ]
    
    print("Validating files...")
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✓ {file} found")
    
    if missing_files:
        print("✗ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✓ All required files present")
    return True


def test_basic_functionality():
    """Test basic classifier functionality"""
    print("Testing basic functionality...")
    
    try:
        # Import and test the classifier
        from query_classifier import QueryClassifier
        
        # Initialize classifier
        classifier = QueryClassifier(
            vocab_file="vocab.txt",
            vocab_type="frequency",
            min_confidence_threshold=0.0
        )
        
        # Test a simple query
        result = classifier.classify_query("where is my order")
        
        print(f"✓ Test classification successful:")
        print(f"  Query: 'where is my order'")
        print(f"  Intent: {result.intent}")
        print(f"  Confidence: {result.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def run_tests():
    """Run the test suite"""
    print("Running test suite...")
    
    try:
        # Run the test file
        result = subprocess.run([sys.executable, "test_query_classifier.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print("✗ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file"""
    print("Creating sample configuration...")
    
    try:
        from query_classifier import QueryClassifier
        from config import DEFAULT_INTENTS
        
        classifier = QueryClassifier(
            vocab_file="vocab.txt",
            intents_config=DEFAULT_INTENTS,
            vocab_type="frequency",
            min_confidence_threshold=0.1
        )
        
        classifier.save_config("sample_config.json")
        print("✓ Sample configuration saved to sample_config.json")
        return True
        
    except Exception as e:
        print(f"✗ Error creating sample configuration: {e}")
        return False


def main():
    """Main setup function"""
    print("=" * 60)
    print("QUERY CLASSIFIER SETUP")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    steps = [
        ("Validating files", validate_files),
        ("Installing requirements", install_requirements),
        ("Testing basic functionality", test_basic_functionality),
        ("Creating sample configuration", create_sample_config),
        ("Running test suite", run_tests)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if step_func():
            success_count += 1
        else:
            print(f"Setup step '{step_name}' failed. Continuing with remaining steps...")
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"Completed: {success_count}/{len(steps)} steps")
    
    if success_count == len(steps):
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python demo.py' to see examples")
        print("2. Check README.md for detailed usage instructions")
        print("3. Customize intents in config.py for your use case")
    else:
        print("⚠ Setup completed with some issues")
        print("Check the error messages above and resolve any missing dependencies")
    
    return success_count == len(steps)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
