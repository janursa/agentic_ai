"""
Test script to verify LlamaIndex setup and basic agent functionality.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import llama_index
        print("✓ llama-index installed")
    except ImportError as e:
        print(f"✗ llama-index not installed: {e}")
        return False
    
    try:
        from llama_index.llms.openai import OpenAI
        print("✓ llama-index-llms-openai installed")
    except ImportError as e:
        print(f"✗ llama-index-llms-openai not installed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv installed")
    except ImportError as e:
        print(f"✗ python-dotenv not installed: {e}")
        return False
    
    return True


def test_data_structure():
    """Test that data directories exist."""
    print("\nTesting data structure...")
    data_dir = Path(__file__).parent.parent / "data"
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    print(f"✓ Data directory exists: {data_dir}")
    
    literature_dir = data_dir / "literature"
    if not literature_dir.exists():
        print(f"✗ Literature directory not found: {literature_dir}")
        return False
    print(f"✓ Literature directory exists: {literature_dir}")
    
    manuscript_file = data_dir / "manuscrtipt.txt"
    if not manuscript_file.exists():
        print(f"✗ Manuscript file not found: {manuscript_file}")
        return False
    print(f"✓ Manuscript file exists: {manuscript_file}")
    
    return True


def test_api_key():
    """Test that OpenAI API key is available."""
    print("\nTesting API key...")
    import os
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("✗ OPENAI_API_KEY not found in environment")
        return False
    
    if not api_key.startswith("sk-"):
        print("✗ OPENAI_API_KEY doesn't look valid (should start with 'sk-')")
        return False
    
    print(f"✓ OPENAI_API_KEY found: {api_key[:10]}...")
    return True


def test_helper_functions():
    """Test that helper functions can be imported."""
    print("\nTesting helper functions...")
    try:
        from helper import (
            get_aging_signature,
            get_intervention_signature,
            get_available_cell_types,
            get_available_interventions,
            get_available_analysis_types
        )
        print("✓ Helper functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import helper functions: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("LlamaIndex Setup Verification")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Structure", test_data_structure),
        ("API Key", test_api_key),
        ("Helper Functions", test_helper_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ All tests passed! You're ready to run the agent.")
        print("\nNext steps:")
        print("  1. Run in interactive mode: bash scripts/run.sh")
        print("  2. Or ask a single question: bash scripts/run.sh 'your question'")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1
    
    print("="*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
