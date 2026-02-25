"""
Quick test of web search functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import WEB_SEARCH_PROVIDER
from web_search import create_web_search_tool


def test_web_search():
    """Test web search tool creation and basic search."""
    print(f"Testing web search with provider: {WEB_SEARCH_PROVIDER}\n")
    
    try:
        # Create the tool
        tool = create_web_search_tool()
        print(f"✓ Web search tool created: {tool.metadata.name}\n")
        
        # Test a simple search
        print("Testing search query: 'immune aging interventions'\n")
        print("="*80)
        
        # Call the tool's function directly
        result = tool.fn(query="immune aging interventions", max_results=2)
        print(result)
        print("="*80)
        
        print("\n✓ Web search test successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Web search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_web_search()
    sys.exit(0 if success else 1)
