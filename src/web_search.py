"""
Web search tools for the agent.
Supports both Tavily and DuckDuckGo search providers.
"""

import os
from pathlib import Path
from typing import Optional
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv

# Load environment variables from .env file in project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from config import WEB_SEARCH_PROVIDER


def create_tavily_search_tool() -> FunctionTool:
    """
    Create a Tavily search tool.
    Requires TAVILY_API_KEY environment variable.
    
    Returns:
        FunctionTool: Tavily web search tool
    """
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "Tavily not installed. Install with: pip install tavily-python"
        )
    
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in environment. "
            "Get your API key from https://tavily.com and set it in your environment or .env file."
        )
    
    client = TavilyClient(api_key=api_key)
    
    def search_web_tavily(query: str, max_results: int = 3) -> str:
        """
        Search the web using Tavily AI-optimized search.
        Returns curated, relevant information formatted for AI consumption.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 3)
        
        Returns:
            Formatted search results with titles, URLs, and content
        """
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced"  # More thorough search
            )
            
            if not response.get("results"):
                return f"No results found for query: {query}"
            
            formatted_results = []
            for i, result in enumerate(response["results"], 1):
                formatted_results.append(
                    f"[{i}] {result.get('title', 'No title')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {result.get('content', 'No content available')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing Tavily search: {str(e)}"
    
    return FunctionTool.from_defaults(
        fn=search_web_tavily,
        name="search_web",
        description="""Search the internet for current information, recent publications, or additional context not available in the indexed literature.

Use this when:
- User asks about recent developments or current events
- Need information about ongoing clinical trials
- Looking for latest research published after the literature indexing date
- Seeking general biological knowledge not covered in the literature

Args:
    query: The search query (be specific and include relevant keywords)
    max_results: Number of results to return (default: 3)

Returns:
    Formatted search results with titles, URLs, and relevant content excerpts."""
    )


def create_duckduckgo_search_tool() -> FunctionTool:
    """
    Create a DuckDuckGo search tool (free, no API key required).
    
    Returns:
        FunctionTool: DuckDuckGo web search tool
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError(
            "DuckDuckGo search not installed. Install with: pip install duckduckgo-search"
        )
    
    def search_web_ddg(query: str, max_results: int = 3) -> str:
        """
        Search the web using DuckDuckGo (free, privacy-focused).
        Returns relevant web results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return (default: 3)
        
        Returns:
            Formatted search results with titles, URLs, and snippets
        """
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            
            if not results:
                return f"No results found for query: {query}"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] {result.get('title', 'No title')}\n"
                    f"URL: {result.get('href', 'N/A')}\n"
                    f"Snippet: {result.get('body', 'No snippet available')}\n"
                )
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing DuckDuckGo search: {str(e)}"
    
    return FunctionTool.from_defaults(
        fn=search_web_ddg,
        name="search_web",
        description="""Search the internet for current information, recent publications, or additional context not available in the indexed literature.

Use this when:
- User asks about recent developments or current events
- Need information about ongoing clinical trials
- Looking for latest research published after the literature indexing date
- Seeking general biological knowledge not covered in the literature

Args:
    query: The search query (be specific and include relevant keywords)
    max_results: Number of results to return (default: 3)

Returns:
    Formatted search results with titles, URLs, and relevant content snippets."""
    )


def create_web_search_tool(provider: Optional[str] = None) -> FunctionTool:
    """
    Create a web search tool based on the configured provider.
    
    Args:
        provider: Search provider ("tavily" or "duckduckgo"). 
                 If None, uses WEB_SEARCH_PROVIDER from config.
    
    Returns:
        FunctionTool: Web search tool for the specified provider
    
    Raises:
        ValueError: If provider is invalid
    """
    if provider is None:
        provider = WEB_SEARCH_PROVIDER
    
    provider = provider.lower()
    
    if provider == "tavily":
        print("Using Tavily search (AI-optimized)")
        return create_tavily_search_tool()
    elif provider == "duckduckgo":
        print("Using DuckDuckGo search (free, privacy-focused)")
        return create_duckduckgo_search_tool()
    else:
        raise ValueError(
            f"Invalid web search provider: {provider}. "
            f"Must be 'tavily' or 'duckduckgo'"
        )
