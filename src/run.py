"""
Main script to run the LlamaIndex ReAct agent for immune aging analysis.
"""

import asyncio
from pathlib import Path
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.core.workflow import Context
from agent import initialize_agent


async def interactive_mode():
    """Run the agent in interactive mode."""
    
    print("\n" + "="*80)
    print("IMMUNE AGING ANALYSIS AGENT - Interactive Mode")
    print("="*80)
    print("\nThis agent combines:")
    print("  • OMICS DATA: Factual findings from immune aging analysis")
    print("  • LITERATURE: Scientific papers for biological context")
    print("  • MANUSCRIPT: Main manuscript about immune aging")
    print("  • INTERNET: Web search for additional information")
    print("\nCapabilities:")
    print("  - Answer questions about aging signatures in different cell types")
    print("  - Retrieve intervention effects on immune cells")
    print("  - Search literature for biological interpretation")
    print("  - Provide comprehensive analysis combining data and context")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end session")
    print("  - Type 'help' to see example questions")
    print("="*80 + "\n")
    
    # Initialize agent
    agent = initialize_agent()
    
    # Create context for conversation history/session state
    ctx = Context(agent)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'help':
            print_help()
            continue
        
        try:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {user_input}")
            print(f"{'='*80}\n")
            
            # Run agent with context to maintain conversation history
            handler = agent.run(user_input, ctx=ctx)
            
            # Stream the response
            print("Agent: ", end="", flush=True)
            async for ev in handler.stream_events():
                if isinstance(ev, AgentStream):
                    print(ev.delta, end="", flush=True)
                elif isinstance(ev, ToolCallResult):
                    print(f"\n[Tool Called: {ev.tool_name}]", flush=True)
            
            # Get final response
            response = await handler
            print("\n")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try rephrasing your question or type 'help' for examples.\n")


def print_help():
    """Print example questions."""
    print("\n" + "="*80)
    print("EXAMPLE QUESTIONS")
    print("="*80)
    print("\n📊 OMICS DATA QUERIES:")
    print("  • What changes occur in CD8T cells with aging?")
    print("  • Show me transcription factors that decrease with age in monocytes")
    print("  • What are the effects of IL-2 on CD4T cells?")
    print("  • Compare aging signatures between CD8T and CD4T cells")
    
    print("\n📚 LITERATURE SEARCHES:")
    print("  • What is known about BACH2 in T cell aging?")
    print("  • Find papers about immune senescence interventions")
    print("  • What mechanisms drive inflammaging?")
    
    print("\n🔬 COMPREHENSIVE ANALYSIS:")
    print("  • What goes wrong with CD8T aging and what does the literature say?")
    print("  • Analyze NK cell aging and suggest interventions based on the data")
    print("  • What interventions tested in the manuscript restore immune function?")
    
    print("\n💡 TIP: The agent will automatically:")
    print("  - Validate cell type and intervention names")
    print("  - Choose appropriate analysis types")
    print("  - Search literature when needed for context")
    print("  - Distinguish between data, literature, and interpretation")
    print("="*80 + "\n")


async def single_query(question: str):
    """
    Run a single query without interactive mode.
    
    Args:
        question: The question to ask the agent
    """
    print(f"\nQuestion: {question}\n")
    print("="*80)
    
    # Initialize agent
    agent = initialize_agent()
    
    # Create context for the session
    ctx = Context(agent)
    
    # Run query with context
    handler = agent.run(question, ctx=ctx)
    
    # Stream response
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, ToolCallResult):
            print(f"\n[Tool: {ev.tool_name}]\n", flush=True)
    
    # Get final response
    response = await handler
    print("\n")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single query mode
        question = " ".join(sys.argv[1:])
        asyncio.run(single_query(question))
    else:
        # Interactive mode
        asyncio.run(interactive_mode())
