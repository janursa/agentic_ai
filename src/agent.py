"""
LlamaIndex ReAct Agent for Immune Aging Analysis.
Combines omics data tools with RAG-powered literature search.
"""

import os
from pathlib import Path
from typing import Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# Import omics helper functions
from helper import (
    get_aging_signature,
    get_intervention_signature,
    get_disease_signature,
    get_available_cell_types,
    get_available_interventions,
    get_available_analysis_types,
    get_available_features
)
from rag_builder import get_or_build_indices


def create_omics_tools():
    """
    Wrap all omics analysis functions as LlamaIndex FunctionTools.
    
    Returns:
        list: List of FunctionTool objects for omics data access
    """
    tools = []
    
    # Wrap each helper function as a LlamaIndex tool
    tools.append(FunctionTool.from_defaults(
        fn=get_available_cell_types,
        name="get_available_cell_types",
        description="""Get list of all available cell types for analysis.
Use this to validate cell type names before querying data."""
    ))
    
    tools.append(FunctionTool.from_defaults(
        fn=get_available_interventions,
        name="get_available_interventions",
        description="""Get list of all available interventions (drugs and cytokines).
Use this to validate intervention names before querying data."""
    ))
    
    tools.append(FunctionTool.from_defaults(
        fn=get_available_analysis_types,
        name="get_available_analysis_types",
        description="""Get all available analysis types with descriptions.
Use this to understand what data is available and select appropriate analysis types based on the user's question."""
    ))
    
    tools.append(FunctionTool.from_defaults(
        fn=get_available_features,
        name="get_available_features",
        description="""Get all available feature types with descriptions.
Use this to understand what biological features can be analyzed."""
    ))
    
    # For functions with parameters, create direct wrappers
    tools.append(FunctionTool.from_defaults(
        fn=get_aging_signature,
        name="get_aging_signature",
        description="""Retrieve aging-associated features for a cell type from a specific analysis.

Args:
    cell_type: Cell type name (e.g., 'CD8T', 'MONO'). Use get_available_cell_types() to see valid names.
    analysis_name: Analysis type to use (e.g., 'tfa_major_b', 'ge_major_b'). Use get_available_analysis_types() to see all options."""
    ))
    
    tools.append(FunctionTool.from_defaults(
        fn=get_intervention_signature,
        name="get_intervention_signature",
        description="""Retrieve intervention-associated features for a cell type.

Args:
    cell_type: Cell type name. Use get_available_cell_types() to validate.
    intervention: Intervention name. Use get_available_interventions() to validate.
    analysis_name: Analysis type to use. Use get_available_analysis_types() to see options."""
    ))
    
    tools.append(FunctionTool.from_defaults(
        fn=get_disease_signature,
        name="get_disease_signature",
        description="""Retrieve disease-associated features for a cell type.

Args:
    cell_type: Cell type name. Use get_available_cell_types() to validate.
    disease: Disease name (e.g., 'SLE').
    analysis_name: Analysis type to use. Use get_available_analysis_types() to see options."""
    ))
    
    return tools


def create_rag_tools(literature_index: VectorStoreIndex, manuscript_index: VectorStoreIndex):
    """
    Create RAG query tools for literature and manuscript search.
    
    Args:
        literature_index: Vector index for scientific literature
        manuscript_index: Vector index for the main manuscript
    
    Returns:
        list: List of QueryEngineTool objects for RAG search
    """
    tools = []
    
    # Literature search tool
    literature_engine = literature_index.as_query_engine(similarity_top_k=3)
    tools.append(QueryEngineTool.from_defaults(
        query_engine=literature_engine,
        name="search_literature",
        description="""Search through indexed scientific papers for biological context and interpretation.
Use this to find relevant information about genes, pathways, mechanisms, or biological processes from the literature.

Example queries:
- "BACH2 role in CD8T cell aging"
- "transcription factors in immune senescence"
- "interventions that restore immune function in aging"
"""
    ))
    
    # Manuscript search tool
    manuscript_engine = manuscript_index.as_query_engine(similarity_top_k=3)
    tools.append(QueryEngineTool.from_defaults(
        query_engine=manuscript_engine,
        name="search_manuscript",
        description="""Search the main manuscript document for information about the immune aging analysis.
Use this to find specific findings, methods, or conclusions from the manuscript.

Example queries:
- "What interventions were tested in the manuscript?"
- "Main findings about CD8T aging"
- "Methods used for transcription factor analysis"
"""
    ))
    
    return tools


def create_agent(
    literature_index: VectorStoreIndex,
    manuscript_index: VectorStoreIndex,
    model_id: str = "gpt-4o",
    verbose: bool = True
):
    """
    Create the main ReAct agent with all tools.
    
    Args:
        literature_index: Vector index for literature
        manuscript_index: Vector index for manuscript
        model_id: OpenAI model to use
        verbose: Whether to show detailed logging
    
    Returns:
        ReActAgent: The configured agent
    """
    # Load environment variables (for OPENAI_API_KEY)
    load_dotenv()
    
    # Initialize LLM
    llm = OpenAI(model=model_id)
    
    # Create all tools
    omics_tools = create_omics_tools()
    rag_tools = create_rag_tools(literature_index, manuscript_index)
    all_tools = omics_tools + rag_tools
    
    # Load system instructions
    instructions_path = Path(__file__).parent / "instructions.txt"
    with open(instructions_path, 'r') as f:
        system_prompt = f.read()
    
    # Create ReAct agent
    agent = ReActAgent(
        llm=llm,
        tools=all_tools,
        system_prompt=system_prompt,
        verbose=verbose
    )
    
    return agent


def initialize_agent(data_dir: Optional[Path] = None, use_cache: bool = True, model_id: str = "gpt-4o"):
    """
    Initialize the complete agent with RAG indices.
    
    Args:
        data_dir: Path to data directory (default: ../data from this file)
        use_cache: Whether to use cached indices
        model_id: OpenAI model to use
    
    Returns:
        ReActAgent: Ready-to-use agent
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    
    # Build/load indices
    print("Initializing RAG indices...")
    literature_index, manuscript_index = get_or_build_indices(data_dir, use_cache=use_cache)
    
    # Create agent
    print("Creating agent...")
    agent = create_agent(literature_index, manuscript_index, model_id=model_id)
    
    print("Agent ready!")
    return agent
