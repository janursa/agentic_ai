"""
Build and manage RAG indices for literature and manuscript.
"""

import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage


def build_literature_index(data_dir: Path, persist_dir: Path = None):
    """
    Build vector index from all documents in the literature folder.
    
    Args:
        data_dir: Path to data directory containing literature folder
        persist_dir: Optional path to persist the index (for faster loading)
    
    Returns:
        VectorStoreIndex: The built index
    """
    literature_path = data_dir / "literature"
    
    if not literature_path.exists():
        raise ValueError(f"Literature folder not found at {literature_path}")
    
    # Check if we have a persisted index
    if persist_dir and persist_dir.exists():
        print(f"Loading existing index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
        return index
    
    print(f"Building new index from {literature_path}")
    
    # Load all documents from literature folder (supports PDF, txt, md, etc.)
    documents = SimpleDirectoryReader(
        str(literature_path),
        recursive=True
    ).load_data()
    
    print(f"Loaded {len(documents)} documents")
    
    # Build index
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    
    # Persist if directory provided
    if persist_dir:
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"Index persisted to {persist_dir}")
    
    return index


def build_manuscript_index(data_dir: Path, persist_dir: Path = None):
    """
    Build vector index from the manuscript file.
    
    Args:
        data_dir: Path to data directory containing manuscrtipt.txt
        persist_dir: Optional path to persist the index
    
    Returns:
        VectorStoreIndex: The built index
    """
    manuscript_path = data_dir / "manuscrtipt.txt"
    
    if not manuscript_path.exists():
        raise ValueError(f"Manuscript not found at {manuscript_path}")
    
    # Check if we have a persisted index
    if persist_dir and persist_dir.exists():
        print(f"Loading existing manuscript index from {persist_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
        index = load_index_from_storage(storage_context)
        return index
    
    print(f"Building manuscript index from {manuscript_path}")
    
    # Load manuscript
    documents = SimpleDirectoryReader(
        input_files=[str(manuscript_path)]
    ).load_data()
    
    # Build index
    index = VectorStoreIndex.from_documents(documents)
    
    # Persist if directory provided
    if persist_dir:
        persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"Manuscript index persisted to {persist_dir}")
    
    return index


def get_or_build_indices(data_dir: Path, use_cache: bool = True):
    """
    Get or build both literature and manuscript indices.
    
    Args:
        data_dir: Path to data directory
        use_cache: Whether to use cached indices if available
    
    Returns:
        tuple: (literature_index, manuscript_index)
    """
    persist_lit_dir = data_dir.parent / ".index_cache" / "literature" if use_cache else None
    persist_man_dir = data_dir.parent / ".index_cache" / "manuscript" if use_cache else None
    
    literature_index = build_literature_index(data_dir, persist_lit_dir)
    manuscript_index = build_manuscript_index(data_dir, persist_man_dir)
    
    return literature_index, manuscript_index
