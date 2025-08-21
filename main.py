"""
Retrieval Augmented Generation (RAG) with LlamaIndex
Includes: 
- Environment setup
- Index creation/loading
- Custom retriever & query engine
- Persistence (storage)
"""

import os
from dotenv import load_dotenv

# ---- LlamaIndex (0.10+) imports ----
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response

# Optional: set default LLM/embeddings explicitly (recommended for 0.10+)
# Uncomment if you face provider/config issues.
# from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding


# -----------------------------
# 1. Environment Setup
# -----------------------------
def setup_environment():
    """Load API keys and environment variables."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    os.environ["OPENAI_API_KEY"] = api_key
    print(" Environment loaded successfully")

    # Optional: set providers programmatically (instead of relying on defaults)
    # Settings.llm = OpenAI(model="gpt-4o-mini")  # or "gpt-4o", "gpt-3.5-turbo", etc.
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# -----------------------------
# 2. Index Handling
# -----------------------------
def get_or_create_index(data_dir="data", persist_dir="./storage"):
    """
    Load existing index if present, else create new one.
    """
    if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
        print(" No existing storage found. Creating new index...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
        print(" Index created and persisted.")
    else:
        print(" Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print(" Index loaded successfully.")
    return index


# -----------------------------
# 3. Query Engines
# -----------------------------
def build_query_engine(index, top_k=4, similarity_cutoff=0.8):
    """
    Build a custom query engine with retriever and postprocessor.
    """
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    postprocessor = SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[postprocessor],
    )
    return query_engine


# -----------------------------
# 4. Run Queries
# -----------------------------
def run_query(query_engine, question, show_source=False):
    """
    Run query against the engine and return response.
    """
    response = query_engine.query(question)
    pprint_response(response, show_source=show_source)
    return response


# -----------------------------
# 5. Main Logic
# -----------------------------
if __name__ == "__main__":
    setup_environment()

    # Create or load index
    index = get_or_create_index(data_dir="data", persist_dir="./storage")

    # Build custom query engine
    query_engine = build_query_engine(index)

    # Example queries
    print("\n Query 1:")
    response1 = run_query(
        query_engine,
        "What is 'Attention is All You Need'?",
        show_source=True,
    )
    print(response1)

    print("\n Query 2:")
    query_engine_basic = index.as_query_engine()
    response2 = run_query(query_engine_basic, "What are transformers?")
    print(response2)
