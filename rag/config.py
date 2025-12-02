# Configuration file for Agentic RAG System

# ============== LLM Configuration ==============
LLM_CONFIG = {
    "api_key": "42f51433ef2e4e39bac90b4cb72a9a71",
    "url": "http://genvoy.jarvis-prod.fkcloud.in/gemini-2.5-flash/:generateContent",
    "temperature": 1,
    "max_output_tokens": 2000,
    "top_p": 1,
    "seed": 0
}

# ============== Embeddings Configuration ==============
EMBEDDINGS_CONFIG = {
    "url": "http://10.83.66.248/predict?"
}

# ============== Text Splitter Configuration ==============
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# ============== Retriever Configuration ==============
RETRIEVER_CONFIG = {
    "search_type": "similarity",
    "k": 2  # Number of documents to retrieve
}

# ============== Wikipedia Configuration ==============
WIKIPEDIA_CONFIG = {
    "lang": "en",
    "top_k_results": 2
}

# ============== arXiv Configuration ==============
ARXIV_CONFIG = {
    "top_k_results": 2
}

# ============== Prompt Template ==============
PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
"""

# ============== UI Configuration ==============
UI_CONFIG = {
    "page_title": "Agentic RAG",
    "page_icon": "🤖",
    "layout": "wide"
}

# ============== Data Source Options ==============
DATA_SOURCES = {
    "pdf": {"name": "PDF Upload", "icon": "📄", "enabled": True},
    "wikipedia": {"name": "Wikipedia", "icon": "🌐", "enabled": True},
    "flipkart": {"name": "Flipkart Webpage", "icon": "🛒", "enabled": True},
    "youtube": {"name": "YouTube Transcript", "icon": "🎥", "enabled": True},
    "arxiv": {"name": "arXiv Papers", "icon": "📚", "enabled": True}
}
