import streamlit as st
import tempfile
import os
import re
import requests
from typing import TypedDict, List, Annotated, Any
from operator import add
import logging

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    WebBaseLoader,
    YoutubeLoader
)
from langchain_community.retrievers import WikipediaRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LangGraph imports
from langgraph.graph import StateGraph, END

# Configuration
from config import (
    LLM_CONFIG, EMBEDDINGS_CONFIG, TEXT_SPLITTER_CONFIG,
    RETRIEVER_CONFIG, WIKIPEDIA_CONFIG, ARXIV_CONFIG,
    PROMPT_TEMPLATE, UI_CONFIG, DATA_SOURCES
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== LLM and Embeddings Classes ==============
class FlipkartGeminiLLM:
    """
    Flipkart Gemini LLM wrapper.
    Can be called directly: llm("prompt")
    Or used in chains via RunnableLambda wrapper
    """
    def __init__(self, api_key=None, url=None):
        self.api_key = api_key or LLM_CONFIG["api_key"]
        self.url = url or LLM_CONFIG["url"]

    def __call__(self, prompt):
        import json
        # Ensure prompt is a string
        if hasattr(prompt, "text"):
            prompt = prompt.text
        elif hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        elif isinstance(prompt, dict):
            # When used in a chain, prompt comes as dict
            prompt = prompt.get("text", str(prompt))
        
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": str(prompt)}]
                }
            ],
            "generationConfig": {
                "temperature": LLM_CONFIG["temperature"],
                "maxOutputTokens": LLM_CONFIG["max_output_tokens"],
                "topP": LLM_CONFIG["top_p"],
                "seed": LLM_CONFIG["seed"],
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            data = response.json()
            answer = data['candidates'][0]['content']['parts'][0]['text']
            return answer
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            raise Exception("Request to LLM timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise Exception(f"Failed to connect to LLM: {str(e)}")
    
    def as_runnable(self):
        """Returns a LangChain Runnable wrapper for use in chains"""
        return RunnableLambda(self.__call__)


class FlipkartEmbeddings(Embeddings):
    def __init__(self, url=None):
        self.url = url or EMBEDDINGS_CONFIG["url"]
    
    def embed_documents(self, texts):
        payload = {"input": texts}
        try:
            response = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            resp_json = response.json()
            
            data = resp_json["result"]["data"]
            embeddings = [item["embedding"] for item in data]
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.error("Embeddings request timed out")
            raise Exception("Request to embeddings service timed out. Please try again.")
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            raise ValueError(f"Error extracting embeddings: {e}")

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ============== State Definition ==============
class AgentState(TypedDict):
    query: str
    selected_sources: List[str]
    uploaded_files: List[Any]
    wikipedia_query: str
    flipkart_url: str
    youtube_url: str
    arxiv_query: str
    all_documents: Annotated[List, add]
    retrieved_context: str
    final_answer: str
    error: str
    stats: dict


# ============== Agent Nodes ==============
def collect_sources_node(state: AgentState) -> AgentState:
    """Node to collect and validate source selections"""
    logger.info(f"Sources selected: {state['selected_sources']}")
    st.info(f"✅ Selected {len(state['selected_sources'])} source(s). Starting data collection...")
    
    # Initialize stats
    state["stats"] = {
        "sources_selected": len(state['selected_sources']),
        "documents_fetched": 0,
        "chunks_created": 0,
        "processing_time": 0
    }
    return state


def fetch_pdf_node(state: AgentState) -> AgentState:
    """Fetch and process PDF files"""
    if "PDF Upload" not in state["selected_sources"]:
        return state
    
    try:
        documents = []
        uploaded_files = state.get("uploaded_files", [])
        
        if not uploaded_files:
            st.warning("⚠️ PDF Upload selected but no files provided")
            return state
            
        with st.status("📄 Processing PDF files...", expanded=True) as status:
            for idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"Loading {uploaded_file.name}...")
                suffix = uploaded_file.name.split('.')[-1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.' + suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                if suffix == "pdf":
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding='utf-8')
                
                docs = loader.load()
                for doc in docs:
                    text = re.sub(r'\s+', ' ', doc.page_content).strip()
                    doc.page_content = text
                
                documents.extend(docs)
                os.remove(tmp_path)
                st.write(f"✓ {uploaded_file.name}: {len(docs)} pages")
            
            status.update(label=f"✅ Loaded {len(documents)} pages from {len(uploaded_files)} file(s)", 
                         state="complete")
        
        state["all_documents"] = state.get("all_documents", []) + documents
        state["stats"]["documents_fetched"] += len(documents)
        
    except Exception as e:
        error_msg = f"PDF Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def fetch_wikipedia_node(state: AgentState) -> AgentState:
    """Fetch documents from Wikipedia"""
    if "Wikipedia" not in state["selected_sources"]:
        return state
    
    try:
        query = state.get("wikipedia_query") or state.get("query")
        
        with st.status("🌐 Fetching from Wikipedia...", expanded=True) as status:
            st.write(f"Searching for: {query}")
            retriever = WikipediaRetriever(
                lang=WIKIPEDIA_CONFIG["lang"], 
                top_k_results=WIKIPEDIA_CONFIG["top_k_results"]
            )
            docs = retriever.invoke(query)
            
            for doc in docs:
                text = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.page_content = text
            
            status.update(label=f"✅ Fetched {len(docs)} Wikipedia article(s)", 
                         state="complete")
        
        state["all_documents"] = state.get("all_documents", []) + docs
        state["stats"]["documents_fetched"] += len(docs)
        
    except Exception as e:
        error_msg = f"Wikipedia Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def fetch_flipkart_node(state: AgentState) -> AgentState:
    """Fetch documents from Flipkart webpage"""
    if "Flipkart Webpage" not in state["selected_sources"]:
        return state
    
    try:
        url = state.get("flipkart_url", "").strip()
        if not url:
            st.warning("⚠️ Flipkart Webpage selected but no URL provided")
            return state
        
        with st.status("🛒 Fetching Flipkart webpage...", expanded=True) as status:
            st.write(f"Loading: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            for doc in docs:
                text = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.page_content = text
            
            status.update(label=f"✅ Fetched Flipkart webpage content", 
                         state="complete")
        
        state["all_documents"] = state.get("all_documents", []) + docs
        state["stats"]["documents_fetched"] += len(docs)
        
    except Exception as e:
        error_msg = f"Flipkart Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def fetch_youtube_node(state: AgentState) -> AgentState:
    """Fetch transcript from YouTube video"""
    if "YouTube Transcript" not in state["selected_sources"]:
        return state
    
    try:
        url = state.get("youtube_url", "").strip()
        if not url:
            st.warning("⚠️ YouTube Transcript selected but no URL provided")
            return state
        
        with st.status("🎥 Fetching YouTube transcript...", expanded=True) as status:
            st.write(f"Loading transcript from: {url}")
            loader = YoutubeLoader.from_youtube_url(url, language="en")
            docs = loader.load()
            
            for doc in docs:
                text = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.page_content = text
            
            status.update(label=f"✅ Fetched YouTube transcript", 
                         state="complete")
        
        state["all_documents"] = state.get("all_documents", []) + docs
        state["stats"]["documents_fetched"] += len(docs)
        
    except Exception as e:
        error_msg = f"YouTube Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def fetch_arxiv_node(state: AgentState) -> AgentState:
    """Fetch papers from arXiv"""
    if "arXiv Papers" not in state["selected_sources"]:
        return state
    
    try:
        from langchain_community.retrievers import ArxivRetriever
        
        query = state.get("arxiv_query") or state.get("query")
        
        with st.status("📚 Fetching arXiv papers...", expanded=True) as status:
            st.write(f"Searching for: {query}")
            retriever = ArxivRetriever(top_k_results=ARXIV_CONFIG["top_k_results"])
            docs = retriever.invoke(query)
            
            for doc in docs:
                text = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.page_content = text
            
            status.update(label=f"✅ Fetched {len(docs)} arXiv paper(s)", 
                         state="complete")
        
        state["all_documents"] = state.get("all_documents", []) + docs
        state["stats"]["documents_fetched"] += len(docs)
        
    except Exception as e:
        error_msg = f"arXiv Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def create_vector_store_node(state: AgentState) -> AgentState:
    """Create vector store and retrieve relevant context"""
    documents = state.get("all_documents", [])
    
    if not documents:
        error_msg = "No documents were fetched from any source"
        state["error"] = error_msg
        st.error(f"❌ {error_msg}")
        return state
    
    try:
        with st.status("🔍 Creating vector store and retrieving context...", expanded=True) as status:
            # Combine all text
            st.write("Combining documents...")
            all_text = " ".join([doc.page_content for doc in documents])
            
            # Split into chunks
            st.write("Splitting into chunks...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"]
            )
            chunks = splitter.create_documents([all_text])
            state["stats"]["chunks_created"] = len(chunks)
            st.write(f"Created {len(chunks)} text chunks")
            
            # Create embeddings and vector store
            st.write("Creating embeddings...")
            embeddings = FlipkartEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # Create retriever
            st.write("Setting up retriever...")
            retriever = vector_store.as_retriever(
                search_type=RETRIEVER_CONFIG["search_type"],
                search_kwargs={"k": RETRIEVER_CONFIG["k"]}
            )
            
            # Retrieve relevant context
            st.write("Retrieving relevant context...")
            query = state["query"]
            retrieved_docs = retriever.invoke(query)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            state["retrieved_context"] = context_text
            status.update(
                label=f"✅ Vector store created ({len(chunks)} chunks, {len(retrieved_docs)} retrieved)",
                state="complete"
            )
        
    except Exception as e:
        error_msg = f"Vector Store Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["error"] = state.get("error", "") + error_msg + "; "
    
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    """Generate final answer using LLM"""
    if state.get("error"):
        state["final_answer"] = f"❌ Error occurred during processing: {state['error']}"
        return state
    
    try:
        with st.status("🤖 Generating answer...", expanded=True) as status:
            context = state.get("retrieved_context", "")
            query = state["query"]
            
            st.write("Preparing prompt...")
            prompt = PROMPT_TEMPLATE.format(context=context, question=query)
            
            # Initialize LLM and call it directly (not in a chain)
            st.write("Calling LLM...")
            llm = FlipkartGeminiLLM()
            
            # Direct call - this works!
            answer = llm(prompt)
            state["final_answer"] = answer
            
            status.update(label="✅ Answer generated successfully", state="complete")
        
    except Exception as e:
        error_msg = f"LLM Error: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        state["final_answer"] = f"Error generating answer: {str(e)}"
    
    return state


# ============== Build LangGraph ==============
def create_agent_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("collect_sources", collect_sources_node)
    workflow.add_node("fetch_pdf", fetch_pdf_node)
    workflow.add_node("fetch_wikipedia", fetch_wikipedia_node)
    workflow.add_node("fetch_flipkart", fetch_flipkart_node)
    workflow.add_node("fetch_youtube", fetch_youtube_node)
    workflow.add_node("fetch_arxiv", fetch_arxiv_node)
    workflow.add_node("create_vector_store", create_vector_store_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Define edges
    workflow.set_entry_point("collect_sources")
    workflow.add_edge("collect_sources", "fetch_pdf")
    workflow.add_edge("fetch_pdf", "fetch_wikipedia")
    workflow.add_edge("fetch_wikipedia", "fetch_flipkart")
    workflow.add_edge("fetch_flipkart", "fetch_youtube")
    workflow.add_edge("fetch_youtube", "fetch_arxiv")
    workflow.add_edge("fetch_arxiv", "create_vector_store")
    workflow.add_edge("create_vector_store", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()


# ============== Streamlit UI ==============
def main():
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout=UI_CONFIG["layout"]
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">🤖 Agentic RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Query multiple data sources simultaneously with AI-powered retrieval</div>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent_graph' not in st.session_state:
        st.session_state.agent_graph = create_agent_graph()
    
    # Sidebar for source selection
    with st.sidebar:
        st.header("📚 Data Sources")
        st.markdown("**Select one or more sources:**")
        st.markdown("---")
        
        sources = []
        uploaded_files = None
        wikipedia_query = ""
        flipkart_url = ""
        youtube_url = ""
        arxiv_query = ""
        
        # PDF Upload
        if DATA_SOURCES["pdf"]["enabled"]:
            if st.checkbox(f"{DATA_SOURCES['pdf']['icon']} {DATA_SOURCES['pdf']['name']}", key="pdf_check"):
                sources.append(DATA_SOURCES['pdf']['name'])
                uploaded_files = st.file_uploader(
                    "Upload files", 
                    type=["pdf", "txt"], 
                    accept_multiple_files=True,
                    key="pdf_uploader"
                )
        
        # Wikipedia
        if DATA_SOURCES["wikipedia"]["enabled"]:
            if st.checkbox(f"{DATA_SOURCES['wikipedia']['icon']} {DATA_SOURCES['wikipedia']['name']}", key="wiki_check"):
                sources.append(DATA_SOURCES['wikipedia']['name'])
                wikipedia_query = st.text_input(
                    "Search query (optional)", 
                    placeholder="Leave empty to use main query",
                    key="wiki_query"
                )
        
        # Flipkart
        if DATA_SOURCES["flipkart"]["enabled"]:
            if st.checkbox(f"{DATA_SOURCES['flipkart']['icon']} {DATA_SOURCES['flipkart']['name']}", key="flipkart_check"):
                sources.append(DATA_SOURCES['flipkart']['name'])
                flipkart_url = st.text_input(
                    "Product URL", 
                    placeholder="https://www.flipkart.com/...",
                    key="flipkart_url"
                )
        
        # YouTube
        if DATA_SOURCES["youtube"]["enabled"]:
            if st.checkbox(f"{DATA_SOURCES['youtube']['icon']} {DATA_SOURCES['youtube']['name']}", key="youtube_check"):
                sources.append(DATA_SOURCES['youtube']['name'])
                youtube_url = st.text_input(
                    "Video URL", 
                    placeholder="https://www.youtube.com/watch?v=...",
                    key="youtube_url"
                )
        
        # arXiv
        if DATA_SOURCES["arxiv"]["enabled"]:
            if st.checkbox(f"{DATA_SOURCES['arxiv']['icon']} {DATA_SOURCES['arxiv']['name']}", key="arxiv_check"):
                sources.append(DATA_SOURCES['arxiv']['name'])
                arxiv_query = st.text_input(
                    "Search query (optional)", 
                    placeholder="Leave empty to use main query",
                    key="arxiv_query"
                )
        
        st.markdown("---")
        st.markdown("**📊 Configuration**")
        st.caption(f"Max tokens: {LLM_CONFIG['max_output_tokens']}")
        st.caption(f"Chunk size: {TEXT_SPLITTER_CONFIG['chunk_size']}")
        st.caption(f"Retrieved chunks: {RETRIEVER_CONFIG['k']}")
    
    # Main query input
    st.markdown("### 🔍 Your Question")
    query = st.text_area(
        "What would you like to know?",
        placeholder="Enter your question here...",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Search and Generate Answer", type="primary", use_container_width=True)
    
    # Process button
    if search_button:
        if not query:
            st.error("❌ Please enter a question!")
        elif not sources:
            st.error("❌ Please select at least one data source!")
        else:
            # Prepare initial state
            initial_state = {
                "query": query,
                "selected_sources": sources,
                "uploaded_files": uploaded_files if uploaded_files else [],
                "wikipedia_query": wikipedia_query,
                "flipkart_url": flipkart_url,
                "youtube_url": youtube_url,
                "arxiv_query": arxiv_query,
                "all_documents": [],
                "retrieved_context": "",
                "final_answer": "",
                "error": "",
                "stats": {}
            }
            
            # Run the agent
            st.markdown("---")
            st.markdown("### 📊 Agent Progress")
            
            final_state = st.session_state.agent_graph.invoke(initial_state)
            
            # Display results
            st.markdown("---")
            st.markdown("### 💡 Answer")
            
            if final_state.get("final_answer"):
                st.markdown(f"""
                <div class="answer-box">
                    <p style="font-size: 16px; line-height: 1.8; margin: 0;">{final_state['final_answer']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ No answer generated. Please check the error messages above.")
            
            # Show statistics
            with st.expander("📈 Detailed Statistics", expanded=False):
                stats = final_state.get("stats", {})
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sources Used", stats.get("sources_selected", 0))
                with col2:
                    st.metric("Documents Fetched", stats.get("documents_fetched", 0))
                with col3:
                    st.metric("Chunks Created", stats.get("chunks_created", 0))
                
                st.markdown("**Query Details:**")
                st.text(f"Query: {final_state['query']}")
                st.text(f"Sources: {', '.join(final_state['selected_sources'])}")
                
                if final_state.get("retrieved_context"):
                    st.markdown("**Retrieved Context:**")
                    st.text_area(
                        "Context used for answer generation",
                        final_state['retrieved_context'],
                        height=200
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>Powered by LangGraph 🦜⛓️</strong> | Built with Streamlit 🎈</p>
        <p style="font-size: 0.9em;">Version 1.1 (Fixed) | Internal Use Only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()