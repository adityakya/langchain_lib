import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
import requests
import re
import tempfile
import os

# Flipkart LLM and Embeddings
class FlipkartGeminiLLM:
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def __call__(self, prompt):
        import json
        # Ensure prompt is a string
        if hasattr(prompt, "text"):
            prompt = prompt.text
        elif hasattr(prompt, "to_string"):
            prompt = prompt.to_string()
        headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': self.api_key
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1,
                "maxOutputTokens": 500,
                "topP": 1,
                "seed": 0,
                "thinkingConfig": {"thinkingBudget": 0}
            }
        }
        response = requests.post(self.url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        answer = data['candidates'][0]['content']['parts'][0]['text']
        return answer

class FlipkartEmbeddings(Embeddings):
    def embed_documents(self, texts):
        payload = {"input": texts}
        response = requests.post(
            "http://10.83.66.248/predict?",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        resp_json = response.json()
        try:
            data = resp_json["result"]["data"]
            embeddings = [item["embedding"] for item in data]
        except Exception as e:
            print("Unexpected response format:", resp_json)
            raise ValueError(f"Error extracting embeddings: {e}")
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Streamlit UI
st.title("Agentic RAG App")

uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
query = st.text_input("Enter your query")

if uploaded_files and query:
    all_text = ""
    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        if suffix == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            text = re.sub(r'\s+', ' ', doc.page_content).strip()
            all_text += text + " "
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([all_text])

    embeddings = FlipkartEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables = ['context', 'question']
    )

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()
    api_key = "6a650d18ef4d443583a4f5136d5c1e1f"
    url = 'http://genvoy.jarvis-prod.fkcloud.in/gemini-2.5-flash/:generateContent'
    llm = FlipkartGeminiLLM(api_key, url)

    main_chain = parallel_chain | prompt | llm | parser

    with st.spinner("Searching and generating answer..."):
        result = main_chain.invoke(query)
        st.markdown("### Answer")
        st.write(result)