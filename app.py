import os
import json
import fitz
import requests
import streamlit as st
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ================== CONFIG ==================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY"
MODEL_NAME = os.getenv("MODEL_NAME") or "deepseek/deepseek-r1-0528:free"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
K_VAL = int(os.getenv("K_VAL") or 4)

# Path to repo root (where notes.pdf is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = BASE_DIR

# ================== STREAMLIT PAGE SETUP ==================
st.set_page_config(page_title="Carbon Buddy", layout="wide")
st.title("🎓 Carbon Buddy")
st.markdown("Your Friendly neighbourhood bot")
import time

def type_like_chatgpt(text, speed=0.004):
    """Types out the text character-by-character with a blinking cursor effect."""
    placeholder = st.empty()
    animated = ""
    for c in text:
        animated += c
        placeholder.markdown(animated + "|")  # add cursor
        time.sleep(speed)
    placeholder.markdown(animated)  # final text without cursor

# ================== VECTOR DB LOADING ==================
@st.cache_resource
def load_vector_db(folder: str):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            try:
                with fitz.open(os.path.join(folder, file)) as doc:
                    text = "\n".join(page.get_text() for page in doc)
                    chunks = splitter.split_text(text)
                    docs.extend([Document(page_content=c, metadata={"source": file}) for c in chunks])
            except Exception as e:
                st.warning(f"Could not read {file}: {e}")

    if not docs:
        st.warning("No PDF documents found — retrieval will return nothing.")
        class EmptyRetriever:
            def get_relevant_documents(self, q): return []
        return EmptyRetriever()

    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    return vectordb.as_retriever(search_type="similarity", k=K_VAL)

# Create retriever at startup
retriever = load_vector_db(PDF_FOLDER)

# ================== OPENROUTER HELPER ==================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}

def query_openrouter(model: str, messages: List[Dict[str, str]]) -> str:
    payload = {"model": model, "messages": messages}
    r = requests.post(OPENROUTER_URL, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"]
    return json.dumps(data)

# ================== VANILLA RAG PIPELINE ==================
def vanilla_rag_answer(question: str, lang: str = "English") -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs]) if docs else "No relevant context found."
    
    prompt = [
        {"role": "system", "content": f"You are BitsBuddy, a knowledgeable assistant for BITS Pilani. Answer in {lang}."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    return query_openrouter(MODEL_NAME, prompt)

# ================== CHAT INTERFACE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

language = st.selectbox("🌐 Response Language", ["English", "Hindi", "Telugu", "Tamil", "Marathi", "Bengali"])

if user_query := st.chat_input("Ask me about Organic Chemistry"):
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        answer = vanilla_rag_answer(user_query, lang=language)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Animate the typing
    type_like_chatgpt(answer)

# Show previous chat instantly (no animation)
for chat in st.session_state.chat_history[:-1]:
    with st.chat_message("user" if chat["role"] == "user" else "assistant"):
        st.markdown(chat["content"])
