import streamlit as st
st.set_page_config(page_title="📚 PDF Chatbot", layout="wide")

import os
import platform
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from groq import Groq
from typing import Optional, List

# === Python version warning ===
if platform.python_version().startswith("3.13"):
    st.warning("⚠️ Python 3.13 is experimental. Prefer Python 3.10 or 3.11.")

# === Set Groq API Key ===
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Groq LLM Wrapper ===
class GroqLLM(LLM):
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Groq()

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stop=stop,
        )
        return response.choices[0].message.content

# === Streamlit App ===
st.title("🤖 Real-time Chatbot for Your PDF")

# Chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Load, split, and embed PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Use "cuda" for GPU
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = GroqLLM()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.success("✅ PDF processed. You can now chat below.")

    # Chat input
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        try:
            # Run LLM
            answer = qa.run(user_input)

            # Save to session state
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("bot", answer))
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display chat history
for role, message in st.session_state.get("chat_history", []):
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
