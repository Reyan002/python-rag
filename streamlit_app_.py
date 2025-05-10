# ✅ Must be first Streamlit command — very top!
import streamlit as st
st.set_page_config(page_title="PDF Q&A with Groq", layout="wide")

# Now safe to import everything else
import os
import platform
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from groq import Groq
from typing import Optional, List
import tempfile

# === Warn about Python version ===
if platform.python_version().startswith("3.13"):
    st.warning("⚠️ Python 3.13 is still experimental for many AI packages. Prefer Python 3.10 or 3.11.")

# === Groq API Key ===
GROQ_API_KEY = "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"  # Replace with your key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Custom GroqLLM class ===
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


# === Streamlit UI ===
st.title("📄 Chat with your PDF using Groq")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("🔄 Processing PDF...")

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # Initialize QA chain with Groq
    llm = GroqLLM()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    st.success("✅ Ready! Ask questions about your PDF.")

    user_question = st.text_input("❓ Ask a question about the document:")
    if user_question:
        try:
            answer = qa.run(user_question)
            st.markdown("### ✅ Answer:")
            st.info(answer)
        except Exception as e:
            st.error(f"❌ Error: {e}")
