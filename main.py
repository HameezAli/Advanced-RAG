import os
import re
import tempfile
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv  # Load from .env file

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Automatically works in Streamlit Cloud via secrets.toml
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Set it in .env or deployment secrets.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
nest_asyncio.apply()

st.set_page_config(page_title="RAG PDF QA App", layout="centered")
st.title("📄 Ask Your PDF – RAG + Groq")

# Upload the PDF
uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save file temporarily
        temp_path = os.path.join(tmpdir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Persistent vector dir
        persist_dir = f"./vectors/{uploaded_file.name.replace('.pdf', '')}"
        hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        if os.path.exists(os.path.join(persist_dir, "index_store.json")):
            st.info("📦 Loading cached vector index...")
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context, embed_model=hf_embeddings)
        else:
            st.info("⚙️ Building new index from uploaded PDF...")
            docs = SimpleDirectoryReader(input_files=[temp_path]).load_data()

            # Optional metadata cleanup
            for doc in docs:
                doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
                if "page_label" not in doc.excluded_embed_metadata_keys:
                    doc.excluded_embed_metadata_keys.append("page_label")

            # Processing pipeline
            llm_transform = Groq(model="gemma2-9b-it", api_key=GROQ_API_KEY)
            text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
            title_extractor = TitleExtractor(llm=llm_transform, nodes=5)
            qa_extractor = QuestionsAnsweredExtractor(llm=llm_transform, questions=3)

            pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])
            nodes = pipeline.run(documents=docs, in_place=True, show_progress=True)

            index = VectorStoreIndex(nodes, embed_model=hf_embeddings)
            index.storage_context.persist(persist_dir=persist_dir)
            st.success("✅ Index created and stored!")

        # Ask a question
        user_query = st.text_input("💬 Ask something about your PDF:")
        if user_query:
            with st.spinner("Thinking..."):
                llm_querying = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
                query_engine = index.as_query_engine(llm=llm_querying)
                response = query_engine.query(user_query)

            # Separate <think> block if present
            raw_response = response.response
            think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
            think_block = think_match.group(1).strip() if think_match else "No internal reasoning found."
            cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

            st.subheader("📌 Answer")
            st.write(cleaned_response)

            st.subheader("🧠 Internal Reasoning")
            st.text(f"<think>\n{think_block}\n</think>")

else:
    st.warning("Please upload a PDF to continue.")
