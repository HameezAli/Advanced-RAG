import os
import streamlit as st
import nest_asyncio
import getpass

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

nest_asyncio.apply()

# --- UI Section ---
st.set_page_config(page_title="RAG QA App", layout="centered")
st.title("📚 Ask Your Docs – RAG + Groq-powered Q&A")

# Ask for Groq API key (hidden input)
groq_api_key = st.text_input("🔑 Enter your GROQ API Key:", type="password")

# Proceed only if key is entered
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Vector storage setup
    persist_dir = "./vectors"
    hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-V1.5")

    # Load or create index
    if os.path.exists(os.path.join(persist_dir, "index_store.json")):
        st.info("Loading existing vector index...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context, embed_model=hf_embeddings)
    else:
        st.info("Building new index from documents...")
        docs = SimpleDirectoryReader(input_dir="./").load_data()

        for doc in docs:
            doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
            if "page_label" not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append("page_label")

        llm_transform = Groq(model="gemma2-9b-it", api_key=groq_api_key)
        text_splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)
        title_extractor = TitleExtractor(llm=llm_transform, nodes=5)
        qa_extractor = QuestionsAnsweredExtractor(llm=llm_transform, questions=3)

        pipeline = IngestionPipeline(transformations=[text_splitter, title_extractor, qa_extractor])
        nodes = pipeline.run(documents=docs, in_place=True, show_progress=False)

        index = VectorStoreIndex(nodes, embed_model=hf_embeddings)
        index.storage_context.persist(persist_dir=persist_dir)
        st.success("Index created and saved!")

    # Ask user for a question
    user_query = st.text_input("💬 Ask a question based on the documents:")

    if user_query:
        with st.spinner("Generating answer..."):
            llm_querying = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
            query_engine = index.as_query_engine(llm=llm_querying)
            response = query_engine.query(user_query)
        st.subheader("📌 Answer:")
        st.write(response.response)

else:
    st.warning("Please enter your GROQ API key to proceed.")
