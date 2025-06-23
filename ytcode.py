import nest_asyncio
import pprint
import os
import getpass
import chromadb
from llama_index.core import SimpleDirectoryReader,Document,VectorStoreIndex,StorageContext,load_index_from_storage
from llama_index.core.schema import MetadataMode
from llama_index.core.extractors import TitleExtractor,QuestionsAnsweredExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
nest_asyncio.apply()

# Data Extraction Code
docs=SimpleDirectoryReader(input_dir="./").load_data() # Load documents 
print(len(docs))
# pprint.pprint(docs) # to view the json metadata in pretty print

# Data Transformation Code
for doc in docs:
    doc.text_template="Metadata:\n{metadata_str}\n---\nContent:\n{content}"
    if "page_label" not in doc.excluded_embed_metadata_keys: # Adding page label to excluded metadata
        doc.excluded_embed_metadata_keys.append("page_label")
#print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))

# More Transformations using LLMs
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
llm_transformations = Groq(model="gemma2-9b-it", api_key=os.environ["GROQ_API_KEY"])

text_splitter = SentenceSplitter( separator=" ", chunk_size=1024, chunk_overlap=128 )
title_extractor = TitleExtractor(llm=llm_transformations, nodes=5)
qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformations, questions=3)
pipeline = IngestionPipeline( transformations=[ text_splitter, title_extractor, qa_extractor ] )
nodes = pipeline.run( documents=docs, in_place=True, show_progress=True, )
print("The processed nodes: "+str(len(nodes)))

# Data Embeddings  and Indexing Code
hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-V1.5")
index = VectorStoreIndex(nodes,embed_model=hf_embeddings)

# Querying the Model
llm_querying = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
query_engine = index.as_query_engine(llm=llm_querying)

# Persistant Storage
index.storage_context.persist(persist_dir="./vectors")
storage_context = StorageContext.from_defaults(persist_dir="./vectors")
index_from_storage = load_index_from_storage(storage_context, embed_model=hf_embeddings)
qa = index_from_storage.as_query_engine(llm=llm_querying)
response = qa.query("what is the significance of forward propagation?")
print(response)

