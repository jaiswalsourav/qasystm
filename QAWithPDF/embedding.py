from pydoc import doc
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    try:
        logging.info("Initializing Gemini Embedding model")
        
        # FIX 1: Use 'model_name' and explicitly set 'output_dimensionality' 
        # to 768 to maintain compatibility with most vector stores.
        gemini_embed_model = GoogleGenAIEmbedding(
            model_name="models/gemini-embedding-001",
            output_dimensionality=768 
        )
        
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20
        
        logging.info("Vector store index is being created")
        
        # IMPORTANT: If you still get an error, MANUALLY DELETE the './storage' folder 
        # in your project directory before running this again.
        index = VectorStoreIndex.from_documents(document, insert_batch_size=10)
        index.storage_context.persist()
        
        logging.info("Index persisted successfully.")
        return index.as_query_engine()
        
    except Exception as e:
        raise customexception(e, sys)