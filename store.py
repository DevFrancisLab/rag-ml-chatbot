from src.helper import *
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
try:
    load_dotenv()
except:
    # Handle case where load_dotenv fails in certain contexts
    pass

# Initialize the vectorstore lazily
_docs = None

def get_docs():
    global _docs
    if _docs is None:
        try:
            print("Loading and embedding documents (this may take a while)...")
            file = file_loader(r"Data/")
            chunk_data = chunking_data(file)
            embedding = get_embedding()
            _docs = FAISS.from_documents(documents=chunk_data, embedding=embedding)
            print("Vectorstore loaded successfully!")
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            _docs = None
    return _docs

# Create a lazy-loaded object
class LazyDocs:
    def __getattr__(self, name):
        docs = get_docs()
        if docs is None:
            raise RuntimeError("Vectorstore failed to initialize")
        return getattr(docs, name)

docs = LazyDocs()

