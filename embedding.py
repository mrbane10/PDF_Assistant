import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss.contrib.torch_utils  # Import to disable GPU usage
import faiss
from config import EMBEDDING_MODEL, VECTOR_DIM, CACHE_DIR
from utils import ensure_dir_exists
import os

def generate_embeddings(chunks, cache_file=None):
    """Generate embeddings for text chunks with caching support."""
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
            # Continue with generating new embeddings
    
    # Force CPU usage
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.to('cpu')  # Force CPU
    
    texts = [chunk["content"] for chunk in chunks]
    
    # Generate embeddings in batches to conserve memory
    batch_size = 16
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    # Cache the embeddings if a cache file is specified
    if cache_file:
        cache_dir = os.path.dirname(cache_file)
        ensure_dir_exists(cache_dir)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")
    
    return embeddings

def build_index(embeddings: np.ndarray):
    """Build a FAISS index with CPU support only."""
    # Ensure we're using CPU-only FAISS
    faiss.get_num_gpus = lambda: 0
    
    # Create a flat L2 index (CPU-only)
    index = faiss.IndexFlatL2(VECTOR_DIM)
    
    # Ensure embeddings are in the right format for FAISS
    embeddings = embeddings.astype('float32')
    
    # Add vectors to the index
    index.add(embeddings)
    
    return index
