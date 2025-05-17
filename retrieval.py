from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from config import EMBEDDING_MODEL

def query_index(query_text, index, chunks, top_k=5):
    """Query the FAISS index to find the most relevant chunks for a given query."""
    # Initialize model on CPU
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.to('cpu')  # Force CPU usage
    
    # Generate query embedding
    query_embedding = model.encode([query_text]).astype('float32')
    
    # Search the index
    try:
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks) and idx >= 0:  # Ensure index is valid
                chunk = chunks[idx]
                section_info = chunk.get("section", "") or "N/A"
                
                results.append({
                    "content": chunk["content"],
                    "page": chunk["page"],
                    "section": section_info,
                    "score": float(distances[0][i]),
                    "source": f"PDF Page {chunk['page']}" + (f", Section: {section_info}" if section_info != "N/A" else "")
                })
        
        # Sort by score (lower distance is better)
        results.sort(key=lambda x: x["score"])
        
        return results
    except Exception as e:
        print(f"Error searching index: {e}")
        return []

def format_context_from_results(results):
    """Format the retrieved chunks into a context string for the LLM."""
    if not results:
        return "No relevant information found in the document."
    
    context = "CONTEXT FROM PDF DOCUMENT:\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"[EXCERPT {i} - Page {result['page']}"
        if result.get('section') and result['section'] != "N/A":
            context += f", Section: {result['section']}"
        
        # Format the similarity score
        score = 1.0 - min(1.0, result['score'] / 100.0)  # Convert distance to similarity
        context += f", Relevance: {score:.2f}]\n"
        
        # Add the content
        context += f"{result['content']}\n\n"
    
    context += "END OF CONTEXT\n\n"
    return context
