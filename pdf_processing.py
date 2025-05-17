import pymupdf
import re
import os
from typing import Dict, List
from utils import safe_filename

def extract_section_info(text: str) -> str:
    """Extract section information from text."""
    patterns = [
        r'(?:Chapter|CHAPTER)\s+(\d+|[IVX]+)(?:\s*[:\.]\s*(.+))?',
        r'(?:Section|SECTION)\s+(\d+\.\d+(?:\.\d+)*)(?:\s*[:\.]\s*(.+))?',
        r'^\s*(\d+\.\d+(?:\.\d+)*)\s+(.+)',
        r'^\s*(\d+)\s+(.+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text[:200])
        if match:
            if match.group(2):
                return f"{match.group(1)}: {match.group(2)}"
            else:
                return match.group(1)
    return ""

def parse_pdf(file_path: str) -> Dict:
    """Parse PDF content with better error handling for cloud environments."""
    try:
        doc = pymupdf.open(file_path)
        text_content = {}
        metadata = {
            "title": doc.metadata.get("title", "Untitled"),
            "author": doc.metadata.get("author", "Unknown"),
            "pages": len(doc)
        }
        
        # Process each page
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text("text")
                section = extract_section_info(text)
                text_content[page_num + 1] = {
                    "text": text,
                    "section": section
                }
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {e}")
                # Add empty placeholder for failed pages
                text_content[page_num + 1] = {
                    "text": f"[Error processing page {page_num + 1}]",
                    "section": ""
                }
        
        doc.close()
        return {"metadata": metadata, "text_content": text_content}
    
    except Exception as e:
        # Provide a fallback for complete failure
        print(f"Error parsing PDF: {e}")
        return {
            "metadata": {"title": "Error", "author": "Unknown", "pages": 0},
            "text_content": {1: {"text": f"Failed to process PDF: {str(e)}", "section": ""}}
        }

def chunk_text(text_content: Dict, chunk_size=900, overlap=40) -> List[Dict]:
    """Split text into manageable chunks with optional section information."""
    chunks = []
    
    for page_num, page_data in text_content.items():
        text = page_data["text"]
        section = page_data.get("section", "")
        
        # Handle empty pages
        if not text.strip():
            continue
            
        # Split into sentences (improved regex for better sentence detection)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Count words as a proxy for tokens
            token_count = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + token_count > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "content": chunk_text,
                    "page": page_num,
                    "section": section
                })
                
                # Keep some sentences for overlap
                overlap_count = min(overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
            
            # Add the current sentence
            current_chunk.append(sentence)
            current_length += token_count
        
        # Don't forget to add the last chunk from the page
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "content": chunk_text,
                "page": page_num,
                "section": section
            })
    
    # Make sure we have at least one chunk
    if not chunks:
        chunks.append({
            "content": "No processable text found in document.",
            "page": 1,
            "section": ""
        })
    
    return chunks
