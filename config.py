import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Vector embedding dimension for sentence-transformers/all-mpnet-base-v2
VECTOR_DIM = 768

# Text chunking parameters
CHUNK_SIZE = 900
OVERLAP = 40

# Embedding model - using a smaller model that's efficient on CPU
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Cache directory - using a cloud-friendly path
# For cloud deployment, we use a relative path that will be created in the app directory
CACHE_DIR = "./cache"

# Supported models on Groq
MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen-qwq-32b",
]

# Enhanced system prompt that includes formatting instructions directly
SYSTEM_PROMPT = """
You are an AI assistant skilled in analyzing uploaded PDF documents and answering user queries with clarity and depth.

When responding to questions about the PDF:

1. Prioritize information directly from the PDF, citing page numbers or sections wherever relevant.
2. Integrate your broader knowledge to elaborate, contextualize, or clarify the material—especially when the document is sparse, ambiguous, or silent on the topic.
3. Clearly distinguish between information drawn from the PDF and external knowledge used to supplement the answer.
4. Maintain continuity across multiple questions, using both the document and prior conversation for context.
5. Reformat equations, data, and notations to improve readability, while preserving their meaning.
6. If a topic is not covered in the PDF, say so, and then provide a helpful, well-informed answer based on general knowledge.
7. If you're unsure or if the information is inconclusive, state your uncertainty rather than speculating.

Be precise, helpful, and clear.  
Avoid hallucinations, overgeneralizations, or invented details.
"""

# Query rewriting prompt to handle follow-up questions
QUERY_REWRITING_PROMPT = """
You are an analyzer who determines if a query is a follow-up to the previous conversation.

Context:
User Message: {user_message}
System Response: {system_response}
New Query: {query}

Instructions:
1. Analyze if the new query directly references any the previous conversation without explicitly naming its subject
2. Only rewrite the query if it contains pronouns (it, this, these, etc.) that depend on previous context
3. If the query is understandable on its own or introduces a new named topic, leave it unchanged
4. For queries like "What about X?" or "How about Y?", only rewrite if X or Y weren't explicitly named
5. When in doubt, return the original query unchanged

Output only the rewritten query or the original query with no additional text.
"""

# Get API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# For cloud deployment, we need to handle the case where the API key is provided differently
if not GROQ_API_KEY:
    # Check if it's available from a different source (like Streamlit Secrets)
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except:
        # Handle the case where no API key is available
        print("WARNING: No GROQ API key found. Please set the GROQ_API_KEY environment variable.")
        GROQ_API_KEY = "YOUR_API_KEY"  # This will cause the app to prompt for an API key

# Cloud-specific configs - memory limits
MAX_CACHE_SIZE_MB = 500  # Adjust based on your cloud provider's limitations
