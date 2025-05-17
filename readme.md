# PDF Assistant

This application provides an interactive chat interface for asking questions about PDF documents. It uses RAG (Retrieval-Augmented Generation) to provide context-aware answers based on the content of uploaded PDFs.

## Features

- PDF upload and processing
- Context-aware question answering
- Multiple chat sessions
- Query rewriting for better conversational context
- Document sectioning and metadata extraction
- Cloud-optimized storage and processing

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API key:

Create a `.env` file in the root directory with your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

## Running the Application

```bash
streamlit run app.py
```

## Cloud Deployment

### Streamlit Cloud

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Add your API key as a secret in the Streamlit Cloud dashboard:
   - Name: `GROQ_API_KEY`
   - Value: `your_api_key_here`

### Other Cloud Providers

For deployment on other cloud platforms, make sure to:
1. Set the `GROQ_API_KEY` environment variable
2. Ensure the cache directory (`./cache`) is writable
3. Note that this application is optimized for CPU-only environments

## Architecture

- `app.py`: Main Streamlit application
- `config.py`: Configuration settings
- `utils.py`: Utility functions
- `pdf_processing.py`: PDF parsing and text chunking
- `embedding.py`: Text embedding generation (CPU-optimized)
- `retrieval.py`: Semantic search functionality
- `chat_utils.py`: Query rewriting for conversation context

## Limitations

- The application is optimized for CPU-only environments
- Large PDFs (>100MB) may be slow to process
- The cache is limited to prevent excessive storage usage
