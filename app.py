import streamlit as st
import os
import tempfile
from groq import Groq
from config import *
from utils import ensure_dir_exists, generate_session_id
from pdf_processing import parse_pdf, chunk_text
from embedding import generate_embeddings, build_index
from retrieval import query_index, format_context_from_results
from chat_utils import rewrite_query

st.set_page_config(page_title="PDF Assistant", layout="wide")
st.title('PDF Assistant')

# Initialize session state variables
if 'groq_client' not in st.session_state:
    st.session_state['groq_client'] = Groq(api_key=GROQ_API_KEY)

if 'sessions' not in st.session_state:
    st.session_state['sessions'] = {"Default Session": {"id": generate_session_id(), "messages": []}}
    st.session_state['current_session'] = "Default Session"
    st.session_state['session_pdf_mapping'] = {}

if 'pdf_indices' not in st.session_state:
    st.session_state['pdf_indices'] = {}

# Ensure cache directory exists
ensure_dir_exists(CACHE_DIR)

# Sidebar for session management
st.sidebar.title('Sessions')

if st.sidebar.button("+ New Session"):
    new_session_name = f"Session {len(st.session_state['sessions']) + 1}"
    st.session_state['sessions'][new_session_name] = {"id": generate_session_id(), "messages": []}
    st.session_state['current_session'] = new_session_name
    st.session_state['session_pdf_mapping'][new_session_name] = None
    st.rerun()

current_session_name = st.session_state['current_session']
current_session_data = st.session_state['sessions'][current_session_name]

col1, col2 = st.sidebar.columns([1, 1])
with col1:
    if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
        st.session_state['sessions'][current_session_name]["messages"] = []
        st.rerun()
with col2:
    if st.button("📄 Reset PDF", key="reset_pdf", use_container_width=True):
        st.session_state['session_pdf_mapping'][current_session_name] = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Your Sessions")
for session_name in list(st.session_state['sessions'].keys()):
    button_label = f"📌 {session_name}" if session_name == current_session_name else f"💬 {session_name}"
    if st.sidebar.button(button_label, key=f"session_{session_name}", use_container_width=True):
        st.session_state['current_session'] = session_name
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.title('PDF Processing')

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Create a temp file in a way that's cloud-friendly
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name
    
    if st.sidebar.button("Process PDF"):
        with st.sidebar.status("Processing PDF..."):
            pdf_name = uploaded_file.name.replace('.pdf', '')
            safe_pdf_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in pdf_name)
            
            st.sidebar.text("Extracting text...")
            parsed_data = parse_pdf(pdf_path)
            
            st.sidebar.text("Creating chunks...")
            chunks = chunk_text(parsed_data["text_content"], CHUNK_SIZE, OVERLAP)
            
            st.sidebar.text("Generating embeddings...")
            embeddings_cache = os.path.join(CACHE_DIR, f"{safe_pdf_name}_embeddings.pkl")
            embeddings = generate_embeddings(chunks, cache_file=embeddings_cache)
            
            st.sidebar.text("Building search index...")
            index = build_index(embeddings)
            
            st.session_state['pdf_indices'][safe_pdf_name] = {
                "index": index,
                "chunks": chunks,
                "metadata": parsed_data["metadata"]
            }
            st.session_state['session_pdf_mapping'][current_session_name] = safe_pdf_name
            st.sidebar.success(f"PDF processed: {pdf_name}")
    
    # Clean up temp file
    try:
        os.unlink(pdf_path)
    except Exception as e:
        st.error(f"Error removing temporary file: {e}")

st.sidebar.markdown("---")
st.sidebar.title('Model Settings')
selected_model = st.sidebar.selectbox("Select Model", options=MODELS, index=0)

# PDF selection dropdown (if any PDFs are processed)
if st.session_state['pdf_indices']:
    current_session_pdf = st.session_state['session_pdf_mapping'].get(current_session_name)
    pdf_options = list(st.session_state['pdf_indices'].keys())
    default_index = pdf_options.index(current_session_pdf) if current_session_pdf in pdf_options else 0
    selected_pdf = st.sidebar.selectbox("Select PDF for context", options=pdf_options, index=default_index)
    st.session_state['session_pdf_mapping'][current_session_name] = selected_pdf
else:
    selected_pdf = None

use_rag = st.sidebar.checkbox("Use PDF context for responses", value=True)
use_query_rewriting = st.sidebar.checkbox("Enable query rewriting", value=True)
show_debug_info = st.sidebar.checkbox("Show debug information", value=False)  # Changed to False by default for production
top_k = st.sidebar.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=5)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
max_tokens = st.sidebar.slider('Max Tokens', min_value=1, max_value=32768, value=1024)

st.subheader(f"Current Session: {current_session_name}")

# Display current PDF info if available
current_session_pdf = st.session_state['session_pdf_mapping'].get(current_session_name)
if current_session_pdf and current_session_pdf in st.session_state['pdf_indices']:
    pdf_info = st.session_state['pdf_indices'][current_session_pdf]['metadata']
    st.info(f"📄 PDF: {current_session_pdf} | Pages: {pdf_info['pages']} | Author: {pdf_info['author']}")
elif selected_pdf and selected_pdf in st.session_state['pdf_indices']:
    pdf_info = st.session_state['pdf_indices'][selected_pdf]['metadata']
    st.info(f"📄 PDF: {selected_pdf} | Pages: {pdf_info['pages']} | Author: {pdf_info['author']}")

# Display chat history
current_messages = current_session_data["messages"]
for message in current_messages:
    role = message.get('role', '')
    content = message.get('content', '')
    if role.lower() == 'system':
        continue
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask a question about the PDF or chat..."):
    current_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            client = st.session_state['groq_client']
            conversation_messages = []
            
            # Get active PDF
            active_pdf = st.session_state['session_pdf_mapping'].get(current_session_name)
            if not active_pdf and selected_pdf:
                active_pdf = selected_pdf
            
            enhanced_system_prompt = SYSTEM_PROMPT
            updated_query = prompt
            
            # Query rewriting (if enabled)
            if use_query_rewriting and len(current_messages) > 1:
                try:
                    with st.status("Rewriting query..."):
                        updated_query = rewrite_query(prompt, current_messages, client)
                    
                    if show_debug_info and updated_query != prompt:
                        st.info(f"Original query: '{prompt}'\nRewritten query: '{updated_query}'")
                except Exception as e:
                    st.warning(f"Query rewriting failed: {e}")
                    updated_query = prompt  # Fallback to original prompt
            
            # Context retrieval (if enabled)
            new_context = None
            if use_rag and active_pdf and active_pdf in st.session_state['pdf_indices']:
                pdf_data = st.session_state['pdf_indices'][active_pdf]
                try:
                    with st.status("Retrieving context..."):
                        results = query_index(updated_query, pdf_data['index'], pdf_data['chunks'], top_k=top_k)
                    
                    if results:
                        new_context = format_context_from_results(results)
                    else:
                        new_context = "No relevant information was found in the PDF for this query."
                    
                
                    context_to_use = new_context
                    
                    if show_debug_info and context_to_use:
                        with st.expander("📄 View Retrieved Context"):
                            st.markdown(f"```markdown\n{context_to_use}\n```")
                    
                    enhanced_system_prompt = f"""
                        {SYSTEM_PROMPT}

                        ---
                        ### 📘 Retrieved Context
                        Below is the relevant context extracted from the uploaded document. Use it as your primary source when answering:

                        {context_to_use}

                         ---
                        ### 🧠 Further Instructions for Answering:
                        --Format any mathematical expressions, equations, or symbols clearly using LaTeX (enclose with `$...$` for inline or `$$...$$` for block equations).
                        --Integrate External Knowledge: Feel free to incorporate your broader knowledge when the PDF lacks sufficient detail or is ambiguous. Provide rich explanations and examples.
                        --Organized and Detailed Responses: Write concise, well-structured responses that break down complex ideas into easy-to-understand steps. Ensure that explanations are thorough, tailored to the user's needs, and balance technical depth with readability.
                    """
                except Exception as e:
                    st.error(f"Error retrieving context: {e}")
                    enhanced_system_prompt = f"""
                        {SYSTEM_PROMPT}

                        ---
                        An error occurred while retrieving information from the PDF. 
                        Please respond using your general knowledge where appropriate, and indicate that the response is not sourced from the document.
                    """
            else:
                enhanced_system_prompt = f"""
                    {SYSTEM_PROMPT}

                    ---
                    No PDF context is being used for this query.
                    Please respond using your general knowledge where appropriate.
                """
            
            # Build conversation history
            conversation_messages.append({
                "role": "system",
                "content": enhanced_system_prompt
            })
            
            history_messages = []
            for message in current_messages:
                if message['role'].lower() != 'system':
                    history_messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                    
            # Limit history to last 5 messages to prevent token overflow
            history_messages = history_messages[-5:] if len(history_messages) > 5 else history_messages
            conversation_messages.extend(history_messages)
            
            # Update the latest query if it was rewritten
            if updated_query != prompt and conversation_messages[-1]["content"] == prompt:
                conversation_messages[-1]["content"] = updated_query
                if show_debug_info:
                    conversation_messages.insert(-1, {
                        "role": "system",
                        "content": f"Note: The user's query has been rewritten from '{prompt}' to '{updated_query}' to better capture the context of the conversation."
                    })
            
            # Generate response
            try:
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=conversation_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                )
                
                response_placeholder = st.empty()
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        full_response += chunk_content
                        response_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                full_response = f"I'm sorry, but I encountered an error while generating a response. Please try again or adjust your query. Technical details: {str(e)}"
                response_placeholder.markdown(full_response)
    
    # Save message to history
    current_messages.append({"role": "assistant", "content": full_response})
    st.session_state['sessions'][current_session_name]["messages"] = current_messages
