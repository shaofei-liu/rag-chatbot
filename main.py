"""AI Chatbot - Streamlit interface with integrated KB initialization and professional styling"""

import os
import shutil
from pathlib import Path
import streamlit as st
from urllib.parse import urlparse, parse_qs

# Configure Streamlit for iframe embedding
st.set_page_config(
    page_title="Shaofei's RAG Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Detect if running in iframe
query_params = st.query_params
is_embedded = "embedded" in query_params or "iframe" in query_params

# Add JavaScript to prevent Streamlit Cloud app from sleeping
# This creates a hidden heartbeat every 5 minutes
st.markdown("""
<script>
    // Keep the app alive by sending periodic pings
    setInterval(() => {
        fetch(window.location.href, {method: 'HEAD'}).catch(() => {});
    }, 300000); // 5 minutes
</script>
""", unsafe_allow_html=True)

# Custom CSS for professional styling
st.markdown("""<style>
    /* Main container and layout */
    .main {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Header and title styling */
    h1 {
        color: #0066cc;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2rem;
    }
    
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    /* Block container padding */
    .block-container {
        padding: 1.5rem 1.5rem;
        max-width: 900px;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
    }
    
    [data-testid="stChatMessage"] [data-testid="chatMessageContent"] {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Assistant message background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f0f4ff;
        border-left: 4px solid #0066cc;
    }
    
    /* User message background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f8f9fa;
    }
    
    /* Input field styling */
    .stChatInputContainer {
        padding: 0 1rem 1rem 1rem;
    }
    
    .stChatInputContainer input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stChatInputContainer input:focus {
        border: 2px solid #0066cc;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        outline: none;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s ease;
        font-size: 14px;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0052a3;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
        transform: translateY(-2px);
    }
    
    /* Radio button styling */
    .stRadio {
        display: flex;
        gap: 2rem;
        justify-content: center;
    }
    
    .stRadio > label {
        padding: 0.5rem 1rem;
        color: #333;
        font-weight: 500;
        cursor: pointer;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stRadio > label:hover {
        background-color: #f0f4ff;
    }
    
    /* Columns spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #0066cc, transparent);
        margin: 1.2rem 0;
        opacity: 0.4;
    }
    
    /* Caption and info text */
    .stCaption {
        color: #999;
        font-size: 0.85rem;
        text-align: center;
    }
    
    /* Error styling */
    .stException {
        background-color: #fee;
        border-left: 4px solid #ff4444;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f0f4ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Success message */
    .success-message {
        color: #2ecc71;
        font-weight: 500;
    }
    
    /* Loading spinner color */
    .stSpinner {
        color: #0066cc;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
    
    [data-testid="stExpander"] > button {
        background-color: #f8f9fa;
        color: #0066cc;
    }
    
    [data-testid="stExpander"] > button:hover {
        background-color: #f0f4ff;
    }
</style>""", unsafe_allow_html=True)


@st.cache_resource
def init_knowledge_base():
    """Initialize knowledge base from PDFs on first run."""
    chroma_dir = "chroma"
    data_dir = "data_sources"

    def is_kb_initialized():
        if not os.path.exists(chroma_dir):
            return False
        files = os.listdir(chroma_dir)
        if not files:
            return False
        has_data = any(f != 'chroma.sqlite3' for f in files)
        if 'chroma.sqlite3' in files:
            db_path = os.path.join(chroma_dir, 'chroma.sqlite3')
            db_size = os.path.getsize(db_path)
            if db_size < 50000:
                return False
        return True

    if is_kb_initialized():
        return {"status": "exists", "message": "Knowledge base loaded"}

    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        return {"status": "no_pdfs", "message": f"No PDFs found in {data_dir}"}

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import shutil

        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )

        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            if docs:
                documents.extend(docs)

        if not documents:
            return {"status": "no_docs", "message": "Could not load any documents"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir
        )

        return {
            "status": "created",
            "message": f"Knowledge base created with {len(chunks)} chunks from {len(documents)} pages"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_model_order(model_stats: dict, available_models: list) -> list:
    """Sort models by success rate, then by success count."""
    models_with_scores = []
    for model in available_models:
        stats = model_stats.get(model, {"successes": 0, "failures": 0})
        successes = stats["successes"]
        failures = stats["failures"]
        total = successes + failures
        success_rate = (successes / total) if total > 0 else 1.0
        models_with_scores.append({
            "model": model,
            "success_rate": success_rate,
            "successes": successes,
        })
    models_with_scores.sort(key=lambda x: (-x["success_rate"], -x["successes"]))
    return [m["model"] for m in models_with_scores]


# Initialize knowledge base
kb_info = init_knowledge_base()

# Import RAG engine
from rag_engine import execute_query, AllModelsExhausted, get_available_models

# Session state initialization
if "language" not in st.session_state:
    st.session_state.language = "en"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "available_models" not in st.session_state:
    st.session_state.available_models = get_available_models()
if "model_stats" not in st.session_state:
    st.session_state.model_stats = {
        model: {"successes": 0, "failures": 0}
        for model in st.session_state.available_models
    }
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


# Header
st.markdown("<h1>💼 Shaofei's Portfolio</h1>", unsafe_allow_html=True)

st.markdown('<p class="subtitle">AI-Powered Retrieval-Augmented Generation Chatbot</p>', unsafe_allow_html=True)

st.divider()

# Language selector
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("**🌍 Language / Sprache:**")
with col2:
    st.session_state.language = st.radio(
        "Lang",
        ["en", "de"],
        format_func=lambda x: "🇬🇧 English" if x == "en" else "🇩🇪 Deutsch",
        index=0 if st.session_state.language == "en" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

st.divider()

st.divider()

# Welcome message
if len(st.session_state.messages) == 0:
    greetings = {
        "en": "👋 **Hello! I'm Shaofei's AI Assistant.**\n\nI'm powered by advanced AI models and a Retrieval-Augmented Generation system. I can answer questions about Shaofei's professional background, technical skills, education, and experience. Feel free to ask me anything!\n\n*Example: What did you study? What sports do you like?*",
        "de": "👋 **Hallo! Ich bin Shaofeis KI-Assistent.**\n\nIch werde von fortschrittlichen KI-Modellen und einem Retrieval-Augmented Generation-System betrieben. Ich kann Fragen zu Shaofeis beruflichem Hintergrund, technischen Fähigkeiten, Bildung und Erfahrung beantworten. Frag mich gerne alles!\n\n*Beispiel: Was hast Du studiert? Welche Sportarten magst Du?*"
    }

    with st.chat_message("assistant"):
        st.markdown(greetings[st.session_state.language])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask me... / Frag mich..."):
    # Set processing flag immediately to disable language switching
    st.session_state.is_processing = True
    
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_count += 1

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        status_placeholder = st.empty()

        try:
            
            ordered_models = get_model_order(st.session_state.model_stats, st.session_state.available_models)
            if not ordered_models:
                st.error("No models available.")
            else:
                thinking_message = "🤔 Thinking..." if st.session_state.language == "en" else "🤔 Ich denke nach..."
                status_placeholder.info(thinking_message)

                try:
                    response, model_used, model_attempts = execute_query(
                        query, st.session_state.language, ordered_models, st.session_state.chat_history
                    )
                    st.session_state.model_stats[model_used]["successes"] += 1
                    for model in model_attempts[:-1]:
                        st.session_state.model_stats[model]["failures"] += 1
                    status_placeholder.empty()

                    st.markdown(response)
                    st.caption(f"✨ Model: {model_used.replace('gemini-', '').replace('-', ' ').title()}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except AllModelsExhausted as e:
                    # Clear loading status on error
                    status_placeholder.empty()
                    error_msg = str(e)
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    # Restore language toggle on error
                    st.session_state.is_processing = False
                    status_placeholder.empty()
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("🔍 Full Error Trace"):
                        st.code(error_details, language="python")

        except AllModelsExhausted as e:
            status_placeholder.empty()
            error_msgs = {
                "en": "😞 Sorry, all models are currently at their quota limit. Please try again later.",
                "de": "😞 Entschuldigung, alle Modelle haben ihre Kontingentgrenze erreicht. Bitte versuchen Sie es später erneut."
            }
            st.error(error_msgs[st.session_state.language])

        except Exception as e:
            status_placeholder.empty()
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Debug Details"):
                st.code(error_details, language="python")
        
        finally:
            # Clear processing flag when done - enables language switching again
            st.session_state.is_processing = False
