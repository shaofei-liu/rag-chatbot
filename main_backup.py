"""AI Chatbot - Streamlit interface with integrated KB initialization"""

import os
import shutil
from pathlib import Path
import streamlit as st
from urllib.parse import urlparse, parse_qs

# Configure Streamlit for iframe embedding
st.set_page_config(
    page_title="Shaofei's Portfolio", 
    page_icon="馃",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Detect if running in iframe
query_params = st.query_params
is_embedded = "embedded" in query_params or "iframe" in query_params


@st.cache_resource
def init_knowledge_base():
    """Initialize knowledge base from PDFs on first run."""
    chroma_dir = "chroma"
    data_dir = "data_sources"
    
    # Check if KB already exists and has content (not just empty sqlite3 file)
    def is_kb_initialized():
        if not os.path.exists(chroma_dir):
            return False
        files = os.listdir(chroma_dir)
        if not files:
            return False
        # Check if we have actual content (more than just .sqlite3 file)
        # A properly initialized chroma has metadata and data files
        has_data = any(f != 'chroma.sqlite3' for f in files)
        # Also check if sqlite3 file is larger than 16KB (empty chroma is ~16KB)
        if 'chroma.sqlite3' in files:
            db_path = os.path.join(chroma_dir, 'chroma.sqlite3')
            db_size = os.path.getsize(db_path)
            if db_size < 50000:  # Less than 50KB means probably empty
                return False
        return True
    
    if is_kb_initialized():
        return {"status": "exists", "message": "Knowledge base loaded"}
    
    # Check if PDFs exist
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        return {"status": "no_pdfs", "message": f"No PDFs found in {data_dir}"}
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import shutil
        
        # Remove incomplete chroma directory if it exists
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        
        # Setup embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Load PDFs
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            if docs:
                documents.extend(docs)
        
        if not documents:
            return {"status": "no_docs", "message": "Could not load any documents"}
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        # Create vector database
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


# Initialize app
# Note: st.set_page_config is now at the top of the file
kb_info = init_knowledge_base()

# Now import rag_engine after KB is ready
from rag_engine import execute_query, AllModelsExhausted, get_available_models

# Session state
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

# UI Setup
st.markdown("""<style>
.main { max-width: 900px; margin: 0 auto; }
h1 { text-align: center; }
/* Optimize for iframe embedding */
.block-container { padding: 1rem; }
</style>""", unsafe_allow_html=True)

# Header
st.title("💼 Shaofei's Portfolio")
st.markdown("*Ask me anything about my background, skills, and experience*")

# Show KB status only if there's an error
if kb_info["status"] == "error":
    st.error(f"❌ Knowledge base error: {kb_info['message']}")
elif kb_info["status"] == "no_pdfs":
    st.error(f"❌ {kb_info['message']}")



# Language selector
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**🌍 Language / Sprache:**")
with col2:
    st.session_state.language = st.radio(
        "Lang",
        ["en", "de"],
        format_func=lambda x: "English 🇬🇧" if x == "en" else "Deutsch 🇩🇪",
        index=0 if st.session_state.language == "en" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

st.divider()

# Welcome message with greeting
if len(st.session_state.messages) == 0:
    greetings = {
        "en": "👋 **Hello! I'm Shaofei's Virtual AI Assistant.**\n\nI'm here to tell you about Shaofei's background, skills, professional experience, and personal interests. Feel free to ask me anything—I'll do my best to help!",
        "de": "👋 **Hallo! Ich bin Shaofeis virtueller KI-Assistent.**\n\nIch bin hier, um dir über Shaofeis Hintergrund, Fähigkeiten, berufliche Erfahrung und persönliche Interessen zu berichten. Frag mich gerne alles—ich werde mein Bestes geben!"
    }
    
    with st.chat_message("assistant"):
        st.markdown(greetings[st.session_state.language])

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask me... / Frag mich..."):
    with st.chat_message("user"):
        st.markdown(query)
    
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_count += 1

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # Use container to update status as model tries different options
        status_placeholder = st.empty()
        
        try:
            ordered_models = get_model_order(st.session_state.model_stats, st.session_state.available_models)
            if not ordered_models:
                st.error("No models available.")
            else:
                # Show thinking message in user's language
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
                    st.caption(f"✓ Model: {model_used.replace('gemini-', '').replace('-', ' ').title()}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except AllModelsExhausted as e:
                    status_placeholder.empty()
                    error_msg = str(e)
                    st.error(f"❌ {error_msg}")
                except Exception as e:
                    status_placeholder.empty()
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("📋 Full Error Trace"):
                        st.code(error_details, language="python")
            
        except AllModelsExhausted as e:
            status_placeholder.empty()
            error_msgs = {
                "en": "😔 Sorry, all models are currently at their quota limit. Please try again later.",
                "de": "😔 Entschuldigung, alle Modelle haben ihre Kontingentgrenze erreicht. Bitte versuchen Sie es später erneut."
            }
            st.error(error_msgs[st.session_state.language])
            
        except Exception as e:
            status_placeholder.empty()
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ Error: {str(e)}")
            with st.expander("📋 Debug Details"):
                st.code(error_details, language="python")
