"""RAG Engine - Retrieval-Augmented Generation with multilingual support"""

import os
from typing import Sequence, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*google.generativeai.*")
import google.generativeai as genai
from model_monitor import quota_manager
from google_pricing_sync import get_free_tier_models

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=api_key)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# Lazy initialization - will be set after knowledge base is ready
vector_db = None
retriever = None

def init_vector_db():
    """Initialize vector database (call after knowledge base is ready)."""
    global vector_db, retriever
    print("[RAG] Initializing vector_db from chroma...")
    try:
        vector_db = Chroma(persist_directory="chroma", embedding_function=embeddings)
        retriever = vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        print(f"[RAG] ✓ Vector DB initialized with {vector_db._collection.count()} documents")
        return True
    except Exception as e:
        print(f"[RAG] ✗ Failed to initialize vector DB: {e}")
        return False

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5, api_key=api_key)

DEFAULT_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
EXCLUDE_PATTERNS = ["gemma", "tts", "audio", "image", "robotics", "deep-research", "computer-use", "vision", "embedding"]
_CACHED_FREE_TIER_MODELS = None

def _get_cached_free_tier():
    """Load free tier models once at startup."""
    global _CACHED_FREE_TIER_MODELS
    if _CACHED_FREE_TIER_MODELS is None:
        _CACHED_FREE_TIER_MODELS = get_free_tier_models()
    return _CACHED_FREE_TIER_MODELS

def is_model_suitable(model_name: str) -> bool:
    """Check if model is suitable for text chatbot."""
    model_lower = model_name.lower()
    if any(p in model_lower for p in EXCLUDE_PATTERNS):
        return False
    if "2.1" in model_name or "1." in model_name:
        return False
    if "latest" in model_lower or "exp-" in model_lower:
        return False
    free_tier = _get_cached_free_tier()
    is_free = any(model_lower.startswith(fm.lower()) for fm in free_tier)
    if "preview" in model_lower:
        return is_free
    if is_free:
        return True
    if "flash" in model_lower and ("2.5" in model_name or "2.0" in model_name):
        return True
    return False

def discover_models():
    """Discover available models from Google API with filtering."""
    try:
        free_tier = get_free_tier_models()
        all_api = [m.name.replace('models/', '') for m in genai.list_models()]
        available = [m for m in all_api if is_model_suitable(m) or m in free_tier]
        return sorted(available, reverse=True) if available else DEFAULT_MODELS
    except:
        return DEFAULT_MODELS

def get_available_models():
    """Get list of TEXT GENERATION models suitable for chat."""
    try:
        all_models = [m.name.replace('models/', '') for m in genai.list_models()]
        suitable = [m for m in all_models if is_model_suitable(m)]
        return suitable if suitable else DEFAULT_MODELS
    except:
        return DEFAULT_MODELS


class AllModelsExhausted(Exception):
    """Raised when all models have exhausted their quota."""
    pass

PROMPTS = {
    "en": {
        "contextualize": "Given chat history and a user question, formulate a standalone question. Do NOT answer, just reformulate if needed.",
        "answer": """You are Shaofei Liu. You MUST respond ONLY in ENGLISH.

CRITICAL RULES (MUST FOLLOW):
1. LANGUAGE: Your answer MUST be in ENGLISH ONLY. This is mandatory regardless of the question language.
2. PERSON: Always speak in FIRST PERSON as Shaofei Liu: "I am Shaofei...", "My experience...", "I like..."
3. CONTENT: Use information from the Context provided. If Context has relevant information, use it even if partial.
4. RESPONSE: Keep responses natural and conversational (2-5 sentences)
5. UNKNOWN: Only say "I don't have information about that" if absolutely no relevant context exists.

Context: {context}
Question: {input}

Your answer in ENGLISH (first person, as Shaofei):"""
    },
    "de": {
        "contextualize": "Angesichts eines Chatverlaufs und einer Benutzerfrage, formulieren Sie eine eigenständige Frage. Beantworten Sie NICHT, reformulieren Sie nur.",
        "answer": """Du bist Shaofei Liu. Du MUSST nur auf DEUTSCH antworten.

KRITISCHE REGELN (MÜSSEN BEFOLGT WERDEN):
1. SPRACHE: Deine Antwort MUSS NUR AUF DEUTSCH sein. Dies ist obligatorisch, unabhängig von der Sprache der Frage.
2. PERSON: Sprich immer in der ERSTEN PERSON als Shaofei Liu: "Ich bin Shaofei...", "Meine Erfahrung...", "Ich mag..."
3. INHALT: Nutze Informationen aus dem bereitgestellten Kontext. Wenn der Kontext relevante Informationen enthält, nutze sie auch wenn nur teilweise.
4. ANTWORT: Halte Antworten natürlich und gesprächig (2-5 Sätze)
5. UNBEKANNT: Sag nur "Dazu habe ich keine Informationen" wenn absolut kein relevanter Kontext existiert.

Kontext: {context}
Frage: {input}

Deine Antwort auf DEUTSCH (erste Person, als Shaofei):"""
    }
}

class RAGState(TypedDict):
    """Application state for conversational workflow."""
    input: str
    language: str
    chat_history: Sequence[BaseMessage]
    context: str
    answer: str
    model: Optional[str]
    _model_order: Optional[list]


def build_rag_chain(language: str = 'en', llm_obj=None):
    """Create RAG chain for the given language."""
    if not llm_obj:
        llm_obj = llm
    
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPTS[language]["contextualize"]),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm_obj, retriever, contextualize_prompt)
    
    answer_prompt = ChatPromptTemplate.from_template(PROMPTS[language]["answer"])
    answer_chain = create_stuff_documents_chain(llm_obj, answer_prompt)
    
    return create_retrieval_chain(history_aware_retriever, answer_chain)


def invoke_rag_pipeline(state: RAGState):
    """Execute RAG with automatic model switching."""
    language = state.get("language", "en")
    models = state.get("_model_order", get_available_models())
    
    print(f"[RAG] Pipeline: {len(models)} models, language={language}")
    
    model_attempts = []
    for idx, model_name in enumerate(models, 1):
        model_attempts.append(model_name)
        status = quota_manager.check_quota_status(model_name)
        
        # Skip if we already know this model had errors
        if status["quota_exhausted_detected"]:
            print(f"[RAG] {idx}/{len(models)}: {model_name} [SKIPPED - previous error]")
            continue
        
        print(f"[RAG] {idx}/{len(models)}: Trying {model_name}...")
        try:
            llm_to_use = ChatGoogleGenerativeAI(model=model_name, temperature=0.5, api_key=api_key)
            rag_chain = build_rag_chain(language, llm_to_use)
            
            response = rag_chain.invoke({
                "input": state["input"],
                "chat_history": state["chat_history"]
            })
            
            print(f"[RAG]   ✓ {model_name} succeeded")
            quota_manager.record_request(model_name, success=True)
            return {
                "chat_history": [
                    HumanMessage(state["input"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
                "model": model_name,
                "_model_attempts": model_attempts,
            }
        except Exception as e:
            error_text = str(e)
            print(f"[RAG]   ✗ {model_name} failed: {error_text[:80]}")
            quota_manager.record_request(model_name, success=False)
            
            # Mark as exhausted only if API explicitly says so
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                quota_manager.mark_quota_exhausted(model_name)
                print(f"[RAG]      (API returned 429, skipping this model for rest of session)")
            
            # Try next model
            continue
    
    print(f"[RAG] All {len(models)} models failed")
    raise AllModelsExhausted("All models failed or returned errors")


workflow = StateGraph(state_schema=RAGState)
workflow.add_edge(START, "rag_node")
workflow.add_node("rag_node", invoke_rag_pipeline)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def execute_query(query_text: str, language: str = "en", model_order: list = None, chat_history: list = None, thread_id: str = "default") -> tuple:
    """Execute a query with language-specific processing.
    
    Returns: (response_text, model_name, model_attempts)
    """
    from langchain_core.messages import BaseMessage
    language = "de" if language.startswith('de') else "en"
    if chat_history is None:
        chat_history = []
    state_to_invoke = {
        "input": query_text,
        "language": language,
        "chat_history": chat_history if isinstance(chat_history, list) and (not chat_history or isinstance(chat_history[0], BaseMessage)) else [],
        "_model_order": model_order if model_order else get_available_models(),
    }
    try:
        result = app.invoke(state_to_invoke, config={"configurable": {"thread_id": thread_id}})
        model_attempts = result.get("_model_attempts", [result.get("model", "unknown")])
        return (result["answer"], result.get("model", "unknown"), model_attempts)
    except AllModelsExhausted:
        raise
    except Exception as e:
        import traceback
        print(f"RAG Error: {str(e)}")
        print(traceback.format_exc())
        raise

