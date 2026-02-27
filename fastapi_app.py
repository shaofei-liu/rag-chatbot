"""
RAG Chatbot - FastAPI backend with WebSocket streaming
Replaces the Streamlit interface with a professional web app
"""

from fastapi import FastAPI, WebSocket, HTTPException, Form, UploadFile, File, WebSocketDisconnect, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import uuid
from datetime import datetime

from rag_engine import (
    execute_query,
    AllModelsExhausted,
    discover_models,
    get_available_models,
    init_vector_db,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# Enable CORS with explicit configuration for HuggingFace Spaces
cors_origins = [
    "http://localhost:3000",
    "http://localhost:7860",
    "http://localhost",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:7860",
    "https://williamcass-portfolio.hf.space",
    "https://www.shaofeiliu.com",
    "https://shaofeiliu.com",
    "https://huggingface.co",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Knowledge base initialization
def init_knowledge_base():
    """Initialize knowledge base from PDFs on startup - same as Streamlit version."""
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
        logger.info("✅ Knowledge base already initialized")
        return {"status": "exists", "message": "Knowledge base loaded"}

    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"⚠️  No PDFs found in {data_dir}")
        return {"status": "no_pdfs", "message": f"No PDFs found in {data_dir}"}

    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import shutil

        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            logger.info(f"Removed existing {chroma_dir}")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"}
        )

        documents = []
        for pdf_file in pdf_files:
            logger.info(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            if docs:
                documents.extend(docs)
                logger.info(f"  Loaded {len(docs)} pages")

        if not documents:
            logger.warning("Could not load any documents from PDFs")
            return {"status": "no_docs", "message": "Could not load any documents"}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        logger.info(f"✅ Knowledge base created with {len(chunks)} chunks from {len(documents)} pages")
        return {
            "status": "created",
            "message": f"Knowledge base created with {len(chunks)} chunks from {len(documents)} pages"
        }

    except Exception as e:
        logger.error(f"❌ Knowledge base initialization failed: {str(e)}")
        return {"status": "error", "message": str(e)}


def get_model_order(model_stats: dict, available_models: list) -> list:
    """Sort models by success rate, then by success count (Streamlit compatible)."""
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


# Global state management (like st.session_state)
_available_models = None
_model_stats = {}

def init_global_models():
    """Initialize available models and stats once at startup."""
    global _available_models, _model_stats
    _available_models = get_available_models()
    _model_stats = {model: {"successes": 0, "failures": 0} for model in _available_models}
    logger.info(f"Initialized {len(_available_models)} models for tracking")
    return _available_models


# Startup event for logging
@app.on_event("startup")
async def startup_event():
    logger.info("="*50)
    logger.info("RAG CHATBOT API STARTING UP")
    logger.info("="*50)
    logger.info("✅ FastAPI server is now listening on port 7860")
    
    # Initialize knowledge base from PDFs first
    kb_result = init_knowledge_base()
    logger.info(f"Knowledge Base: {kb_result['message']}")
    
    # Then initialize vector DB from the knowledge base
    init_vector_db()
    
    # Initialize global models
    init_global_models()

# Session management
class ChatSession:
    """Manage chat session state"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[Dict] = []
        self.language = "en"
        self.model = "gemini-2.5-flash-lite"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()
    
    def get_history(self, limit: int = 20):
        """Get recent conversation history"""
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.last_activity = datetime.now()

# In-memory session storage (for production, use database)
_sessions: Dict[str, ChatSession] = {}

def get_or_create_session(session_id: str = None) -> ChatSession:
    """Get existing session or create new one"""
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(session_id)
    
    return _sessions[session_id]

# REST API Endpoints

@app.get("/ping")
async def ping():
    """Lightweight ping endpoint for diagnostics"""
    return {"pong": True, "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint - just returns API status"""
    return {
        "status": "RAG Chatbot API is running",
        "version": "2.0.0",
        "endpoints": {
            "ping": "/ping (lightweight test)",
            "health": "/health",
            "chat": "/v1/chat (POST with JSON)",
            "models": "/models",
            "sessions": "/session/create"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately without waiting for component initialization"""
    timestamp = datetime.now().isoformat()
    logger.info(f"[HEALTHCHECK] Called at {timestamp}")
    
    # Write to file so we can verify HEALTHCHECK is running
    try:
        with open("healthcheck.log", "a") as f:
            f.write(f"HEALTHCHECK called at {timestamp}\n")
    except:
        pass
    
    response = {"status": "healthy", "timestamp": timestamp}
    logger.info(f"[HEALTHCHECK] Returning: {response}")
    return response

@app.get("/models")
async def get_models():
    """Get available language models"""
    try:
        models = get_available_models()
        return {
            "models": models,
            "default": "gemini-2.5-flash-lite",
            "current_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        return {
            "models": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
            "default": "gemini-2.5-flash-lite"
        }

@app.post("/session/create")
async def create_session():
    """Create new chat session"""
    session = get_or_create_session()
    return {
        "session_id": session.session_id,
        "language": session.language,
        "model": session.model,
        "created_at": session.created_at.isoformat()
    }

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session info"""
    session = get_or_create_session(session_id)
    return {
        "session_id": session.session_id,
        "language": session.language,
        "model": session.model,
        "history_count": len(session.conversation_history),
        "created_at": session.created_at.isoformat()
    }

@app.post("/session/{session_id}/language")
async def set_language(session_id: str, language: str = Form(...)):
    """Set session language"""
    if language not in ["en", "de", "zh", "fr", "es"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    session = get_or_create_session(session_id)
    session.language = language
    
    return {"language": session.language}

@app.post("/session/{session_id}/model")
async def set_model(session_id: str, model: str = Form(...)):
    """Set session model"""
    session = get_or_create_session(session_id)
    session.model = model
    
    return {"model": session.model}

@app.get("/session/{session_id}/history")
async def get_history(session_id: str, limit: int = 20):
    """Get conversation history"""
    session = get_or_create_session(session_id)
    return {
        "session_id": session.session_id,
        "history": session.get_history(limit),
        "total_messages": len(session.conversation_history)
    }

@app.post("/session/{session_id}/clear")
async def clear_history(session_id: str):
    """Clear conversation history"""
    session = get_or_create_session(session_id)
    previous_count = len(session.conversation_history)
    session.clear_history()
    
    return {
        "session_id": session.session_id,
        "cleared_messages": previous_count,
        "message": "Conversation history cleared"
    }

# JSON-based chat endpoint (more cross-browser compatible)
@app.post("/v1/chat")
async def chat_json(request: Request):
    """
    Chat endpoint accepting JSON payload
    More compatible with cross-origin requests than FormData
    """
    request_data = await request.json()
    session_id = request_data.get("session_id")
    message = request_data.get("message")
    language = request_data.get("language", "en")
    
    if not session_id or not message or not message.strip():
        raise HTTPException(status_code=400, detail="Missing session_id or message")
    
    session = get_or_create_session(session_id)
    # Update language if provided
    if language in ["en", "de", "zh", "fr", "es"]:
        session.language = language
    session.add_message("user", message)
    
    try:
        # Get response from RAG engine
        response_text = ""
        try:
            from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
            
            # Prepare context from history
            history = session.get_history(10)
            chat_history = []
            for h in history[:-1]:  # Exclude current user message
                if h["role"] == "assistant":
                    chat_history.append(AIMessage(h["content"]))
                else:
                    chat_history.append(HumanMessage(h["content"]))
            
            # Get model order based on stats (Streamlit compatible)
            ordered_models = get_model_order(_model_stats, _available_models)
            
            # Call execute_query with full model list
            answer, model_used, attempts = execute_query(
                message,
                language=session.language,
                model_order=ordered_models,
                chat_history=chat_history,
                thread_id="default"
            )
            
            # Update model stats (Streamlit compatible)
            _model_stats[model_used]["successes"] += 1
            for model in attempts[:-1]:  # Mark failed attempts
                _model_stats[model]["failures"] += 1
            
            response_text = answer
        except AllModelsExhausted:
            response_text = "All models quota exhausted. Please try again later."
        
        session.add_message("assistant", response_text)
        
        return {
            "session_id": session.session_id,
            "user_message": message,
            "assistant_message": response_text,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    message: str = Form(...),
    language: str = Form(default="en"),
    use_streaming: bool = Form(default=True)
):
    """
    Chat endpoint - regular HTTP request (non-streaming)
    For streaming responses, use WebSocket instead
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    session = get_or_create_session(session_id)
    # Update language if provided
    if language in ["en", "de", "zh", "fr", "es"]:
        session.language = language
    session.add_message("user", message)
    
    try:
        # Get response from RAG engine
        response_text = ""
        try:
            from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
            
            # Prepare context from history
            history = session.get_history(10)
            chat_history = []
            for h in history[:-1]:  # Exclude current user message
                if h["role"] == "assistant":
                    chat_history.append(AIMessage(h["content"]))
                else:
                    chat_history.append(HumanMessage(h["content"]))
            
            # Get model order based on stats (Streamlit compatible)
            ordered_models = get_model_order(_model_stats, _available_models)
            
            # Call execute_query with full model list
            answer, model_used, attempts = execute_query(
                message,
                language=session.language,
                model_order=ordered_models,
                chat_history=chat_history,
                thread_id="default"
            )
            
            # Update model stats (Streamlit compatible)
            _model_stats[model_used]["successes"] += 1
            for model in attempts[:-1]:  # Mark failed attempts
                _model_stats[model]["failures"] += 1
            
            response_text = answer
        except AllModelsExhausted:
            response_text = "All models quota exhausted. Please try again later."
        
        session.add_message("assistant", response_text)
        
        return {
            "session_id": session.session_id,
            "user_message": message,
            "assistant_message": response_text,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# WebSocket for streaming responses
class ConnectionManager:
    """Manage WebSocket connections"""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """Connect a client"""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
    
    async def disconnect(self, session_id: str, websocket: WebSocket):
        """Disconnect a client"""
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
    
    async def broadcast(self, session_id: str, message: dict):
        """Broadcast message to all clients in session"""
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")

manager = ConnectionManager()

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat responses
    Protocol:
    - Client sends: {"type": "message", "content": "user message"}
    - Server streams: {"type": "chunk", "content": "streamed text"}
    - Server sends: {"type": "complete", "message_id": "..."}
    """
    
    await manager.connect(session_id, websocket)
    session = get_or_create_session(session_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                user_message = data.get("content", "").strip()
                
                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "content": "Message cannot be empty"
                    })
                    continue
                
                session.add_message("user", user_message)
                
                try:
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "start",
                        "user_message": user_message
                    })
                    
                    # Get chatbot response with streaming
                    # Prepare context from history
                    history = session.get_history(10)
                    
                    # Stream the response
                    response_text = ""
                    try:
                        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
                        
                        # Prepare chat history
                        chat_history = []
                        for h in history[:-1]:  # Exclude current user message
                            if h["role"] == "assistant":
                                chat_history.append(AIMessage(h["content"]))
                            else:
                                chat_history.append(HumanMessage(h["content"]))
                        
                        # Get model order based on stats (Streamlit compatible)
                        ordered_models = get_model_order(_model_stats, _available_models)
                        
                        # Call execute_query with full model list
                        answer, model_used, attempts = execute_query(
                            user_message,
                            language=session.language,
                            model_order=ordered_models,
                            chat_history=chat_history,
                            thread_id="default"
                        )
                        
                        # Update model stats (Streamlit compatible)
                        _model_stats[model_used]["successes"] += 1
                        for model in attempts[:-1]:  # Mark failed attempts
                            _model_stats[model]["failures"] += 1
                        
                        response_text = answer
                        
                        # Send response in chunks (simulate streaming)
                        chunk_size = 50
                        for i in range(0, len(response_text), chunk_size):
                            chunk = response_text[i:i+chunk_size]
                            await websocket.send_json({
                                "type": "chunk",
                                "content": chunk
                            })
                            await asyncio.sleep(0.01)  # Small delay for streaming effect
                    
                    except AllModelsExhausted:
                        response_text = "All models quota exhausted. Please try again later."
                        await websocket.send_json({
                            "type": "chunk",
                            "content": response_text
                        })
                    
                    # Send completion marker
                    session.add_message("assistant", response_text)
                    await websocket.send_json({
                        "type": "complete",
                        "total_length": len(response_text)
                    })
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error: {str(e)}"
                    })
            
            elif data.get("type") == "clear":
                # Clear conversation history
                session.clear_history()
                await websocket.send_json({
                    "type": "cleared"
                })
    
    except WebSocketDisconnect:
        await manager.disconnect(session_id, websocket)
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(session_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )
