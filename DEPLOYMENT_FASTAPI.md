# RAG Chatbot - FastAPI Deployment Guide

## Overview

RAG Chat is now powered by **FastAPI instead of Streamlit**, providing:
- ✅ Real-time WebSocket streaming responses
- ✅ Custom professional web UI
- ✅ Better session management
- ✅ Lighter deployment footprint
- ✅ Full language support (EN, DE, ZH, FR, ES)

## Architecture

```
Frontend (index.html + JavaScript)
         ↓ WebSocket + REST API
         ↓
Backend (FastAPI)
         ↓
RAG Engine (LangChain + Chroma)
         ↓
Google Gemini API
```

## Local Development

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export GEMINI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"  # Alternative
```

### 3. Run Application

```bash
python -m uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

Then visit: `http://localhost:8000/index`

## HuggingFace Spaces Deployment

### 1. Prepare Files

File structure must be:
```
rag-chatbot/
├── fastapi_app.py      (FastAPI backend)
├── index.html          (Frontend)
├── rag_engine.py       (RAG logic)
├── requirements.txt    (Dependencies)
├── Dockerfile         (For containerization)
├── .env              (API Keys - create locally)
└── chroma/           (Vector DB)
```

### 2. Create `.env` File

```
GEMINI_API_KEY=your_key_here
```

### 3. Push to HuggingFace

```bash
git add .
git commit -m "Migrate from Streamlit to FastAPI"
git push
```

### 4. Space Configuration

In HuggingFace Space settings:
- **Docker** runtime
- **8000** port expose
- **_timeout: 3600** (1 hour)

## API Endpoints

### REST API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| GET | `/models` | Available models |
| POST | `/session/create` | Create chat session |
| GET | `/session/{id}` | Get session info |
| POST | `/session/{id}/language` | Change language |
| POST | `/session/{id}/model` | Change model |
| GET | `/session/{id}/history` | Get conversation history |
| POST | `/session/{id}/clear` | Clear history |
| POST | `/chat` | Chat (non-streaming) |

### WebSocket

```
ws://localhost:8000/ws/chat/{session_id}

Client → Server: {"type": "message", "content": "user message"}
Server → Client: {"type": "start", "user_message": "..."}
Server → Client: {"type": "chunk", "content": "streamed text"}
Server → Client: {"type": "complete", "total_length": 123}
```

## Features

### 1. Real-time WebSocket Streaming
- Messages are streamed character-by-character
- Smooth UX with live typing effect
- Automatic reconnection on disconnect

### 2. Multi-language Support
- English (en)
- German (de)
- Chinese (zh)
- French (fr)
- Spanish (es)

Language selection is saved to localStorage

### 3. Model Selection
- gemini-2.5-flash-lite (Fast, free tier)
- gemini-2.5-flash (Balanced)
- gemini-2.5-pro (Best quality, limited)

### 4. Session Management
- Unique session ID per user
- Persisted conversation history
- Automatic session creation
- Clear history button

### 5. RAG Features
- Vector database (Chroma) with PDF documents
- Context-aware retrieval (MMR)
- Multi-model fallback for quota management
- Model switching on API limits

## Configuration

### Temperature & Sampling

In `rag_engine.py`, adjust synthesis parameters:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,      # 0=deterministic, 1=creative
    top_p=0.85,          # Nucleus sampling
    top_k=50,            # Top-k sampling
)
```

### Knowledge Base

Place PDF files in `data_sources/`:

```bash
python main_backup.py  # Initialize KB from PDFs
```

## Monitoring

### Logs

Check application logs:
```bash
docker logs <container-id>
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### WebSocket Connection Fails
- Check if server is running on port 8000
- Verify CORS settings in fastapi_app.py
- Check browser console for errors

### Models Not Available
- Verify GOOGLE_API_KEY is set correctly
- Check quota at console.cloud.google.com
- Review logs for model discovery errors

### Responses Slow
- Try smaller model (Flash Lite)
- Check Chroma vector DB performance
- Verify network latency

### CORS Issues

FastAPI already allows all origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)
```

## Performance Tips

1. **Use Flash Lite** for faster responses
2. **Keep context window small** - fewer history items
3. **Enable browser caching** for static files
4. **Monitor vector DB size** - too many embeddings slow retrieval

## Migration from Streamlit

### What Changed
- ❌ Removed Streamlit dependency (faster startup)
- ✅ Added WebSocket for streaming (better UX)
- ✅ Custom HTML/CSS/JS frontend (more control)
- ✅ API-first architecture (REST + WebSocket)
- ✅ Better session management (database-ready)

### What Stayed Same
- ✅ Same RAG engine (LangChain)
- ✅ Same knowledge base (Chroma)
- ✅ Same LLM (Google Gemini)
- ✅ Same prompts and logic

## Future Improvements

- [ ] Persistent database for sessions
- [ ] User authentication
- [ ] Upload custom documents
- [ ] Streaming from LLM directly
- [ ] Rate limiting per session
- [ ] Analytics dashboard

## Support

For issues, check:
1. `.env` file has GOOGLE_API_KEY
2. Dependencies installed: `pip list | grep -E "fastapi|websockets|langchain|google"`
3. Port 8000 is available: `netstat -an | grep 8000`
4. Vector DB initialized: check `chroma/` directory exists
