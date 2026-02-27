---
title: RAG Chatbot
emoji: 💬
colorFrom: purple
colorTo: purple
sdk: docker
pinned: false
---

# 💬 AI RAG Chatbot

A modern, multilingual AI-powered chatbot built with FastAPI, LangChain, and Google Gemini API. Real-time WebSocket streaming with Retrieval-Augmented Generation (RAG) for intelligent responses.

## ✨ Features

- **🌍 Multilingual**: English, German, Chinese, French, Spanish
- **⚡ Real-time Chat**: WebSocket streaming for instant responses
- **🤖 Multi-Model Support**: Gemini Flash Lite, Flash, Pro (with automatic fallback)
- **📚 RAG Engine**: Retrieval-Augmented Generation with vector database
- **💾 Session Management**: Persistent conversation history with localStorage
- **🎨 Modern UI**: Professional gradient interface with responsive design

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API Key ([get free here](https://aistudio.google.com/app/apikey))
- Docker (for HuggingFace deployment)

### Local Development

```bash
# Clone repository
git clone https://github.com/shaofei-liu/portfolio-chatbot.git
cd rag-chatbot

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# OR (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export GEMINI_API_KEY=your_api_key_here  # Linux/Mac
set GEMINI_API_KEY=your_api_key_here     # Windows
```

## Running the Server

```bash
# Run FastAPI server
python fastapi_app.py
```

Visit [http://localhost:8000](http://localhost:8000)

## 🐳 Docker Deployment

```bash
# Build image
docker build -t rag-chatbot .

# Run container
docker run -e GEMINI_API_KEY=your_key -p 8000:8000 rag-chatbot
```



## 📁 Project Structure

```
├── fastapi_app.py           # FastAPI backend with WebSocket
├── index.html               # Professional web UI
├── rag_engine.py            # RAG pipeline & embeddings
├── model_monitor.py         # API quota tracking
├── google_pricing_sync.py   # Model discovery
├── requirements.txt         # Dependencies
├── Dockerfile               # Container configuration
├── DEPLOYMENT_FASTAPI.md    # Detailed deployment guide
├── data_sources/            # PDF knowledge base (auto-indexed)
└── chroma/                  # Vector database (auto-generated)
```

## 🏗️ Tech Stack

| Component | Technology |
|-----------|----------|
| Backend | [FastAPI](https://fastapi.tiangolo.com) 0.104.1 |
| Frontend | HTML5 + CSS3 + JavaScript (WebSocket) |
| LLM | [Google Gemini API](https://ai.google.dev) |
| Framework | [LangChain](https://langchain.com) 0.2+ |
| Embeddings | HuggingFace + Sentence Transformers |
| Vector DB | [Chroma](https://trychroma.com) |
| Server | [Uvicorn](https://www.uvicorn.org) 0.24.0 |

## ⚙️ Configuration

Set environment variable:
```bash
GEMINI_API_KEY=your_google_api_key_here
```

Optional: Configure additional settings in `fastapi_app.py`:
- `HOST`: Default "0.0.0.0"
- `PORT`: Default 8000
- `MAX_HISTORY`: Conversation history limit
- `CHROMA_PERSIST_DIR`: Vector DB location

## 📡 API Endpoints

**REST API:**
- `POST /session/create` - Create new conversation session
- `POST /chat` - Send message (sync endpoint)
- `GET /history/{session_id}` - Get conversation history
- `POST /clear/{session_id}` - Clear conversation

**WebSocket:**
- `WS /ws/chat/{session_id}` - Real-time chat streaming
  - Send: `{"message": "Your question"}`
  - Receive: Streamed response chunks

**Health:**
- `GET /health` - Server status check

## 📖 Knowledge Base

The chatbot learns from PDF files in `data_sources/`:
- Any `.pdf` files in this folder are automatically indexed
- Vector database is created on first run
- Updates are reflected on next restart

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "GEMINI_API_KEY not found" | Set environment variable before starting |
| "Port 8000 already in use" | Change PORT in fastapi_app.py |
| "No PDFs found" | Add PDF files to `data_sources/` folder |
| "429 Resource Exhausted" | API quota reached; fallback to lighter model |
| Connection refused | Ensure server is running (`python fastapi_app.py`) |

## 📚 Documentation

See [DEPLOYMENT_FASTAPI.md](DEPLOYMENT_FASTAPI.md) for:
- Detailed deployment instructions
- API documentation
- WebSocket protocol specification
- Architecture diagram
- Production recommendations

## License

MIT

---

**Built with ❤️ by Shaofei Liu**
