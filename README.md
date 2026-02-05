# RAG Chatbot

A multilingual AI-powered chatbot built with Streamlit, LangChain, and Google Gemini API. Uses Retrieval-Augmented Generation (RAG) with vector database to provide intelligent, context-aware responses.

## Overview

This project implements a modern conversational AI system that combines:
- **Retrieval-Augmented Generation (RAG)**: Retrieves relevant context from knowledge base before generating responses
- **Vector Database**: Efficient semantic search using embeddings
- **Large Language Model**: Google Gemini API for natural language understanding and generation
- **Bilingual Support**: English and Deutsch interfaces
- **Smart Error Handling**: Automatic fallback mechanisms when API calls fail

Perfect for building custom question-answering systems, documentation assistants, or knowledge base chatbots.

## Features

- **Bilingual Support**: Seamlessly switch between English and German interfaces
- **RAG Architecture**: Combines retrieval + generation for accurate, contextual responses
- **Vector Database**: Chroma vector database for fast semantic search
- **LangChain Integration**: Modular pipeline with LangChain components
- **Google Gemini API**: State-of-the-art language model
- **HuggingFace Embeddings**: High-quality multilingual embeddings
- **Smart Fallback System**: Automatically tries alternative models if primary fails
- **Conversation History**: Maintains chat context across sessions
- **Knowledge Base**: Initialize from PDFs or text documents
- **Real-time Responses**: Fast inference for interactive experience
- **Error Recovery**: Robust handling of API failures and timeouts

## Quick Start

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Run Streamlit app
streamlit run main.py
```

### Docker Deployment

```bash
docker build -t rag-chatbot .
docker run -e GOOGLE_API_KEY=your_key -p 8501:8501 rag-chatbot
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```
GOOGLE_API_KEY=your_google_gemini_api_key
CHROMA_DB_PATH=./chroma_db  # Optional: custom database path
```

### Knowledge Base Setup

The chatbot can be initialized with knowledge from:
1. **PDF Documents**: Automatically processed and indexed
2. **Text Files**: Plain text knowledge base
3. **Structured Data**: From `data_sources/` directory

Place your documents in the `data_sources/` folder before starting.

## Project Structure

```
rag-chatbot/
├── main.py                    # Streamlit app entry point
├── rag_engine.py             # Core RAG engine logic
├── requirements.txt          # Python dependencies
├── data_sources/             # Knowledge base documents
│   └── README.md             # Documentation about data sources
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                 # This file
```

## Technical Stack

- **Frontend**: Streamlit 1.28+
- **Backend**: LangChain 0.1+
- **LLM**: Google Gemini API
- **Vector DB**: Chroma 0.4+
- **Embeddings**: HuggingFace Transformers
- **Language**: Python 3.9+

## API & Components

### Main Components

#### RAG Engine (`rag_engine.py`)
- `RAGEngine`: Core class handling retrieval and generation
- Vector database initialization
- Document loading and processing
- Query response pipeline

#### Streamlit App (`main.py`)
- Web UI with sidebar language selector
- Chat interface
- Document upload (optional)
- Conversation history management

### Key Functions

```python
# Initialize RAG engine
from rag_engine import RAGEngine

engine = RAGEngine(
    api_key="your_google_key",
    language="en"  # or "de" for German
)

# Get response
response = engine.query("What is the weather?")
print(response)
```

## Usage Examples

### Web Interface
1. Open application in browser (typically `http://localhost:8501`)
2. Select language from sidebar (English/Deutsch)
3. Type your question in the chat input
4. Receive context-aware response

### Python Integration
```python
from rag_engine import RAGEngine

# Initialize
engine = RAGEngine(
    api_key="your_api_key",
    language="en"
)

# Query
response = engine.query("Tell me about renewable energy")
print(f"Response: {response}")
```

## Knowledge Base Management

### Adding Documents
1. Place PDF/text files in `data_sources/` folder
2. Restart the application
3. System automatically processes and indexes documents

### Supported Formats
- PDF (.pdf)
- Text (.txt)
- Markdown (.md)

## Performance & Reliability

- **Response Time**: 2-5 seconds average (depends on API latency)
- **Context Window**: 4K tokens per query
- **Embedding Model**: Multilingual, 384-dimensional vectors
- **Vector DB**: Supports millions of documents

## Error Handling

The system includes robust error handling:
- **API Failures**: Automatic fallback to alternative models
- **Timeout Handling**: Graceful degradation with user notification
- **Rate Limiting**: Smart retry logic with exponential backoff
- **Invalid Input**: User-friendly error messages

## Troubleshooting

**Issue**: API key not recognized
- **Solution**: Ensure `GOOGLE_API_KEY` is set correctly in `.env`

**Issue**: Knowledge base not loading
- **Solution**: Check file permissions in `data_sources/` directory

**Issue**: Slow responses
- **Solution**: Reduce knowledge base size or use GPU acceleration

**Issue**: Language not switching
- **Solution**: Clear browser cache and restart app

## Advanced Configuration

### Custom Embeddings
Modify `rag_engine.py` to use different embedding models:
```python
embeddings = HuggingFaceEmbeddings(model_name="your-model")
```

### Database Persistence
Vector database is stored in `chroma_db/` directory and persists across sessions.

### Language Customization
Add new languages by extending the language configuration in `main.py`.

## Deployment

### Streamlit Cloud
```bash
git push origin main
# Deploy via Streamlit Cloud dashboard
```

### HuggingFace Spaces
Deploy using Docker with Streamlit SDK.

### Cloud Platforms
Compatible with AWS, Google Cloud, Azure, etc.

## Related Projects

- **Portfolio**: https://github.com/shaofei-liu/portfolio
- **Dog Breed Classifier**: https://github.com/shaofei-liu/dog-breed-classifier
- **IRevRNN Research**: https://github.com/shaofei-liu/irevrnn

## Dependencies

See `requirements.txt` for complete list. Key packages:
- streamlit
- langchain
- chromadb
- google-generativeai
- sentence-transformers
- python-dotenv

## License

This project is provided for educational and commercial use.

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Note**: Requires valid Google Gemini API key. Get one for free at https://ai.google.dev/
