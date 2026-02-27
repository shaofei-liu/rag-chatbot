FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p chroma output

# Expose port
EXPOSE 7860

# Run FastAPI app with unbuffered output and detailed logging
CMD ["python", "-u", "-m", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
