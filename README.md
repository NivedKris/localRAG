# Local RAG System with Streamlit

This is a fully local RAG (Retrieval-Augmented Generation) system that allows you to upload PDF files and chat with their contents, all while keeping your data private and local.

## Features

- Upload PDF documents
- Automatic text extraction, embedding, and indexing
- Chat interface to query your documents
- View sources and references for generated responses
- Fully local operation - no data leaves your machine

## Requirements

- Python 3.8+
- Weaviate running locally
- Ollama running locally with the following models:
  - all-minilm (for embeddings)
  - tinyllama (for text generation)

## Setup

1. Make sure you have Weaviate running locally:
   ```
   docker run -d -p 50051:50051 --name weaviate-server semitechnologies/weaviate:latest
   ```

2. Make sure you have Ollama running locally and the required models installed:
   ```
   ollama pull all-minilm
   ollama pull tinyllama
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Navigate to the "Upload Documents" tab
2. Upload one or more PDF files
3. Click "Process Documents" to extract text and store embeddings
4. Switch to the "Chat" tab
5. Ask questions about your documents

## How it works

1. PDF documents are processed and split into chunks by page
2. Each chunk is embedded using the all-minilm model via Ollama
3. Embeddings are stored in Weaviate vector database
4. When you ask a question, it is embedded and similar chunks are retrieved
5. The retrieved context is sent to tinyllama along with your question
6. The response is generated based only on the retrieved context
