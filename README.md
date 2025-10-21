## Project Overview

This project is a comprehensive demonstration of Generative AI applications in the healthcare domain, built using Streamlit for an interactive web interface. It leverages advanced AI models and techniques to provide a multi-modal health assistant capable of conversational chat, document-based retrieval-augmented generation (RAG), and optical character recognition (OCR) for prescription analysis.

### Key Highlights
- **Purpose**: To showcase practical AI integrations for health-related tasks, including rule-based chat, document search, and prescription parsing, all powered by open-source models and local processing where possible.
- **Tech Stack**:
  - **Frontend/UI**: Streamlit for building the web app with a clean, responsive interface.
  - **AI Models**: Mistral-7B-Instruct-v0.3 via Hugging Face Inference API for natural language generation, summarization, and structured data extraction.
  - **Embeddings & Search**: Sentence Transformers (all-MiniLM-L6-v2) for text embeddings, FAISS for efficient vector similarity search.
  - **OCR**: Tesseract for text extraction from images, integrated with Mistral for intelligent structuring of prescription data.
  - **Data Handling**: Support for multiple file types (PDF, DOCX, XLSX, images) using libraries like PyPDF2, python-docx, and pandas.
  - **Persistence**: Local storage for vector indices and document metadata in JSON format.
- **Architecture**: Modular backend with separate modules for RAG, chat agents, and OCR, ensuring scalability and maintainability.
- **Features**: Four main modes – Chat, Upload & RAG, OCR, and About – each demonstrating different AI capabilities.
- **Performance Metrics**: Tracks uploaded documents, embeddings, queries, and average latency for monitoring.
- **Security & Ethics**: Includes disclaimers for educational use only, emphasizing that outputs are not for real medical decisions.

## Features

### 1. Chat Mode
- Interactive conversational interface for health-related queries.
- Combines RAG search on uploaded documents with fallback to direct LLM responses.
- Maintains chat history for context-aware conversations.
- Advises consulting professionals for symptom-related queries.

### 2. Upload & RAG Mode
- Upload documents in various formats (TXT, PDF, DOCX, XLSX, images).
- Extracts text using appropriate parsers (e.g., OCR for images).
- Chunks text and creates embeddings for efficient search.
- Performs semantic search with top-k results, including relevance scores.
- Generates Mistral-powered answers based on retrieved context.
- Lists all uploaded documents for reference.

### 3. OCR Mode
- Upload prescription images (PNG, JPG, JPEG).
- Extracts raw text using Tesseract OCR.
- Structures extracted text into JSON format via Mistral, including fields like doctor details, patient info, diagnosis, and medicines.
- Displays structured data in a table for medicines.
- Provides a human-readable summary of the prescription.

### 4. About Mode
- Provides information about the demo, including the disclaimer and feature overview.

## Architecture

The application follows a modular architecture:

- **app.py**: Main Streamlit application handling UI logic, mode selection, and user interactions.
- **backend/**:
  - **rag.py**: Core RAG implementation with text extraction, chunking, embedding, FAISS indexing, search, and context-based answering.
  - **agents.py**: Chat agent integrating RAG with LLM for smart responses.
  - **ocr.py**: OCR pipeline for image processing, text extraction, structuring, and summarization.
- **config/settings.py**: Configuration file with disclaimers and settings.
- **db/**: Directory for persistent storage of FAISS index and document metadata.
- **.streamlit/secrets.toml**: Secure storage for API keys (Hugging Face, etc.).

Data flows from user uploads through processing modules to generate outputs, with all AI inferences handled via Hugging Face API for privacy and efficiency.
