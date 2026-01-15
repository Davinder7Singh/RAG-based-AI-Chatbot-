# RAG-based-AI-Chatbot-
![Screenshot_15-1-2026_093_localhost](https://github.com/user-attachments/assets/fd646fb5-544d-4d09-8bdf-e19a55275ecb)
A strict, document-grounded Retrieval-Augmented Generation (RAG) chatbot built using LangGraph, FAISS, HuggingFace embeddings, and Groq LLMs. The system answers questions only from a provided PDF document (Agentic AI eBook) and avoids hallucinations by enforcing context-only generation.

# Project Overview-
This project implements an Agentic AI RAG pipeline with:<br>
* PDF ingestion and semantic chunking<br>
* Vector search using FAISS<br>
* Controlled answer generation using Groq LLMs
* Agent-style workflow orchestration using LangGraph
* Interactive chat UI built with Streamlit

The chatbot is designed for academic and technical accuracy


# Architecture-

PDF (Ebook-Agentic-AI.pdf)
↓
PDF Loader (PyPDFLoader)
↓
Text Chunking (RecursiveCharacterTextSplitter)
↓
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
↓
FAISS Vector Store
↓
LangGraph (Retrieve → Generate)
↓
Groq LLM (llama-3.1-8b-instant)
↓
Streamlit Chat Interface


# Project Structure-
RAG BASED AI CHATBOT/
│
├── Ebook-Agentic-AI.pdf # Source document
├── embedding.py # PDF ingestion & FAISS index creation
├── rag_graph.py # LangGraph RAG pipeline
├── app.py # Streamlit UI
├── faiss_index/ # Saved vector store
├── .env # Environment variables (Groq API key)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


# Files Explained-
1. embedding.py – Document Ingestion
* Loads the Agentic AI PDF
* Splits text into semantic chunks
* Generates embeddings using HuggingFace
* Stores vectors locally using FAISS

2. rag_graph.py-
Implements a LangGraph workflow with:
* Retrieve node
MMR-based vector search
Noise filtering (short chunks, TOC removal)
Confidence estimation
* Generate node
Strict context-only answering
No outside knowledge
Partial-answer summarization if applicable

3. app.py-
Chat-style UI:
* User question
* Grounded answer
* Retrieved context chunks
* Confidence score
 
4.  .env-
* groq_api_key_here(Create a .env file in the projec)


# How to Run the Project-
1.Create and Activate Virtual Environment
    python -m venv .venv
   .venv\Scripts\activate
2. Install Dependencies
  pip install -r requirements.txt
3. Run Once-
   embedding.py
   rag_graph.py
   app.py
4.Start the Streamlit application:
  streamlit run app.py
  The chatbot will open in your browser at http://localhost:8501
   





