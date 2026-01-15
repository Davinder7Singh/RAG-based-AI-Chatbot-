# RAG-based-AI-Chatbot-
![Screenshot_15-1-2026_093_localhost](https://github.com/user-attachments/assets/fd646fb5-544d-4d09-8bdf-e19a55275ecb)
A strict, document-grounded Retrieval-Augmented Generation (RAG) chatbot built using LangGraph, FAISS, HuggingFace embeddings, and Groq LLMs. The system answers questions only from a provided PDF document (Agentic AI eBook) and avoids hallucinations by enforcing context-only generation.

# Project Overview-
This project implements an Agentic AI RAG pipeline with:
* PDF ingestion and semantic chunking
* Vector search using FAISS
* Controlled answer generation using Groq LLMs
* Agent-style workflow orchestration using LangGraph
* Interactive chat UI built with Streamlit

The chatbot is designed for academic and technical accuracy<br>


# Architecture-

PDF (Ebook-Agentic-AI.pdf)<br>
↓<br>
PDF Loader (PyPDFLoader)<br>
↓<br>
Text Chunking (RecursiveCharacterTextSplitter)<br>
↓<br>
Embeddings (sentence-transformers/all-MiniLM-L6-v2)<br>
↓<br>
FAISS Vector Store<br>
↓<br>
LangGraph (Retrieve → Generate)<br>
↓<br>
Groq LLM (llama-3.1-8b-instant)<br>
↓<br>
Streamlit Chat Interface<br>


# Project Structure-
RAG BASED AI CHATBOT/<br>
│<br>
├── Ebook-Agentic-AI.pdf # Source document<br>
├── embedding.py # PDF ingestion & FAISS index creation<br>
├── rag_graph.py # LangGraph RAG pipeline<br>
├── app.py # Streamlit UI<br>
├── faiss_index/ # Saved vector store<br>
├── .env # Environment variables (Groq API key)<br>
├── requirements.txt # Python dependencies<br>
└── README.md # Project documentation<br>


# Files Explained-
1. embedding.py – Document Ingestion<br>
* Loads the Agentic AI PDF
* Splits text into semantic chunks
* Generates embeddings using HuggingFace
* Stores vectors locally using FAISS

2. rag_graph.py-<br>
Implements a LangGraph workflow with:<br>
* Retrieve node
MMR-based vector search<br>
Noise filtering (short chunks, TOC removal)<br>
Confidence estimation<br>
* Generate node
Strict context-only answering<br>
No outside knowledge<br>
Partial-answer summarization if applicable<br>

** 3. app.py-<br>
Chat-style UI:<br>
* User question
* Grounded answer
* Retrieved context chunks
* Confidence score
 
4.  .env-<br>
* groq_api_key_here(Create a .env file in the projec)


# How to Run the Project-
1.Create and Activate Virtual Environment<br>
    python -m venv .venv<br>
   .venv\Scripts\activate<br>
2. Install Dependencies<br>
  pip install -r requirements.txt<br>
3. Run Once-<br>
   embedding.py<br>
   rag_graph.py<br>
   app.py<br>
4.Start the Streamlit application:<br>
  streamlit run app.py<br>
  The chatbot will open in your browser at http://localhost:8501<br>
   





