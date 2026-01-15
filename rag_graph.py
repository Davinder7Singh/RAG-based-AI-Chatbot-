import os
from dotenv import load_dotenv
from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Check .env file.")

print("✅ GROQ_API_KEY loaded:", bool(GROQ_API_KEY))

# --------------------------------------------------
# Constants
# --------------------------------------------------
DB_PATH = "faiss_index"

# --------------------------------------------------
# LangGraph State
# --------------------------------------------------
class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str
    confidence: float

# --------------------------------------------------
# Embeddings (kept same to match FAISS index)
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# Load FAISS Vector Store
# --------------------------------------------------
vectordb = FAISS.load_local(
    DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

# --------------------------------------------------
# Groq LLM
# --------------------------------------------------
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

# --------------------------------------------------
# Retrieve Node (MMR + filtering)
# --------------------------------------------------
def retrieve(state: RAGState):
    docs = vectordb.max_marginal_relevance_search(
        state["question"],
        k=5,
        fetch_k=15
    )

    contexts = []
    scores = []

    for doc in docs:
        text = doc.page_content.strip()

        # Basic noise filtering
        if len(text) < 80:
            continue
        if "table of contents" in text.lower():
            continue

        contexts.append(text)

        # Use metadata score if available (FAISS similarity proxy)
        score = doc.metadata.get("score", 0.5)
        scores.append(score)

    if not contexts:
        return {
            "context": [],
            "confidence": 0.0
        }

    # Normalize confidence (0–1)
    confidence = round(min(1.0, sum(scores) / len(scores)), 2)

    return {
        "context": contexts,
        "confidence": confidence
    }

# --------------------------------------------------
# Generate Node (FIXED ANSWER LOGIC)
# --------------------------------------------------
def generate(state: RAGState):
    if not state["context"]:
        return {
            "answer": "I couldn't find relevant information in the provided document."
        }

    context_text = "\n\n".join(state["context"])

    prompt = f"""
You are an AI assistant working in a Retrieval-Augmented Generation (RAG) system.

Instructions:
- Answer the QUESTION using ONLY the information in the CONTEXT.
- If the context partially answers the question, summarize what is available.
- Do NOT say "I don't know" if any relevant information exists.
- Do NOT use outside knowledge.
- Be clear, concise, and factual.

Context:
{context_text}

Question:
{state['question']}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content.strip()
    }

# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
