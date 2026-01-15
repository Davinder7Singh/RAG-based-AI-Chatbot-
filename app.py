import streamlit as st
from rag_graph import build_graph

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    layout="wide"
)

st.title("📘 Agentic AI RAG Chatbot")
#st.caption("Answers strictly grounded in the Agentic AI eBook")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question from the Agentic AI eBook")

if user_input:
    with st.spinner("Thinking..."):
        result = st.session_state.graph.invoke({
            "question": user_input
        })

        st.session_state.chat_history.append((
            user_input,
            result["answer"],
            result["context"],
            result["confidence"]
        ))

for q, a, ctx, conf in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        st.write(a)

        with st.expander("📚 Retrieved Context"):
            for i, c in enumerate(ctx):
                st.markdown(f"**Chunk {i+1}:** {c}")

        st.markdown(f"**Confidence Score:** `{conf}`")
