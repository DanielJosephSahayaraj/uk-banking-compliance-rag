
from graph import app as rag_app
import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="AI Compliance Assistant", layout="wide")

st.title("📊 AI Regulatory Compliance Assistant")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "state" not in st.session_state:
    st.session_state.state = {
        "query": "",
        "messages": [],
        "document_text": None
    }

# Sidebar mode selection
mode = st.sidebar.radio("Choose Mode", ["Chat (RAG)", "Document Compliance"])

user_input = st.text_input("Enter your query:")

# ---------------------------
# CHAT MODE (RAG)
# ---------------------------
if mode == "Chat (RAG)":
    if st.button("Ask") and user_input:

        # update state
        st.session_state.state["query"] = user_input
        st.session_state.state["messages"].append(
            HumanMessage(content=user_input)
        )

        # CALL GRAPH
        result = rag_app.invoke(st.session_state.state)

        # IMPORTANT: merge instead of overwrite
        st.session_state.state.update({
            "final_response": result.get("final_response"),
            "compliance": result.get("compliance"),
            "messages": result.get("messages", st.session_state.state["messages"])
        })

        st.write("### 🤖 Response")
        st.write(result.get("final_response"))

        if result.get("compliance"):
            st.write("### 📊 Compliance")
            st.json(result["compliance"])
# ---------------------------
# DOCUMENT MODE
# ---------------------------
elif mode == "Document Compliance":

    doc_text = st.text_area("Paste your document text here:")

    if st.button("Analyze Document") and doc_text:

        st.session_state.state["query"] = "document_compliance"
        st.session_state.state["document_text"] = doc_text

        result = rag_app.invoke(st.session_state.state)

        st.session_state.state.update(result)

        st.write("### 📄 Compliance Report")
        st.json(result.get("compliance"))