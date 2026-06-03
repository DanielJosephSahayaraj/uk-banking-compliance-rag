# app.py
from dotenv import load_dotenv
load_dotenv()  # CRITICAL: This must run before importing graph or nodes!

import streamlit as st
from langchain_core.messages import HumanMessage
from graph import app as rag_app

st.set_page_config(page_title="AI Compliance Assistant", layout="wide")
st.title("📊 AI Regulatory Compliance Assistant")

# Streamlined Session state (Removed chat message history tracking)
if "state" not in st.session_state:
    st.session_state.state = {
        "query": "",
        "document_text": None
    }

# ---------------------------
# DOCUMENT COMPLIANCE MODE ONLY
# ---------------------------
doc_text = st.text_area("Paste your document text here:", height=300)

if st.button("Analyze Document") and doc_text:
    with st.spinner("Analyzing document compliance..."):
        st.session_state.state["query"] = "document_compliance"
        st.session_state.state["document_text"] = doc_text

        # Call your LangGraph workflow
        result = rag_app.invoke(st.session_state.state)
        st.session_state.state.update(result)

        st.write("### 📄 Compliance Report")
        st.json(result.get("compliance"))