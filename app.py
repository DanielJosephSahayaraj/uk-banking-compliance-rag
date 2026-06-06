from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage
from graph import app as rag_app
from history import load_history, save_history, clear_history

# ── Page config ──
st.set_page_config(
    page_title="UK Banking Compliance Assistant",
    page_icon="🏦",
    layout="wide"
)

# ── Header ──
st.title("🏦 UK Banking Regulatory Compliance Assistant")
st.caption("Powered by Claude AI · FCA & PRA Regulatory Knowledge Base")

# ── Session state ──
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

if "state" not in st.session_state:
    st.session_state.state = {
        "query": "",
        "messages": st.session_state.messages,
        "document_text": None,
        "retrieved_chunks": [],
        "context": "",
        "final_response": "",
        "critique": "",
        "compliance": None,
        "rewrite_query": None,
        "hyde_answer": None
    }

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio(
        "Choose Mode",
        ["💬 Chat (RAG)", "📄 Document Compliance"]
    )
    st.divider()
    if st.button("🗑️ Clear History"):
        clear_history()
        st.session_state.messages = []
        st.session_state.state["messages"] = []
        st.success("History cleared!")
    st.divider()
    st.markdown("**Knowledge Base:**")
    st.markdown("- FCA Consumer Duty")
    st.markdown("- PRA Rulebook")
    st.markdown("- AML Guidelines")
    st.markdown("- Operational Resilience")
    st.markdown("- TCF Management Information")
    st.divider()
    st.markdown("**API Docs:**")
    st.markdown("[📡 Swagger UI](http://localhost:8000/docs)")


# ──────────────────────────────────────────
# CHAT MODE
# ──────────────────────────────────────────
if mode == "💬 Chat (RAG)":

    # Display chat history
    for msg in st.session_state.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    # Chat input
    user_input = st.chat_input(
        "Ask about FCA regulations, Consumer Duty, AML, operational resilience..."
    )

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)

        # Update state
        st.session_state.state["query"] = user_input
        st.session_state.state["document_text"] = None
        st.session_state.state["messages"] = st.session_state.messages

        # Run graph
        with st.spinner("Searching regulatory documents..."):
            result = rag_app.invoke(st.session_state.state)

        # Update session
        st.session_state.state.update(result)
        st.session_state.messages = result.get(
            "messages",
            st.session_state.messages
        )

        # Save history
        save_history(st.session_state.messages)

        # Show response
        with st.chat_message("assistant"):
            st.write(result.get("final_response", "No response generated."))

        # Show retrieved chunks
        chunks = result.get("retrieved_chunks", [])
        if chunks:
            with st.expander(f"📚 Source Documents ({len(chunks)} chunks)"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.text(chunk[:300] + "...")
                    st.divider()


# ──────────────────────────────────────────
# DOCUMENT COMPLIANCE MODE
# ──────────────────────────────────────────
elif mode == "📄 Document Compliance":
    st.subheader("📄 Document Compliance Checker")
    st.caption(
        "Paste any banking document to check for "
        "FCA/PRA/GDPR compliance issues"
    )

    doc_text = st.text_area(
        "Paste your document text here:",
        height=300,
        placeholder="Paste a policy document, procedure, or any banking text..."
    )

    if st.button("🔍 Analyse Document", type="primary") and doc_text:
        st.session_state.state["query"] = "document_compliance"
        st.session_state.state["document_text"] = doc_text

        with st.spinner("Analysing document for compliance issues..."):
            result = rag_app.invoke(st.session_state.state)

        st.session_state.state.update(result)
        compliance = result.get("compliance", {})

        if compliance:
            # Metrics row
            risk = compliance.get("risk_level", "Unknown")
            color = {
                "Low": "🟢", "Medium": "🟡", "High": "🔴"
            }.get(risk, "⚪")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Level", f"{color} {risk}")
            with col2:
                st.metric(
                    "Issues Found",
                    len(compliance.get("violations", []))
                )
            with col3:
                st.metric(
                    "Recommendations",
                    len(compliance.get("recommendations", []))
                )

            st.divider()

            # Summary
            st.markdown("### 📋 Summary")
            st.info(compliance.get("summary", ""))

            # Violations
            violations = compliance.get("violations", [])
            if violations:
                st.markdown("### ⚠️ Compliance Issues")
                for v in violations:
                    severity = v.get("severity", "")
                    msg = (
                        f"**{v.get('type')}**\n\n"
                        f"{v.get('description')}\n\n"
                        f"📌 *{v.get('text_span', '')}*\n\n"
                        f"💡 **Fix:** {v.get('recommendation')}"
                    )
                    if severity == "High":
                        st.error(msg)
                    elif severity == "Medium":
                        st.warning(msg)
                    else:
                        st.info(msg)
            else:
                st.success("✅ No compliance issues found!")

            # Recommendations
            recs = compliance.get("recommendations", [])
            if recs:
                st.markdown("### 💡 General Recommendations")
                for rec in recs:
                    st.markdown(f"- {rec}")

            # Full JSON
            with st.expander("🔧 Full Compliance Report (JSON)"):
                st.json(compliance)