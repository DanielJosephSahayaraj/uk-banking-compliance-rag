import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from graph import app as rag_app
from langchain_core.messages import HumanMessage

load_dotenv()

# ── FastAPI app ──
api = FastAPI(
    title="UK Banking Regulatory Compliance Assistant",
    description="""
    An intelligent RAG-powered compliance assistant for UK banking regulations.
    Built with Claude AI, LangGraph, and Qdrant.

    ## Features
    * **Chat** — Ask questions about FCA/PRA regulations
    * **Document** — Analyse documents for compliance violations
    * **Health** — Check API status
    """,
    version="1.0.0",
    contact={
        "name": "Daniel Joseph",
        "url": "https://linkedin.com/in/daniel-joseph-sahayaraj-aws-engineer"
    }
)

# ── CORS ──
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
@api.get("/", tags=["System"])
async def root():
    """Root endpoint — API is running."""
    return {
        "message": "UK Banking Regulatory Compliance API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }
# ──────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the FCA Consumer Duty requirements?",
                "session_id": "user_123"
            }
        }

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: Optional[list] = []

class DocumentRequest(BaseModel):
    document_text: str
    session_id: Optional[str] = "default"

    class Config:
        json_schema_extra = {
            "example": {
                "document_text": "Our bank shares customer data with third parties without consent...",
                "session_id": "user_123"
            }
        }

class DocumentResponse(BaseModel):
    summary: str
    risk_level: str
    violations: list
    recommendations: list
    session_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    vector_db: str

# ── Session storage ──
sessions = {}

def get_session_state(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "query": "",
            "messages": [],
            "document_text": None,
            "retrieved_chunks": [],
            "context": "",
            "final_response": "",
            "critique": "",
            "compliance": None,
            "rewrite_query": None,
            "hyde_answer": None
        }
    else:
        # Reset these fields for each new request
        sessions[session_id]["retrieved_chunks"] = []
        sessions[session_id]["context"] = ""
        sessions[session_id]["final_response"] = ""
        sessions[session_id]["rewrite_query"] = None
        sessions[session_id]["hyde_answer"] = None
        sessions[session_id]["compliance"] = None
        sessions[session_id]["document_text"] = None
    return sessions[session_id]
# ──────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────

@api.get("/health",
         response_model=HealthResponse,
         tags=["System"],
         summary="Check API health")
async def health_check():
    """Check if the API is running correctly."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model="claude-sonnet-4-5",
        vector_db="qdrant-cloud"
    )


@api.post("/chat",
          response_model=ChatResponse,
          tags=["RAG"],
          summary="Ask a banking compliance question")
async def chat(request: ChatRequest):
    try:
        # Get or create session
        state = get_session_state(request.session_id)

        # Set fresh state for this request
        state["query"] = request.query
        state["document_text"] = None
        state["rewrite_query"] = None
        state["hyde_answer"] = None
        state["retrieved_chunks"] = []
        state["context"] = ""
        state["final_response"] = ""
        state["critique"] = ""
        state["compliance"] = None

        # Add message to history
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=request.query)
        ]

        print(f"Running pipeline for: {request.query}")

        # Run LangGraph pipeline
        result = rag_app.invoke(state)

        print(f"Pipeline complete")
        print(f"Rewrite: {result.get('rewrite_query', 'None')}")
        print(f"Chunks: {len(result.get('retrieved_chunks', []))}")
        print(f"Answer preview: {str(result.get('final_response', ''))[:100]}")

        # Update session
        sessions[request.session_id] = result

        # Build sources
        sources = []
        chunks = result.get("retrieved_chunks") or []
        for i, chunk in enumerate(chunks):
            sources.append({
                "chunk_id": i + 1,
                "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
            })

        answer = result.get("final_response") or "No response generated."

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            sources=sources
        )

    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

@api.post("/document",
          response_model=DocumentResponse,
          tags=["Compliance"],
          summary="Analyse a document for compliance violations")
async def analyse_document(request: DocumentRequest):
    """
    Analyse a banking document for FCA/PRA/GDPR compliance violations.

    Returns:
    - Risk level (Low/Medium/High)
    - Specific violations with severity
    - Actionable recommendations
    """
    try:
        state = get_session_state(request.session_id)
        state["query"] = "document_compliance"
        state["document_text"] = request.document_text

        result = rag_app.invoke(state)
        sessions[request.session_id] = result

        compliance = result.get("compliance", {})

        return DocumentResponse(
            summary=compliance.get("summary", "Analysis complete"),
            risk_level=compliance.get("risk_level", "Unknown"),
            violations=compliance.get("violations", []),
            recommendations=compliance.get("recommendations", []),
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document analysis error: {str(e)}"
        )


@api.delete("/session/{session_id}",
            tags=["System"],
            summary="Clear a session")
async def clear_session(session_id: str):
    """Clear conversation history for a specific session."""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}


@api.get("/sessions",
         tags=["System"],
         summary="List active sessions")
async def list_sessions():
    """List all active conversation sessions."""
    return {
        "active_sessions": list(sessions.keys()),
        "count": len(sessions)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:api", host="0.0.0.0", port=port)