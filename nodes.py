import os
import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, TEMPERATURE,
    ALLOWED_KEYWORDS, BLOCKED_KEYWORDS
)
from cache import get_cached_response, save_to_cache
from retriever import hybrid_retrieve
from history import summarize_history

# ── LLM ──
llm = ChatAnthropic(
    model=CLAUDE_MODEL,
    api_key=ANTHROPIC_API_KEY,
    temperature=TEMPERATURE,
    max_tokens=3000
    
)


# ──────────────────────────────────────────
# GUARDRAIL NODE
# ──────────────────────────────────────────
def guardrail_node(state: dict) -> dict:
    q_lower = state["query"].lower()

    if any(word in q_lower for word in BLOCKED_KEYWORDS):
        return {
            "final_response": "I cannot assist with harmful or illegal topics.",
            "next": "end"
        }

    if not any(word in q_lower for word in ALLOWED_KEYWORDS):
        return {
            "final_response": (
                "I am a UK Banking Regulatory Compliance Assistant. "
                "I can only answer questions related to FCA/PRA regulations, "
                "compliance, AML, KYC, Consumer Duty, and banking policy. "
                "How can I help you today?"
            ),
            "next": "end"
        }

    return {"next": "cache_check"}


# ──────────────────────────────────────────
# CACHE CHECK NODE
# ──────────────────────────────────────────
def cache_check_node(state: dict) -> dict:
    q_lower = state["query"].lower()

    skip_words = [
        "date", "time", "today", "now", "current",
        "news", "latest", "recent", "search", "web"
    ]
    if any(word in q_lower for word in skip_words):
        print("Time-sensitive — skipping cache")
        return {"next": "router"}

    cached = get_cached_response(state["query"])
    if cached:
        print("Cache HIT — skipping pipeline")
        return {"final_response": cached, "next": "end"}

    return {"next": "router"}


# ──────────────────────────────────────────
# ROUTER NODE
# ──────────────────────────────────────────
def router_node(state: dict) -> dict:
    q_lower = state["query"].lower()

    # Document compliance route
    if state.get("query", "").startswith("document_compliance"):
        return {"next": "document_compliance"}

    # Date/time route
    if any(word in q_lower for word in ["date", "time", "today", "now"]):
        return {"next": "date_tool"}

    # Web search route
    search_keywords = [
        "current", "today", "news", "latest", "recent",
        "when did", "who won", "stock price", "weather"
    ]
    if any(word in q_lower for word in search_keywords):
        return {"next": "web_search"}

    # LLM classifier — retrieve or not
    prompt = f"""You are a binary classifier. Reply with exactly one word.

RETRIEVE — if the question needs information from the banking 
           compliance database (FCA rules, PRA regulations, 
           AML, KYC, Consumer Duty, Basel, banking policy)

NO_RETRIEVE — if the question can be answered directly

Do not explain. Do not add punctuation.

Question: {state["query"]}
Answer:"""

    decision = llm.invoke(prompt).content.strip().upper()
    print(f"Router decision: {decision}")

    if "RETRIEVE" in decision:
        return {"next": "rewrite"}

    return {"next": "direct_answer"}


# ──────────────────────────────────────────
# DATE TOOL NODE
# ──────────────────────────────────────────
def date_tool_node(state: dict) -> dict:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    response = f"Current date and time: {now}"

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["query"]),
        AIMessage(content=response)
    ]

    return {
        "final_response": response,
        "messages": new_messages,
        "next": "end"
    }


# ──────────────────────────────────────────
# WEB SEARCH NODE
# ──────────────────────────────────────────
def web_search_node(state: dict) -> dict:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        response = client.search(state["query"], max_results=3, include_answer=True)

        snippets = []
        for i, r in enumerate(response["results"], 1):
            snippet = (
                f"[{i}] {r['title']}\n"
                f"   URL: {r['url']}\n"
                f"   {r['content'][:200]}..."
            )
            snippets.append(snippet)

        tool_response = "Web search results:\n" + "\n\n".join(snippets)
        if response.get("answer"):
            tool_response = f"Summary: {response['answer']}\n\n" + tool_response

    except Exception as e:
        tool_response = f"Web search unavailable: {str(e)}"

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["query"]),
        AIMessage(content=tool_response)
    ]

    return {
        "final_response": tool_response,
        "messages": new_messages,
        "next": "end"
    }


# ──────────────────────────────────────────
# REWRITE NODE
# ──────────────────────────────────────────
def rewrite_node(state: dict) -> dict:
    prompt = f"""You are an expert at reformulating questions for 
semantic search in a UK banking regulatory compliance database.

Rewrite the question to:
- Use precise regulatory terminology (FCA, PRA, Basel, AML etc)
- Be a clear standalone sentence
- Match language found in official regulatory documents
- Length: 1-3 sentences

Output ONLY the rewritten question, nothing else.

Original: {state["query"]}
Rewritten:"""

    rewrite_query = llm.invoke(prompt).content.strip()
    print(f"Rewritten query: {rewrite_query}")
    return {"rewrite_query": rewrite_query, "next": "hyde"}


# ──────────────────────────────────────────
# HYDE NODE
# ──────────────────────────────────────────
def hyde_node(state: dict) -> dict:
    prompt = f"""You are a UK banking regulatory compliance expert.

Write a detailed hypothetical answer (3-6 sentences) to this question
as if it came from an official FCA or PRA regulatory document.
Use precise regulatory language and terminology.
Do NOT say "I don't know".

Question: {state["rewrite_query"]}
Hypothetical regulatory answer:"""

    hyde_answer = llm.invoke(prompt).content.strip()
    print("HyDE answer generated")
    return {"hyde_answer": hyde_answer, "next": "retrieve"}


# ──────────────────────────────────────────
# RETRIEVE NODE
# ──────────────────────────────────────────
def retrieve_node(state: dict) -> dict:
    chunks = hybrid_retrieve(
        query=state["query"],
        rewrite_query=state.get("rewrite_query", state["query"]),
        hyde_answer=state.get("hyde_answer", state["query"])
    )

    context = "\n\n".join(
        f"[Chunk {i}] {chunk}"
        for i, chunk in enumerate(chunks)
    )

    print(f"Retrieved {len(chunks)} chunks")
    return {
        "retrieved_chunks": chunks,
        "context": context,
        "next": "summarize_history"
    }


# ──────────────────────────────────────────
# SUMMARIZE HISTORY NODE
# ──────────────────────────────────────────
def summarize_history_node(state: dict) -> dict:
    messages = state.get("messages", [])
    summarized = summarize_history(messages, llm)
    return {
        "messages": summarized,
        "context": state.get("context", ""),
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "next": "generate"
    }


# ──────────────────────────────────────────
# GENERATE NODE
# ──────────────────────────────────────────
def generate_node(state: dict) -> dict:
    history_text = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in state.get("messages", [])
    ])

    prompt = f"""You are a UK Banking Regulatory Compliance Assistant.
You help compliance officers, analysts, and bank staff understand
FCA and PRA regulations accurately.

Previous conversation:
{history_text}

Relevant regulatory context:
{state.get("context", "No context available.")}

Question: {state["query"]}

Instructions:
- Answer using ONLY the regulatory context provided
- Cite which chunk the information came from
- Be precise and professional
- If the answer is not in the context, say "I cannot find 
  this in the available regulatory documents"
- Never make up regulatory rules"""

    response = llm.invoke(prompt).content
    print("Response generated")

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["query"]),
        AIMessage(content=response)
    ]

    return {
        "final_response": response,
        "messages": new_messages,
        "next": "compliance_check"
    }


# ──────────────────────────────────────────
# COMPLIANCE CHECK NODE
# ──────────────────────────────────────────
def compliance_check_node(state: dict) -> dict:
    # ── Only run on document queries ──
    if not state.get("document_text"):
        print("Skipping compliance check — chat query not document")
        return {**state, "next": "critique"}
    return {**state, "next": "critique"}


# ──────────────────────────────────────────
# CRITIQUE NODE
# ──────────────────────────────────────────
def critique_node(state: dict) -> dict:
    prompt = f"""You are a strict quality checker for regulatory compliance answers.

Evaluate this answer:
- Is it faithful to the context? (no made up regulations)
- Is it complete? (covers the question fully)
- Is it professional and clear?

Question: {state["query"]}
Answer: {state["final_response"]}
Context: {state.get("context", "")[:1500]}

Reply with exactly:
YES - one sentence reason
or
NO - one sentence reason"""

    result = llm.invoke(prompt).content.strip()
    lines = result.split("\n", 1)
    verdict = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""

    is_good = "YES" in verdict
    print(f"Critique: {verdict} — {reason}")

    if is_good:
        save_to_cache(state["query"], state["final_response"])
        return {"next": "end"}

    return {"critique": reason, "next": "retry"}


# ──────────────────────────────────────────
# RETRY NODE
# ──────────────────────────────────────────
def retry_node(state: dict) -> dict:
    reason = state.get("critique", "Answer was incomplete")
    history_text = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in state.get("messages", [])
    ])

    # Retrieve more chunks for retry
    chunks = hybrid_retrieve(
        query=state["query"],
        rewrite_query=state.get("rewrite_query", state["query"]),
        hyde_answer=state.get("hyde_answer", state["query"])
    )
    context = "\n\n".join(
        f"[Chunk {i}] {chunk}"
        for i, chunk in enumerate(chunks)
    )

    prompt = f"""Previous answer was rejected because: {reason}

Using this improved regulatory context, provide a better answer.
Stick strictly to the context — never invent regulations.
If the answer truly cannot be found, say so clearly.

Context:
{context}

Question: {state["query"]}
Improved answer:"""

    new_response = llm.invoke(prompt).content
    save_to_cache(state["query"], new_response)

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["query"] + " (retry)"),
        AIMessage(content=new_response)
    ]

    return {
        "final_response": new_response,
        "messages": new_messages,
        "next": "end"
    }


# ──────────────────────────────────────────
# DOCUMENT COMPLIANCE NODE
# ──────────────────────────────────────────
def document_compliance_node(state: dict) -> dict:
    document_text = state.get("document_text", "")

    if not document_text:
        return {
            **state,
            "final_response": "No document provided.",
            "next": "end"
        }

    prompt = f"""You are a UK banking compliance auditor.
Analyse this document for FCA, PRA, GDPR and AML issues.

You MUST return valid JSON only. No markdown. No text before or after.
Start your response with {{ and end with }}

{{
  "summary": "what this document is about",
  "risk_level": "High",
  "violations": [
    {{
      "type": "name of regulation breached",
      "description": "explain the issue clearly",
      "severity": "High",
      "text_span": "copy exact text from document",
      "recommendation": "how to fix this"
    }}
  ],
  "recommendations": [
    "general recommendation 1",
    "general recommendation 2"
  ]
}}

DOCUMENT:
{document_text[:3000]}

JSON response:"""

   
    result = llm.invoke(prompt).content.strip()
    print("="*50)
    print("RAW RESPONSE:")
    print(result)
    print("="*50)
    print(f"Raw LLM response:\n{result[:300]}")

    try:
        # Find JSON start and end
        start = result.index("{")
        end = result.rindex("}") + 1
        clean = result[start:end]
        parsed = json.loads(clean)
        print("JSON parsed successfully")
    except Exception as e:
        print(f"JSON parse failed: {e}")
        print(f"Raw: {result[:500]}")
        parsed = {
            "summary": "Document analysed",
            "risk_level": "High",
            "violations": [
                {
                    "type": "Compliance Review Required",
                    "description": "Document requires manual compliance review",
                    "severity": "High",
                    "text_span": document_text[:100],
                    "recommendation": "Submit to compliance team for full review"
                }
            ],
            "recommendations": [
                "Manual compliance review recommended",
                "Consult FCA handbook for applicable rules"
            ]
        }

    risk = parsed.get("risk_level", "High")
    violations = len(parsed.get("violations", []))

    return {
        **state,
        "compliance": parsed,
        "final_response": f"Compliance analysis complete. Risk Level: {risk}. Issues found: {violations}",
        "next": "end"
    }
# ──────────────────────────────────────────
# DIRECT ANSWER NODE
# ──────────────────────────────────────────
def direct_answer_node(state: dict) -> dict:
    return {
        "final_response": (
            "I cannot find relevant information in the "
            "regulatory database for this query."
        ),
        "next": "end"
    }


# ──────────────────────────────────────────
# END NODE
# ──────────────────────────────────────────
def end_node(state: dict) -> dict:
    return state