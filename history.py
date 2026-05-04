import os
import json
from langchain_core.messages import HumanMessage, AIMessage
from config import HISTORY_FILE


def load_history() -> list:
    """Load conversation history from disk."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                messages = []
                for msg in data:
                    if msg["type"] == "human":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "ai":
                        messages.append(AIMessage(content=msg["content"]))
                print(f"Loaded {len(messages)} messages from history")
                return messages
        except Exception as e:
            print("History load failed:", e)
    return []


def save_history(messages: list):
    """Save conversation history to disk."""
    try:
        data = [{"type": msg.type, "content": msg.content} for msg in messages]
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"History saved — {len(messages)} messages")
    except Exception as e:
        print("History save failed:", e)


def clear_history():
    """Wipe conversation history."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    print("History cleared")


def summarize_history(messages: list, llm) -> list:
    """
    If history is too long, summarise it to prevent
    context window overflow.
    """
    from langchain_core.messages import AIMessage

    if len(messages) <= 10:
        return messages

    history_text = "\n".join([
        f"{msg.type.capitalize()}: {msg.content}"
        for msg in messages
    ])

    summary_prompt = f"""Summarise this conversation concisely under 200 words.
    Capture key topics, questions asked, and conclusions reached.
    Be factual and neutral.

    Conversation:
    {history_text}

    Summary:"""

    summary = llm.invoke(summary_prompt).content.strip()
    print("History summarised to prevent overflow")

    return [AIMessage(content=f"Conversation summary: {summary}")]