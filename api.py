from fastapi import FastAPI
from graph import app as rag_app

app = FastAPI()

@app.post("/ask")
def ask(payload: dict):
    state = {
        "query": payload["question"],
        "messages": []
    }
    result = rag_app.invoke(state)
    return {"answer": result["final_response"]}