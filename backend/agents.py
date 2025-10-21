from backend import rag
from huggingface_hub import InferenceClient
import streamlit as st
import numpy as np

HF_TOKEN = st.secrets["HF_TOKEN"]
hf_client = InferenceClient(token=HF_TOKEN)

def smart_chat(prompt: str) -> str:
    query = prompt.strip()
    if not query:
        return "Please type a question."

    # 1️⃣ Retrieve relevant docs
    retrieved_docs = rag.search(query, top_k=5)

    # 2️⃣ If docs are retrieved, try to answer with context
    if retrieved_docs:
        rag_answer = rag.answer_with_context(query, retrieved_docs)
        # If RAG says not enough info, fall back to LLM
        if "Not enough information" in rag_answer:
            pass  # Fall through to LLM
        else:
            return rag_answer

    # 3️⃣ Fallback to LLM only (no context or RAG insufficient)
    try:
        completion = hf_client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant. Provide helpful, accurate information based on general medical knowledge. If the query involves symptoms, advise consulting a healthcare professional for personalized advice."},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"⚠️ Failed to generate answer: {str(e)}"
