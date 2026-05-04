
import numpy as np
import faiss
import os
from transformers import AutoTokenizer, AutoModel
import torch
import requests

# EMBEDDING MODEL
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()[0]

# LOAD DATA 
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

index = faiss.read_index(os.path.join(base_dir, "data", "knowledge_base_index.faiss"))
knowledge = np.load(os.path.join(base_dir, "data", "knowledge.npy"), allow_pickle=True).tolist()

# response using ollama phi model
def resp_phi(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return ""
 
# Main function     (will be used for streamlit UI)
def get_diagnosis(query):

    # embedding + retrieval
    query_vector = get_embedding(query).astype("float32").reshape(1, -1)
    k = 1
    distances, indices = index.search(query_vector, k)
    results = []

    for i in indices[0]:
        entry = knowledge[i]

        if "| Source:" in entry:
            content, source = entry.split("| Source:")
        else:
            content, source = entry, "Not defined in knowledge base"

        results.append({
            "content": content.strip(),
            "source": source.strip()
        })
    
    context = "\n".join([r["content"] for r in results])

    # retrieved_text = "\n".join([knowledge[i] for i in indices[0]])
    
    prompt = f"""
    You are an automotive diagnostic assistant.

    Based on the context, determine the MOST LIKELY issue and provide a concise diagnosis.

    Context:
    {context}

    User Query:
    {query}

    
    Rules:
    - Choose ONE best diagnosis
    - Ignore less relevant possibilities
    - Do NOT copy sentences from context
    - Do NOT provide any additional information beyond the diagnosis
    ` Do NOT include extra text or example
    - Keep answers concise (max 1 line each)

    Give Final Answer ONLY in the following format:

    Diagnosis:
    Cause:
    Recommended Action:
    """
    
    response = resp_phi(prompt)
    
    return response , results





#for testing/ learning purpose
#  QUERY 
# query = "Engine temperature is high and coolant level is low"
# query = "temperature sensor values are changing frequently and irregularly"

# # RETRIEVAL 
# query_vector = get_embedding(query).astype("float32").reshape(1, -1)
# k = 1
# distances, indices = index.search(query_vector, k)

# retrieved_text = "\n".join([knowledge[i] for i in indices[0]])



# #  PROMPT 
# prompt = f"""
# You are an automotive diagnostic expert.

# Based on the context, determine the MOST LIKELY issue.

# Context:
# {retrieved_text}

# User Query:
# {query}

# Think internally, but DO NOT show reasoning.

# Then give ONLY the final answer.

# Rules:
# - Choose ONE best diagnosis
# - Ignore less relevant possibilities
# - Do NOT copy sentences from context
# - Keep answers concise (max 1 line each)

# Give Final Answer in the following format:

# Diagnosis:
# Cause:
# Recommended Action:
# """

# #GENERATE RESPONSE 
# response = resp_phi(prompt)

# print("\n--- FINAL OUTPUT ---\n")
# print(response)

