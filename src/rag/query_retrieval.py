from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import faiss
import os

# Load model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to generate embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()[0]

# Load paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Load FAISS index
index = faiss.read_index(os.path.join(base_dir, "data", "knowledge_base_index.faiss"))

# Load knowledge
knowledge = np.load(os.path.join(base_dir, "data", "knowledge.npy"), allow_pickle=True)

# USER QUERY 
query = "Engine temperature is high and coolant level is low"

# Convert query to embedding
query_vector = get_embedding(query).astype("float32").reshape(1, -1)

# Search in FAISS
k = 3  # number of results
distances, indices = index.search(query_vector, k)

# Print results
print("\nQuery:", query)
print("\nTop matches:\n")

for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}:")
    print(knowledge[idx])
    print("----------")