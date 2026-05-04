# reads knowledge file, covert each ruleinto embedding, stores the embeddings to FAISS database

import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
# from openai import OpenAI

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
file_path = os.path.join(base_dir, "data", "knowledge_base.txt")
print("Looking for  file at:", file_path)

# initialize openAI client   - if using openAI API
# client = OpenAI(api_key= "apiKey")

#i am using HuggingFace embedding model instead of openAI API for embeddings
#load model + tokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to get embedding

def get_embedding(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)


    # Take mean of token embeddings (simple approach)

    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.detach().numpy()[0]


# load knowledge base file
with open(file_path, "r") as file:
    knowledge = file.read().split("\n\n")

# clean the text - remove empty lines /white spaces
knowledge = [item.strip() for item in knowledge if item.strip()]

if not knowledge:
    raise ValueError("Knowledge base file is empty or missing content.")

# create embeddings for each rule in the knowledge base
embeddings = []   
for rule in knowledge:
    embed_response = get_embedding(rule)  
    embeddings.append(embed_response)

# convert list of embeddings to numpy array
embedding_matrix = np.array(embeddings).astype('float32')

#create FAISS index
dimension = embedding_matrix.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)  # type: ignore # add embeddings to the index


# save_path = os.path.join(base_dir, "data", "knowledge_base_index.faiss")
# print("Saving index at:", save_path)
#save index
faiss.write_index(index, os.path.join(base_dir, "data", "knowledge_base_index.faiss"))

# save the original knowledge base seperately for later retrieval 
np.save(os.path.join(base_dir, "data", "knowledge.npy"), np.array(knowledge))
print("Knowledge base embeddings created and stored in FAISS index successfully.\n vector database created")





