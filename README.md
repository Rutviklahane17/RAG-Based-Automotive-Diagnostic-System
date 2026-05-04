# RAG-Based Automotive Diagnostic System

A Retrieval-Augmented Generation (RAG) system that analyzes vehicle issues provided by user using a structured knowledge base and provides context (from knowledge base) and source-grounded diagnostic insights.

## Project Motivation

This project was built to gain hands-on experience with Retrieval-Augmented Generation (RAG) systems and Large Language Models in a practical, domain-specific context.

The goal was to understand how to:
+ Structure and process a domain-specific knowledge base
+ Build a vector database using FAISS
+ Integrate embedding models for semantic retrieval
+ Generate grounded responses using a local LLM (Ollama)
+ Design an interactive UI for querying (using Streamlit)

The project focuses on applying these concepts to automotive diagnostics, demonstrating how AI systems can assist in interpreting technical information and providing explainable outputs with source references.




## Features

- RAG pipeline using FAISS
- Local LLM via Ollama (phi model)
- Source-grounded responses
- Streamlit UI
- Automotive diagnostic knowledge base



## How it works

1. User enters issue description
2. Query is embedded using transformer model
3. FAISS retrieves relevant knowledge
4. LLM generates diagnosis
5. Sources are displayed for traceability



## Prerequisites 
 - python 3.9+
 - ollama (https://ollama.com)

## Installation
1. clone the repository

2. Install dependencies:
'''bash
 pip install -r requirements.txt

3. Run Ollama and start the model:
'''bash
    ollama run phi

4. Run the app
'''bash
streamlit run vd_rag_app.py

