#!/usr/bin/env python3

from flask import Flask, request, render_template
import fitz
import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
API_KEY = os.getenv('API_KEY')

# Print to verify
print(f"API Key: {API_KEY}")  # Should print the key you placed in .env

app = Flask(__name__)

# Load model and data once
embedding_model = SentenceTransformer("ntproctor/mass-academy-faq-embedder")
pdf_path = "newData.txt"  # Make sure this is the correct file name

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

def chunk_text(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def build_faiss_index(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, 7)
    index.train(np.array(embeddings, dtype=np.float32))
    index.add(np.array(embeddings, dtype=np.float32))
    return index, text_chunks

def search_faiss(query, area, index, text_chunks, top_k=3):
    feed = query + " " + area
    query_embedding = embedding_model.encode([feed])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]

def ask_gemini(question, retrieved_text, area):
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
    You are an AI assistant answering based only on the provided text.
    Tailor your responses around the {area}.
    
    Context:
    {retrieved_text}

    Question: {question}

    If the answer isn't in the provided text, say 'I don't know.'
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Load and process data at startup
full_text = extract_text_from_pdf(pdf_path)
text_chunks = chunk_text(full_text)
index, stored_texts = build_faiss_index(text_chunks)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    area = request.form.get("area_of_interest", "").strip()
    question = request.form.get("question", "").strip()

    if not area or not question:
        return render_template("index.html", answer="Please provide both area of interest and question.")

    retrieved = search_faiss(question, area, index, stored_texts)
    answer = ask_gemini(question, " ".join(retrieved), area)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True,)
