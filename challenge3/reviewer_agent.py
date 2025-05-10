from together import Together
import faiss
import os
import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index/index.faiss"
METADATA_PATH = "faiss_index/index.pkl"

class ReviewerAgent:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
        self.vectorstore = self.load_faiss_index(INDEX_PATH)
        self.metadata = self.load_metadata(METADATA_PATH)
        # Initialize Together API client
        
    def __call__(self, state: dict) -> dict:
        """Process the state in LangGraph's workflow"""
        text = state.get("text", "")
        
        if not text:  # Handle empty text case
            return {"error": "No text provided for analysis."}

        # Vector similarity search
        search_results = self.search(text, k=100)
        context = "\n\n".join([text for text, _ in search_results])
       
        client = Together()
        # Generate review using Together's API
        response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=[
            {
              "role":"system",
              "content":'''You are the AAOIFI Standard Reviewer Agent. Your task is to extract the core components of the standard provided:
Purpose and scope
Key accounting treatments
Definitions
Exceptions and limitations

Present your output in clearly labeled sections. Do not analyze or critique the content. Do not propose changes.
Using the provided context, answer the user's question to the best of your ability using the resources provided.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure" and stop after that. Refuse to answer any question not about the info. Never break character.
------------

{context}

------------
REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure". Don't try to make up an answer. Never break character.'''
            }
            ,
            
            {
                "role": "user",
                "content": text
            }]
  )
        

        # Update and return state
       
        state["review"] = response.choices[0].message.content
        return state
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generates embedding for the query text"""
        return np.array(self.model.encode([query])).astype("float32")
    
    def load_faiss_index(self, index_path: str) -> faiss.Index:
        """Load the FAISS index"""
        try:
            return faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None

    def load_metadata(self, metadata_path: str) -> List[str]:
        """Load metadata"""
        try:
            with open(metadata_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Metadata file not found.")
            return []

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Perform vector similarity search on the FAISS index"""
        query_vector = self.embed_query(query)
        distances, indices = self.vectorstore.search(query_vector, k)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.metadata):
                results.append((self.metadata[i], dist))
        return results
