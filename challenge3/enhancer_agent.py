from together import Together
import faiss
import os
import pickle
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
INDEX_PATH = "faiss_index\index.faiss"
METADATA_PATH = "faiss_index\index.pkl"

class EnhancerAgent:
    def __init__(self):
        # Load resources once during initialization
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorstore =faiss.read_index(INDEX_PATH)
        self.metadata = self.load_metadata()
        
        try:
            with open("faiss_index/index.pkl", "rb") as f:
                self.documents = pickle.load(f)
        except FileNotFoundError:
            self.documents = []

    def __call__(self, state: dict) -> dict:
        """Process the state in LangGraph's workflow"""
        text = state.get("text", "")  # Changed to match initial state key
        
        # Vector similarity search
        search_results = self.search(text, k=100)
        context = "\n\n".join([text for text, _ in search_results])

        client = Together()
        # Generate review
        response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
         messages=[
            {
              "role":"system",
              "content":'''You are the AAOIFI Enhancer Agent. Based on the extracted content from the Reviewer Agent, propose updates, clarifications, or expansions to the standard. Justify each proposal with:
- be creative
- Why the change is needed
- What modern context or use-case it addresses
- A short reference to similar treatment in IFRS or Shariah reasoning

Present your suggestions in bullet points with clear headers: [Proposed Change], [Reason], [Reference].
Using the provided context, answer the user's question to the best of your ability using the resources provided.
be creative at the limits of islamic finance. Refuse to answer any question not about the info. Never break character.
------------

{context}

------------'''
            },
         {
            "role": "user",
            "content": text
         }
        ])
        


        # Update and return state
        print('----------------------Enhancement----------------------')
        print(response.choices[0].message.content)
        state["enhancement"] = response.choices[0].message.content
        return state
    def embed_query(self,query: str) -> np.ndarray:
     embedding = self.model.encode([query])
     return np.array(embedding).astype("float32")
     
    def load_metadata(self) -> List[str]:
      with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)


    def search(self,query: str, k: int = 5) -> List[Tuple[str, float]]:
     
     query_vector = self.embed_query(query)

     distances, indices = self.vectorstore.search(query_vector, k)
     results = []
     for i, dist in zip(indices[0], distances[0]):
        if i < len(self.metadata):
            results.append((self.metadata[i], dist))
     return results