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


class MatcherAgent:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorstore =faiss.read_index(INDEX_PATH)
        self.metadata = self.load_metadata()
        
        # Load metadata
        try:
            with open("faiss_index/index.pkl", "rb") as f:
                self.metadata = pickle.load(f)
        except FileNotFoundError:
            self.metadata = []

    def __call__(self, state: dict) -> dict:
        """Process state for LangGraph"""
        text = state.get("extractor", "")
        
        # Vector similarity search
        search_results = self.search(text, k=100)
        context = "\n\n".join([text for text, _ in search_results])
        # Create validation prompt
        prompt = (
            "Évaluation de conformité AAOIFI\n\n"
            f"Proposition d'amélioration :\n{text}\n\n"
            f"Documents de référence :\n{context}\n\n"
            "Cette proposition est-elle conforme aux normes AAOIFI ? "
            "Identifiez les points forts et les risques potentiels."
        )

        # Get validation from GPT-4
        client = Together()
        # Generate review
        response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
         messages=[
             {
              "role":"system",
              "content":'''ou are a Matcher Agent responsible for matching the extracted financial data to the appropriate AAOIFI FAS (Financial Accounting Standards). Based on the extracted details from the transaction, identify which AAOIFI standards are applicable (e.g., FAS 4 for Ijarah MBT). The goal is to:

Retrieve and reference the most relevant AAOIFI FAS.

Explain how each standard applies to the provided transaction.

If the transaction involves complex or ambiguous terms, use the AAOIFI rules to make reasonable assumptions and clarifications.

For each identified standard, provide the standard number and a brief description of how it applies to the transaction.
Using the provided context, answer the user's question to the best of your ability using the resources provided.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure" and stop after that. Refuse to answer any question not about the info. Never break character.
------------

{context}

------------
REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure". Don't try to make up an answer. Never break character.'''
            },
         {
            "role": "user",
            "content": text
         }
        ])
       

        # Update state
        state["matcher"] = response.choices[0].message.content
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