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


class ValidatorAgent:
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
        text = state.get("text", "")
        
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
              "content":'''you are a specialized compliance validation agent tasked with reviewing proposed accounting treatments or journal entries submitted by the Enhancer Agent. Your evaluation focuses on Islamic finance principles and the AAOIFI Financial Accounting Standards. Your task is to assess each proposed update for: - Compliance with core Shariah principles (e.g., prohibition of riba, gharar, unjust enrichment) 
- Consistency with the scope, requirements, and guidance of other relevant AAOIFI FAS standards - Risk of misinterpretation or ambiguity in practical implementation Use the vector database connected to you, which contains authoritative content from all AAOIFI FAS standards, Shariah rulings, and related interpretations. Instructions: - please return also the input with the final output - Mark as "Valid" only if the proposed update fully complies with AAOIFI and Islamic finance rules - Mark as "Needs Revision" if it generally complies but requires clarification, better alignment, or minor changes - Mark as "Reject" if the proposal contradicts any AAOIFI standard or fundamental Shariah principle - Your reasoning must be evidence-based, using semantic search from the AAOIFI corpus - Never guess or improvise standards. If no clear standard applies, say so and explain. Your final goal is to help ensure that accounting treatments used in Islamic financial institutions are Shariah-compliant, consistent, and clearly interpretable. using the resources provided. If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure" and stop after that. Refuse to answer any question not about the info. Never break character. ------------ {context} ------------'''
            },
         {
            "role": "user",
            "content": text
         }
        ])
       

        # Update state
        state["validation"] = response.choices[0].message.content
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