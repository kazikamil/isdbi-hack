import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

INDEX_PATH = "faiss_index\index.faiss"
METADATA_PATH = "faiss_index\index.pkl"

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index() -> faiss.Index:
    return faiss.read_index(INDEX_PATH)

def load_metadata() -> List[str]:
    with open(METADATA_PATH, "rb") as f:
        return pickle.load(f)

def embed_query(query: str) -> np.ndarray:
    embedding = model.encode([query])
    return np.array(embedding).astype("float32")

def search(query: str, k: int = 100) -> List[Tuple[str, float]]:
    index = load_index()
    metadata = load_metadata()
    query_vector = embed_query(query)

    distances, indices = index.search(query_vector, k)
    print(index)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(metadata):
            results.append((metadata[i], dist))
    return results

# Run from CLI
if __name__ == "__main__":
    query = input("🔍 Entrez votre question AAOIFI : ")
    results = search(query, k=100)
    
    print("\n📚 Résultats les plus pertinents :\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"[{i}] Score: {score:.2f}")
        print(chunk)
        print("-" * 80)
