import faiss
from together import Together
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
    results = []
    for i, dist in zip(indices[0], distances[0]):
        if i < len(metadata):
            results.append((metadata[i], dist))
    return results

# Run from CLI
if __name__ == "__main__":
    text = input("🔍 Enter the use case : ")
    context = search(text, k=100)
    client = Together()
        # Generate review using Together's API
    response = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-fp8-tput",
        messages=[
            {
              "role":"system",
              "content":'''You are an Islamic Finance accounting assistant. Your task is to analyze lease contract scenarios and produce Shariah compliant accounting entries using Islamic finance rules and AAOIFI standards. You are connected to a vector database containing all AAOIFI standards. Your job: When given a user input describing a lease, extract and compute all relevant data. Perform every calculation for the extracted variables. If the contract is Ijarah Muntahia Bittamleek (Ijarah MBT): Compute the right of use asset as total acquisition cost (purchase plus import tax plus freight) minus the promised purchase price using AAOIFI FAS 23 paragraph 31(c). Compute the retained benefit as expected residual value minus promised purchase price. Calculate the net amortizable amount as right of use asset minus retained benefit. Apply straight line amortization over the Ijarah term using AAOIFI FAS 23 paragraphs 29 and 30. Explain each step and cite the relevant paragraph numbers from FAS 23. For any other Islamic finance operation, draw on the relevant AAOIFI FAS as needed, while remembering that only FAS 23 governs Ijarah MBT amortization. Rules: Use precise Islamic finance terminology and never use conventional finance terms such as interest or discount rate. Provide the final response in valid, well formatted JSON with quoted keys and values and without trailing commas. Do not invent values or make assumptions; if data is missing, state exactly what is missing and stop. If the context lacks relevant AAOIFI guidance, reply “Hmm, I’m not sure.”



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
    print('----------------------Result----------------------')
    print(response.choices[0].message.content)
    

    
