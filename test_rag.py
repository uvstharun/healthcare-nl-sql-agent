import chromadb
from chromadb.utils import embedding_functions

def search_guidelines(query: str, n_results: int = 3) -> str:
    """Search clinical guidelines for relevant chunks."""
    client = chromadb.PersistentClient(path="docs/chroma_db")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_collection(
        name="clinical_guidelines",
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Format results
    output = ""
    for i, (doc, metadata) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        output += f"\n--- Source: {metadata['source']} (chunk {metadata['chunk']}) ---\n"
        output += doc + "\n"

    return output


if __name__ == "__main__":
    print("=== Test 1: Diabetes and Metformin ===")
    print(search_guidelines("Metformin treatment for type 2 diabetes"))

    print("\n=== Test 2: Statins and cholesterol ===")
    print(search_guidelines("statin therapy for high cholesterol LDL"))

    print("\n=== Test 3: Diet and diabetes management ===")
    print(search_guidelines("dietary recommendations for diabetes patients"))