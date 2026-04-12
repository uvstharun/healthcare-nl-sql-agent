import os
import chromadb
from pypdf import PdfReader
from chromadb.utils import embedding_functions

# -------------------------------------------------------
# STEP 1: LOAD AND EXTRACT TEXT FROM PDFS
# -------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# -------------------------------------------------------
# STEP 2: CHUNK THE TEXT
# Why chunk? LLMs have context limits and embedding models
# work better on smaller focused pieces than entire documents.
# We split into overlapping chunks so no information is lost
# at chunk boundaries.
# -------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap keeps context between chunks

    return chunks


# -------------------------------------------------------
# STEP 3: BUILD THE VECTOR STORE
# ChromaDB stores each chunk as a vector (embedding).
# When we search later, it finds the chunks most
# semantically similar to the query.
# -------------------------------------------------------

def build_vectorstore():
    # Use a local ChromaDB that saves to disk
    client = chromadb.PersistentClient(path="docs/chroma_db")

    # Use sentence-transformers for embeddings (free, runs locally)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("clinical_guidelines")
        print("Cleared existing collection.")
    except:
        pass

    collection = client.create_collection(
        name="clinical_guidelines",
        embedding_function=embedding_fn
    )

    # Process each PDF in the docs folder
    doc_id = 0
    for filename in os.listdir("docs"):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join("docs", filename)
        print(f"\nProcessing: {filename}")

        # Extract text
        text = extract_text_from_pdf(filepath)
        print(f"  Extracted {len(text.split())} words")

        # Chunk it
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        print(f"  Created {len(chunks)} chunks")

        # Add to ChromaDB
        collection.add(
            documents=chunks,
            ids=[f"doc_{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))]
        )
        doc_id += 1
        print(f"  Added to vector store")

    total = collection.count()
    print(f"\nVector store built. Total chunks stored: {total}")


if __name__ == "__main__":
    build_vectorstore()