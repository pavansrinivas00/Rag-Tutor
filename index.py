"""Index notes.md into a local Chroma DB for a minimal RAG tutor."""

from pathlib import Path
import shutil

import chromadb
import joblib
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer


NOTES_FILE = Path("notes.md")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notes"
VECTORIZER_FILE = CHROMA_DIR / "tfidf_vectorizer.joblib"


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into simple overlapping chunks."""
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += step

    return [chunk for chunk in chunks if chunk]


def main() -> None:
    """Load notes, chunk, embed, and store in Chroma."""
    if not NOTES_FILE.exists():
        raise FileNotFoundError(f"Could not find {NOTES_FILE}")

    raw_text = NOTES_FILE.read_text(encoding="utf-8")
    chunks = split_text(raw_text)
    if not chunks:
        raise ValueError("notes.md is empty after splitting")

    # Use LangChain Document objects to keep content + metadata structured.
    documents = [
        Document(page_content=chunk, metadata={"source": str(NOTES_FILE), "chunk": i})
        for i, chunk in enumerate(chunks)
    ]

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Local embeddings using TF-IDF (no network/model downloads required).
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([doc.page_content for doc in documents])

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(documents))],
        documents=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents],
        embeddings=matrix.toarray().tolist(),
    )

    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Indexed {len(documents)} chunks into {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
