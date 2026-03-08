"""Command-line RAG tutor over notes.md using Chroma retrieval."""

from pathlib import Path

import chromadb
import joblib
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notes"
VECTORIZER_FILE = CHROMA_DIR / "tfidf_vectorizer.joblib"
TOP_K = 3

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a tutor. Answer only with facts found in CONTEXT.
If context is insufficient, say "I don't know based on the notes.".

QUESTION:
{question}

CONTEXT:
{context}
"""
)


def answer_from_context(question: str, contexts: list[str]) -> str:
    """Create a grounded answer string from retrieved contexts only."""
    if not contexts:
        return "I don't know based on the notes."

    context_block = "\n\n---\n\n".join(contexts)

    # We format a LangChain prompt to keep answer generation constrained/transparent.
    _ = ANSWER_PROMPT.format_messages(question=question, context=context_block)

    # Minimal extractive response: return the most relevant chunk(s) verbatim.
    return "\n\n".join(contexts[:2])


def main() -> None:
    """Run the interactive tutor loop."""
    if not CHROMA_DIR.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError("Run `python index.py` first to build chroma_db/")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    vectorizer = joblib.load(VECTORIZER_FILE)

    print("RAG Tutor ready. Ask a question (type 'exit' to quit).")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        query_embedding = vectorizer.transform([question]).toarray()[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        contexts = results["documents"][0]
        metadatas = results["metadatas"][0]

        print("\nTutor:")
        print(answer_from_context(question, contexts))

        print("\nSources:")
        for i, metadata in enumerate(metadatas, start=1):
            print(f"{i}. {metadata.get('source', 'unknown')} (chunk {metadata.get('chunk', '?')})")


if __name__ == "__main__":
    main()
