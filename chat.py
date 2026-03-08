"""Command-line RAG tutor over notes.md using Chroma retrieval."""

from pathlib import Path
import argparse
import os
import re
import sys

import chromadb
import joblib
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notes"
VECTORIZER_FILE = CHROMA_DIR / "tfidf_vectorizer.joblib"
TOP_K = 3
MIN_KEYWORD_OVERLAP = 1
DEFAULT_MODEL = "gpt-4o-mini"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are a tutor. Use only the facts in CONTEXT to answer QUESTION.
Return 2-4 concise sentences tailored to the question.
Do not quote the context verbatim unless needed.
If context is insufficient, say exactly: "I don't know based on the notes.".

QUESTION:
{question}

CONTEXT:
{context}
"""
)


def build_chat_model():
    """Build an optional chat model for natural-language answer formatting."""
    model_name = os.getenv("RAG_TUTOR_MODEL", DEFAULT_MODEL)
    try:
        return init_chat_model(model_name)
    except Exception:
        return None


def format_context_fallback(context: str) -> str:
    """Create a concise non-verbatim answer when no LLM is available."""
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("#")]

    statement_lines = [
        line for line in lines if not line.startswith("-") and not line.endswith(":")
    ]
    bullet_lines = [line[1:].strip() for line in lines if line.startswith("-")]

    parts = []
    if statement_lines:
        parts.append(statement_lines[0])
    if len(statement_lines) > 1:
        parts.append(statement_lines[1])
    if bullet_lines:
        parts.append("Mitigations include " + ", ".join(bullet_lines) + ".")

    return " ".join(parts).strip() or "I don't know based on the notes."


def answer_from_context(question: str, contexts: list[str], chat_model) -> str:
    """Create a grounded answer string from retrieved contexts only."""
    if not contexts:
        return "I don't know based on the notes."

    context_block = "\n\n---\n\n".join(contexts)

    # Avoid answering when the retrieved context does not contain question keywords.
    if not has_keyword_overlap(question, context_block, MIN_KEYWORD_OVERLAP):
        return "I don't know based on the notes."

    messages = ANSWER_PROMPT.format_messages(question=question, context=context_block)

    if chat_model is not None:
        try:
            response = chat_model.invoke(messages)
            if response and getattr(response, "content", ""):
                return str(response.content).strip()
        except Exception:
            pass

    # Fallback if no LLM is configured/reachable.
    return format_context_fallback(contexts[0])


def tokenize(text: str) -> set[str]:
    """Convert text into normalized tokens."""
    return {token for token in re.findall(r"[a-zA-Z]+", text.lower())}


def has_keyword_overlap(question: str, context: str, min_overlap: int) -> bool:
    """Check whether question keywords appear in retrieved context."""
    question_tokens = tokenize(question) - STOPWORDS
    if not question_tokens:
        return True

    context_tokens = tokenize(context)
    overlap = question_tokens & context_tokens
    return len(overlap) >= min_overlap


def ask_question(question: str, collection: chromadb.Collection, vectorizer, chat_model) -> tuple[str, list[dict]]:
    """Retrieve contexts and format a grounded response for one question."""
    query_embedding = vectorizer.transform([question]).toarray()[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]
    answer = answer_from_context(question, contexts, chat_model)
    return answer, metadatas


def print_response(question: str, collection: chromadb.Collection, vectorizer, chat_model) -> None:
    """Print tutor answer and sources for one user question."""
    answer, metadatas = ask_question(question, collection, vectorizer, chat_model)

    print("\nTutor:")
    print(answer)

    print("\nSources:")
    for i, metadata in enumerate(metadatas, start=1):
        print(f"{i}. {metadata.get('source', 'unknown')} (chunk {metadata.get('chunk', '?')})")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description="Interactive RAG tutor over notes.md")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        default=[],
        help="Ask a question directly (can be provided multiple times).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read questions line-by-line from stdin.",
    )
    return parser


def main() -> None:
    """Run the tutor in interactive mode or batch modes."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if not CHROMA_DIR.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError("Run `python index.py` first to build chroma_db/")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    vectorizer = joblib.load(VECTORIZER_FILE)
    chat_model = build_chat_model()

    # Batch question mode.
    if args.question:
        for question in args.question:
            if not question.strip():
                continue
            print(f"\nYou: {question.strip()}")
            print_response(question.strip(), collection, vectorizer, chat_model)
        return

    # Piped stdin mode (useful in non-interactive environments).
    if args.stdin or not sys.stdin.isatty():
        print("RAG Tutor stdin mode. Provide one question per line.")
        for line in sys.stdin:
            question = line.strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            print(f"\nYou: {question}")
            print_response(question, collection, vectorizer, chat_model)
        return

    # Interactive TTY mode.
    print("RAG Tutor ready. Ask a question (type 'exit' to quit).")
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        print_response(question, collection, vectorizer, chat_model)


if __name__ == "__main__":
    main()
