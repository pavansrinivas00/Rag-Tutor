"""Command-line RAG tutor over notes.md using Chroma retrieval."""

from pathlib import Path
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

import chromadb
import joblib
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "notes"
VECTORIZER_FILE = CHROMA_DIR / "tfidf_vectorizer.joblib"
TOP_K = 3
MIN_KEYWORD_OVERLAP = 1
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.getenv("RAG_TUTOR_OPENAI_MODEL", "gpt-4o-mini")

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

    # Avoid answering when the retrieved context does not contain question keywords.
    if not has_keyword_overlap(question, context_block, MIN_KEYWORD_OVERLAP):
        return "I don't know based on the notes."

    # Keep prompt construction explicit and inspectable for grounding.
    prompt_messages = ANSWER_PROMPT.format_messages(question=question, context=context_block)

    llm_answer = answer_with_openai(prompt_messages)
    if llm_answer:
        return llm_answer

    # Fallback to extractive behavior when no LLM credentials are available.
    return "\n\n".join(contexts[:2])


def answer_with_openai(prompt_messages: list) -> str | None:
    """Use OpenAI chat completions to produce a readable grounded answer."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "messages": [{"role": msg.type, "content": msg.content} for msg in prompt_messages],
    }
    data = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(
        OPENAI_CHAT_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        return None

    choices = payload.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content", "")
    content = content.strip()
    return content or None


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


def ask_question(question: str, collection: chromadb.Collection, vectorizer) -> tuple[str, list[dict]]:
    """Retrieve contexts and format a grounded response for one question."""
    query_embedding = vectorizer.transform([question]).toarray()[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]
    answer = answer_from_context(question, contexts)
    return answer, metadatas


def print_response(question: str, collection: chromadb.Collection, vectorizer) -> None:
    """Print tutor answer and sources for one user question."""
    answer, metadatas = ask_question(question, collection, vectorizer)

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

    # Batch question mode.
    if args.question:
        for question in args.question:
            if not question.strip():
                continue
            print(f"\nYou: {question.strip()}")
            print_response(question.strip(), collection, vectorizer)
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
            print_response(question, collection, vectorizer)
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

        print_response(question, collection, vectorizer)


if __name__ == "__main__":
    main()
