# Rag-Tutor
My first LangChain project

## Chat tutor

Run interactive mode:

```bash
python chat.py
```

Ask specific questions directly:

```bash
python chat.py -q "what is Cache stampede ?" -q "what is rate limiting"
```

### Optional LLM-formatted answers

`chat.py` now supports polished LLM-generated responses (still grounded on retrieved notes)
when `OPENAI_API_KEY` is configured.

```bash
export OPENAI_API_KEY="..."
python chat.py -q "what is Cache stampede ?"
```

Optional model override:

```bash
export RAG_TUTOR_OPENAI_MODEL="gpt-4o-mini"
```

Without an API key, it falls back to extractive answers from the notes.
