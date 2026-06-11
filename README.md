# RAG video transcripts

## Quickstart

### Fetch transcripts from YouTube video IDs
```bash
uv run python -m rag_app.fetch_transcripts cFMhLbLSpz8 4xq6bVbS-Pw RVB3PBPxMWg ZoGH7d51bvc hmo2uQbpdbI tatogXG-Who vmOMdY1Ia-M
```

### Populate the vector database
```bash
uv run python -m rag_app.populate_database
```

### Perform a query
```bash
uv run python -m rag_app.rag --llm-backend openai "Quelles sont les trois grandes étapes de l’apprentissage décrites d’un point de vue neuroscientifique ?"
```

### Run the app
```bash
make run
```
