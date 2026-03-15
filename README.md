# RAG video transcripts

## Quickstart

### Fetch transcripts from YouTube video IDs
```bash
poetry run python fetch_transcripts.py cFMhLbLSpz8 4xq6bVbS-Pw RVB3PBPxMWg ZoGH7d51bvc hmo2uQbpdbI tatogXG-Who vmOMdY1Ia-M
```

### Populate the vector database
```bash
poetry run python populate_database.py
```

### Perform a query
```bash
poetry run python rag.py --llm-backend openai "Quelles sont les trois grandes étapes de l’apprentissage décrites d’un point de vue neuroscientifique ?"
```

### Run the app
```bash
poetry run streamlit run app.py
```
