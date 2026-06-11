format:
	@uv run ruff format
	@uv run ruff check --fix

lint:
	@uv run ruff format --check
	@uv run ruff check
	@uv run mypy .

run:
	@uv run streamlit run app.py --server.fileWatcherType none
