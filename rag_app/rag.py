import argparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from rag_app.config import settings
from rag_app.populate_database import embedding_function
from rag_app.prompts.loader import Prompt, load_prompt


def query_rag(
    query_text: str, llm_backend: str
) -> tuple[str, list[tuple[Document, float]]]:
    """
    Query the RAG system and return the response.
    """
    db = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=embedding_function(),
    )

    results = db.similarity_search_with_score("query: " + query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content.removeprefix("passage: ") for doc, _ in results]
    )
    prompt_template = load_prompt(Prompt.QA)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = select_llm_backend(llm_backend)
    response = str(model.invoke(prompt).content)

    return response, results


def select_llm_backend(llm_backend: str) -> BaseChatModel:
    """
    Select and return the LLM backend based on the provided name.
    """
    if llm_backend == "ollama":
        return ChatOllama(model="mistral", temperature=0)
    elif llm_backend == "openai":
        return ChatOpenAI(
            model="mistralai/mistral-small-3.2-24b-instruct",
            temperature=0,
            base_url=settings.OPENROUTER_API_BASE,
            api_key=settings.OPENROUTER_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported LLM backend: {llm_backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query text.")
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend to use.",
    )
    args = parser.parse_args()

    response, results = query_rag(args.query, args.llm_backend)
    print(response)
