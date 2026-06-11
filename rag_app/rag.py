import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from rag_app.config import CHROMA_PATH
from rag_app.populate_database import embedding_function

load_dotenv()

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = SecretStr(os.environ["OPENROUTER_API_KEY"])
PROMPT_TEMPLATE = """Tu es un assistant précis et factuel.

Règles :
- Réponds uniquement à partir du CONTEXTE.
- N'invente rien.
- Si l'information n'est pas dans le contexte, réponds : "Je ne sais pas".

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""


def query_rag(
    query_text: str, llm_backend: str
) -> tuple[str, list[tuple[Document, float]]]:
    """
    Query the RAG system and return the response.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())

    results = db.similarity_search_with_score("query: " + query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content.removeprefix("passage: ") for doc, _ in results]
    )
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

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
            base_url=OPENROUTER_API_BASE,
            api_key=OPENROUTER_API_KEY,
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
