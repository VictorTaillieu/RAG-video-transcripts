import argparse
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAI
from pydantic import SecretStr

from rag_app.populate_database import CHROMA_PATH, embedding_function

load_dotenv()

MAMMOUTH_API_BASE = "https://api.mammouth.ai/v1"
MAMMOUTH_API_KEY = SecretStr(os.environ["MAMMOUTH_API_KEY"])
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
    response = model.invoke(prompt)

    return response, results


def select_llm_backend(llm_backend: str) -> OllamaLLM | OpenAI:
    """
    Select and return the LLM backend based on the provided name.
    """
    if llm_backend == "ollama":
        return OllamaLLM(model="mistral", temperature=0)
    elif llm_backend == "openai":
        return OpenAI(
            model="mistral-small-3.2-24b-instruct",
            temperature=0,
            max_tokens=1024,
            base_url=MAMMOUTH_API_BASE,
            api_key=MAMMOUTH_API_KEY,
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
