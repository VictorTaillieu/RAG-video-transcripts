import argparse
import json
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
# from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
# 800/80 800/150 1000/150
CHROMA_PATH = "chroma_db"


def load_documents() -> list[Document]:
    """
    Load documents from the "data" directory.
    """
    documents = []

    for filename in Path("data").glob("*.json"):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        document = Document(
            page_content=" ".join([entry["text"] for entry in data["transcript"]]),
            metadata={
                "title": data["title"],
                "creator": data["creator"],
                "source": filename.stem
            }
        )

        documents.append(document)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


def embedding_function():
    """
    Create and return the embedding function.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


def add_to_chroma(chunks: list[Document]):
    """
    Add new chunks to the Chroma database.
    """
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function()
    )

    chunks_with_prefix = [prefix_document(chunk) for chunk in chunks]
    chunks_with_ids = calculate_chunk_ids(chunks_with_prefix)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents...")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("No new documents to add")


def prefix_document(document: Document) -> Document:
    """
    Prefix a document content with "passage: ".
    """
    return Document(
        page_content="passage: " + document.page_content,
        metadata=document.metadata
    )


def calculate_chunk_ids(chunks):
    """
    Calculate unique IDs for each chunk based on its source and index.
    """
    last_source = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")

        if source == last_source:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{source}:{current_chunk_index}"
        last_source = source

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Clear the Chroma database.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear the database.")
    args = parser.parse_args()
    if args.clear:
        print("Clearing Database...")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
