#!/usr/bin/env python3

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Configuration
MODEL_NAME = "google/embeddinggemma-2"
DATABASE_DIR = Path(".image_chroma_db")
COLLECTION_NAME = "images"


def search_images(model, collection, query, top_k=10):
    """Search the image database using a text query."""

    if collection.count() == 0:
        print("The image database is empty. Run index_images.py first.")
        return

    # Encode the text query.
    # SearchQuery is the recommended task prompt for text retrieval.
    query_embedding = model.encode(
        query,
        prompt_name="SearchQuery",
        convert_to_numpy=True,
    )

    # Search the database using the query embedding.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not ids:
        print("No matching images found.")
        return

    print(f"\nQuery: {query}")
    print(f"Top {len(ids)} results:\n")

    results = []

    for rank, (image_id, distance, metadata) in enumerate(
        zip(ids, distances, metadatas),
        start=1,
    ):
        # Chroma's cosine distance is 1 - cosine similarity.
        similarity = 1.0 - distance
        media_path = metadata["path"]

        if not Path(media_path).is_file():
            continue

        results.append({
            "path": media_path,
            "similarity": similarity,
        })

    return results


def main():
    # Open the existing persistent database.
    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        print(f"Could not open the image collection: {exc}")
        return

    if not DATABASE_DIR.exists():
        print("Image database not found. Run index_images.py first.")
        return


    # Load the embedding model.
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("Search for images: ").strip()

        if not query:
            print("Please enter a search query.")
            continue

        search_images(
            model=model,
            collection=collection,
            query=query,
            top_k=3,
        )


if __name__ == "__main__":
    main()
