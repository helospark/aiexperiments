#!/usr/bin/env python3

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Configuration
MODEL_NAME = "google/embeddinggemma-2"

IMAGE_DIR = Path.home() / "Pictures"
#IMAGE_DIR = Path.home() /  "Music"
CHROMA_DIR = Path(".image_chroma_db")
COLLECTION_NAME = "images"

# Image formats
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff",
}

# Animated GIFs and video formats
VIDEO_EXTENSIONS = {
    ".gif", ".mp4", ".mov", ".avi",
    ".mkv", ".m4v", ".mpeg", ".mpg",
}

# Animated GIFs and video formats
SOUND_EXTENSIONS = {
    ".mp3", ".webm", ".ogg"
}



def main():
    if not IMAGE_DIR.is_dir():
        print(f"ERROR: Directory not found: {IMAGE_DIR.resolve()}")
        return

    # Find all supported media files recursively.
    files = sorted(
        path.resolve()
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS | SOUND_EXTENSIONS # | VIDEO_EXTENSIONS
        and not "automatic_output" in str(path)
    )

    if not files:
        print(f"No supported media files found in {IMAGE_DIR.resolve()}")
        return


    # Load model and ChromaDB.
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    files = [path for path in files if str(path) not in existing_ids]
    print(f"Found {len(files)} media files.")

    indexed = 0
    failed = 0

    for i, path in enumerate(files, start=1):
        extension = path.suffix.lower()

        if extension in VIDEO_EXTENSIONS:
            media_type = "video"
            model_input = {"video": str(path)}
            processing_kwargs = {
                "video": {
                    "fps": 1,
                    "max_frames": 16,
                    "overflow_strategy": "uniform",
                    "add_timestamps": True,
                }
            }
        elif extension in IMAGE_EXTENSIONS:
            media_type = "image"
            model_input = {"image": str(path)}
            processing_kwargs = {
                "image": {"max_soft_tokens": 70}
            }
        elif extension in SOUND_EXTENSIONS:
            media_type = "audio"
            model_input = {"audio": str(path)}
            processing_kwargs = {
            }
        print(f"[{i}/{len(files)}] {media_type}: {path}")

        try:
            # Pass the file path directly to EmbeddingGemma 2.
            embedding = model.encode(
                model_input,
                convert_to_numpy=True,
                show_progress_bar=False,
                processing_kwargs=processing_kwargs,
            )

            # Use the absolute path as a stable, unique ID.
            file_id = str(path)

            collection.upsert(
                ids=[file_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "path": str(path),
                    "filename": path.name,
                    "media_type": media_type,
                }],
            )

            indexed += 1

        except Exception as exc:
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print("\nIndexing complete.")
    print(f"Successfully indexed: {indexed}")
    print(f"Failed:               {failed}")
    print(f"Collection size:      {collection.count()}")
    print(f"Database directory:   {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
