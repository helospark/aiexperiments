#!/usr/bin/env python3

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-2"


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/animation.gif")
        sys.exit(1)

    gif_path = Path(sys.argv[1]).expanduser().resolve()

    if not gif_path.is_file():
        print(f"ERROR: File not found: {gif_path}")
        sys.exit(1)

    if gif_path.suffix.lower() != ".gif":
        print("WARNING: The supplied file does not have a .gif extension.")

    print(f"GIF:   {gif_path}")
    print(f"Size:  {gif_path.stat().st_size / 1024:.1f} KB")
    print(f"Model: {MODEL_NAME}")
    print("\nLoading model...")

    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model loaded on: {model.device}")

        # Try passing the GIF path directly to the video interface.
        print("\nTesting GIF as video input...")
        start = time.perf_counter()

        embedding = model.encode(
            {"video": str(gif_path)},
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start

        print("\nSUCCESS!")
        print("The model accepted the GIF as a video input.")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dtype: {embedding.dtype}")
        print(f"Processing time: {elapsed:.2f} seconds")

        print("\nFirst 10 embedding values:")
        print(embedding[:10])

    except Exception as exc:
        print("\nFAILED!")
        print("The model could not process the GIF through the video interface.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Error message: {exc}")

        print("\nFull traceback:")
        import traceback
        traceback.print_exc()

        sys.exit(2)


if __name__ == "__main__":
    main()
