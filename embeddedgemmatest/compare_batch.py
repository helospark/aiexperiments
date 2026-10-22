import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

# Replace these with 32 real image paths.
paths = [
    
]

inputs = [{"image": path} for path in paths]

for batch_size in [1, 2, 4, 8, 16]:
    start = time.perf_counter()

    embeddings = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        processing_kwargs={"image": {"max_soft_tokens": 70}},
    )

    elapsed = time.perf_counter() - start

    print(
        f"Batch {batch_size:2}: "
        f"{elapsed:.2f}s total, "
        f"{len(paths) / elapsed:.3f} images/s, "
        f"{elapsed / len(paths):.2f}s/image"
    )
