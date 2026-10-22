import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

IMAGE_PATH = Path.home() / "Pictures/2024/2024_12_20/20241122_070709.jpg"

model = SentenceTransformer("google/embeddinggemma-2")

for token_budget in [280, 140, 70]:
    # Warm up the model before timing.
    model.encode(
        {"image": IMAGE_PATH},
        processing_kwargs={"image": {"max_soft_tokens": token_budget}},
        show_progress_bar=False,
    )

    start = time.perf_counter()

    for _ in range(3):
        embedding = model.encode(
            {"image": IMAGE_PATH},
            processing_kwargs={"image": {"max_soft_tokens": token_budget}},
            show_progress_bar=False,
        )

    elapsed = time.perf_counter() - start

    print(
        f"Tokens={token_budget:3}: "
        f"{elapsed / 3:.2f}s/image, "
        f"embedding shape={embedding.shape}"
    )
