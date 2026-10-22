from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-2")

query = "What causes the northern lights?"

document = (
    "The northern lights are caused by charged "
    "particles from the Sun interacting with Earth's atmosphere."
)

query_embedding = model.encode(
    query,
    prompt_name="SearchQuery"
)

document_embedding = model.encode(
    document,
    prompt_name="Document"
)

similarity = model.similarity(
    query_embedding,
    document_embedding
)

print("Similarity:", similarity.item())
