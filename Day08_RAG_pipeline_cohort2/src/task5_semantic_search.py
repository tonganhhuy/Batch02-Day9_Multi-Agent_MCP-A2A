"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
from sentence_transformers import SentenceTransformer
import chromadb

EMBEDDING_MODEL = "BAAI/bge-m3"

from pathlib import Path
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "rag_documents"

model = SentenceTransformer(EMBEDDING_MODEL)
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    # TODO: Implement semantic search
    #
    # Bước 1: Embed query bằng cùng model ở Task 4
    # Bước 2: Query vector store (cosine similarity)
    # Bước 3: Return top_k results
    #
    # Ví dụ với Weaviate:
    # import weaviate
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer("BAAI/bge-m3")
    # query_embedding = model.encode(query).tolist()
    #
    # client = weaviate.connect_to_local()
    # collection = client.collections.get("DrugLawDocs")
    #
    # results = collection.query.near_vector(
    #     near_vector=query_embedding,
    #     limit=top_k,
    #     return_metadata=MetadataQuery(distance=True)
    # )
    #
    # return [
    #     {
    #         "content": obj.properties["content"],
    #         "score": 1 - obj.metadata.distance,  # distance → similarity
    #         "metadata": {"source": obj.properties["source"], ...}
    #     }
    #     for obj in results.objects
    # ]
    # 1. Embed query bằng cùng model với Task 4
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # 2. Kết nối ChromaDB
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collection = client.get_collection(
        COLLECTION_NAME
    )

    # 3. Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 4. Format output
    output = []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(
            docs,
            metas,
            distances
    ):
        # cosine distance -> similarity
        similarity = 1.0 - distance

        output.append({
            "content": doc,
            "score": float(similarity),
            "metadata": meta
        })

    output.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return output


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
