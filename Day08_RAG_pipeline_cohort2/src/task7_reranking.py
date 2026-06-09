"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional
import numpy as np


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Helper function to calculate cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    vec_a = np.array(a)
    vec_b = np.array(b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model với fallback an toàn.
    """
    if not candidates:
        return []

    # Make a copy of candidates to avoid side effects
    results = [dict(c) for c in candidates]

    try:
        from sentence_transformers import CrossEncoder
        # Sử dụng model nhỏ nhất để chạy nhanh trên CPU
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, r["content"]] for r in results]
        scores = model.predict(pairs)
        for r, score in zip(results, scores):
            r["score"] = float(score)
    except Exception as e:
        print(f"Fallback to lexical overlap rerank: {e}")
        query_words = set(query.lower().split())
        for r in results:
            doc_words = set(r["content"].lower().split())
            overlap = len(query_words.intersection(doc_words))
            r["score"] = float(r.get("score", 0.0) + overlap * 0.1)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates:
        return []

    # Make a copy of candidates
    results = [dict(c) for c in candidates]

    # Đảm bảo các candidate có embedding, nếu không thì lấy model bge-m3 để embed
    remaining = list(range(len(results)))
    for idx in remaining:
        if "embedding" not in results[idx] or not results[idx]["embedding"]:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("BAAI/bge-m3")
                for r in results:
                    if "embedding" not in r or not r["embedding"]:
                        r["embedding"] = model.encode(r["content"]).tolist()
            except:
                # Nếu hoàn toàn không thể embed, fallback trả về sort theo score
                results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
                return results[:top_k]

    selected = []
    for _ in range(min(top_k, len(results))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = cosine_sim(query_embedding, results[idx]["embedding"])

            # Max similarity to already selected
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(results[idx]["embedding"], results[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return [results[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    """
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item.copy()

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content]
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-m3")
            query_embedding = model.encode(query).tolist()
        except:
            query_embedding = [0.0] * 1024
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
