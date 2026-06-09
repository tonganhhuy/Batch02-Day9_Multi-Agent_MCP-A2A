"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =============================================================================
# CONFIG
# =============================================================================

STANDARDIZED_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "standardized"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

CORPUS = []
BM25_INDEX = None


# =============================================================================
# LOAD DOCUMENTS
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Load toàn bộ markdown files và chunk giống Task 4.

    Returns:
        List[
            {
                "content": str,
                "metadata": {
                    "source": str,
                    "type": str,
                    "chunk_index": int
                }
            }
        ]
    """

    documents = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):

        content = md_file.read_text(
            encoding="utf-8"
        )

        doc_type = (
            "legal"
            if "legal" in str(md_file)
            else "news"
        )

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type
                }
            }
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for doc in documents:

        split_docs = splitter.split_text(
            doc["content"]
        )

        for i, chunk in enumerate(split_docs):

            chunks.append(
                {
                    "content": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i
                    }
                }
            )

    return chunks


# =============================================================================
# BUILD BM25
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index.

    Args:
        corpus: list chunks

    Returns:
        BM25Okapi object
    """

    tokenized_corpus = [
        doc["content"].lower().split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    return bm25


# =============================================================================
# SEARCH
# =============================================================================

def lexical_search(
    query: str,
    top_k: int = 10
) -> list[dict]:
    """
    BM25 search.

    Args:
        query: query string
        top_k: max results

    Returns:
        [
            {
                "content": str,
                "score": float,
                "metadata": dict
            }
        ]
    """

    global CORPUS
    global BM25_INDEX

    if BM25_INDEX is None:

        CORPUS = load_corpus()

        BM25_INDEX = build_bm25_index(
            CORPUS
        )

        print(
            f"Loaded {len(CORPUS)} chunks"
        )

    tokenized_query = (
        query.lower().split()
    )

    scores = BM25_INDEX.get_scores(
        tokenized_query
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        if scores[idx] <= 0:
            continue

        results.append(
            {
                "content":
                    CORPUS[idx]["content"],

                "score":
                    float(scores[idx]),

                "metadata":
                    CORPUS[idx]["metadata"]
            }
        )

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
