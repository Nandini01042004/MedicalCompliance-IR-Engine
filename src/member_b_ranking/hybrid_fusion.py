"""
Hybrid Retrieval Engine with Reciprocal Rank Fusion (RRF).
Merges ranked candidate lists from sparse lexical (BM25 + WAND) and dense semantic (FAISS) retrieval.
"""

from typing import List, Tuple, Dict
from src.common.config import RRF_K


class HybridRankFusion:
    """Combines lexical and dense rankings using Reciprocal Rank Fusion."""
    
    def __init__(self, k_constant: float = RRF_K):
        self.k_constant = k_constant

    def fuse(
        self,
        lexical_results: List[Tuple[int, float]],
        dense_results: List[Tuple[int, float]],
        top_k: int = 50
    ) -> List[Dict[str, float]]:
        """
        Fuses sparse and dense rankings.
        Returns:
            List of dicts:
            {
               'doc_int_id': int,
               'rrf_score': float,
               'lexical_score': float,
               'dense_score': float,
               'lexical_rank': int,
               'dense_rank': int
            }
        """
        rrf_scores: Dict[int, float] = {}
        lexical_meta: Dict[int, Tuple[float, int]] = {}
        dense_meta: Dict[int, Tuple[float, int]] = {}

        # Process Lexical rankings
        for rank, (doc_id, score) in enumerate(lexical_results, start=1):
            lexical_meta[doc_id] = (score, rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k_constant + rank))

        # Process Dense rankings
        for rank, (doc_id, score) in enumerate(dense_results, start=1):
            dense_meta[doc_id] = (score, rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k_constant + rank))

        # Sort documents by accumulated RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        merged_results = []
        for doc_id, rrf_score in sorted_docs:
            lex_score, lex_rank = lexical_meta.get(doc_id, (0.0, -1))
            den_score, den_rank = dense_meta.get(doc_id, (0.0, -1))
            merged_results.append({
                "doc_int_id": doc_id,
                "rrf_score": rrf_score,
                "lexical_score": lex_score,
                "dense_score": den_score,
                "lexical_rank": lex_rank,
                "dense_rank": den_rank
            })

        return merged_results
