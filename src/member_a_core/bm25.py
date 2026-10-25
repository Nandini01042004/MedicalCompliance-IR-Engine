"""
Okapi BM25 Retrieval Engine (Exhaustive Baseline).
Provides standard full-index scoring to benchmark against WAND dynamic pruning.
"""

from typing import List, Tuple, Dict
from src.member_a_core.inverted_index import InvertedIndex
from src.common.config import BM25_K1, BM25_B


class BM25Scorer:
    """Computes Okapi BM25 relevance scores over the InvertedIndex."""
    
    def __init__(self, index: InvertedIndex, k1: float = BM25_K1, b: float = BM25_B):
        self.index = index
        self.k1 = k1
        self.b = b

    def compute_term_score(self, term: str, tf: int, doc_len: int) -> float:
        """Computes BM25 score contribution for a single term-doc occurrence."""
        idf = self.index.get_idf(term)
        if idf <= 0.0 or tf <= 0:
            return 0.0
        
        avg_l = max(1.0, self.index.avg_doc_len)
        denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_l))
        return idf * ((tf * (self.k1 + 1.0)) / max(1e-6, denom))

    def exhaustive_search(self, query_tokens: List[str], top_k: int = 50) -> Tuple[List[Tuple[int, float]], int]:
        """
        Performs exhaustive linear evaluation of all postings for the given query terms.
        Returns:
            - List of (doc_int_id, bm25_score) sorted in descending score order.
            - Total number of postings evaluated (for computational complexity comparison).
        """
        scores: Dict[int, float] = {}
        postings_evaluated = 0
        
        # Accumulate score for every posting across all query terms
        for term in query_tokens:
            postings = self.index.get_postings(term)
            for p in postings:
                postings_evaluated += 1
                contrib = self.compute_term_score(term, p.term_freq, p.doc_len)
                scores[p.doc_int_id] = scores.get(p.doc_int_id, 0.0) + contrib
                
        # Sort in descending score order
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_results, postings_evaluated
