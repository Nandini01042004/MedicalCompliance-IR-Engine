"""
Information Retrieval Evaluation Metrics Suite.
Implements academic gold-standard metrics for Relevance and Diversity:
- MAP (Mean Average Precision)
- NDCG@K (Normalized Discounted Cumulative Gain)
- alpha-NDCG@K (TREC Intent-Aware Diversity Metric)
- Subtopic Recall (S-Recall@K)
- ILAD (Intra-List Average Distance)
"""

import math
from typing import List, Dict, Set
import numpy as np


class IRMetrics:
    """Calculates quantitative relevance and diversification metrics."""

    @staticmethod
    def precision_at_k(retrieved_docs: List[str], ground_truth_qrels: Dict[str, int], k: int = 10) -> float:
        """Computes Precision@K: fraction of top-K documents that are relevant (qrel >= 1)."""
        if not retrieved_docs or k <= 0:
            return 0.0
        top_k = retrieved_docs[:k]
        rel_count = sum(1 for doc_id in top_k if ground_truth_qrels.get(doc_id, 0) >= 1)
        return rel_count / float(k)

    @staticmethod
    def average_precision(retrieved_docs: List[str], ground_truth_qrels: Dict[str, int]) -> float:
        """Computes Average Precision (AP) for a single query."""
        relevant_docs = {doc_id for doc_id, rel in ground_truth_qrels.items() if rel >= 1}
        if not relevant_docs:
            return 0.0

        hits = 0
        sum_precisions = 0.0
        for i, doc_id in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                hits += 1
                sum_precisions += (hits / float(i))

        return sum_precisions / float(len(relevant_docs))

    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], ground_truth_qrels: Dict[str, int], k: int = 10) -> float:
        """
        Computes Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance.
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
        """
        if not retrieved_docs or k <= 0:
            return 0.0

        top_k = retrieved_docs[:k]
        dcg = 0.0
        for i, doc_id in enumerate(top_k, start=1):
            rel = ground_truth_qrels.get(doc_id, 0)
            dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        # Compute Ideal DCG (IDCG@K)
        ideal_rels = sorted(ground_truth_qrels.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

        if idcg <= 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def alpha_ndcg_at_k(
        retrieved_docs: List[str],
        doc_aspects_map: Dict[str, Set[str]],
        query_aspects: List[str],
        k: int = 10,
        alpha: float = 0.5
    ) -> float:
        """
        Computes alpha-NDCG@K (Clarke et al., SIGIR 2008).
        Penalizes redundant retrieval of the same compliance aspects.
        Gain G(i) = sum_{a in A(d_i)} (1 - alpha)^{c_a(i - 1)}
        """
        if not retrieved_docs or not query_aspects or k <= 0:
            return 0.0

        q_aspect_set = set(query_aspects)
        top_k = retrieved_docs[:k]

        # Calculate actual alpha-DCG
        alpha_dcg = 0.0
        aspect_counts: Dict[str, int] = {a: 0 for a in query_aspects}

        for i, doc_id in enumerate(top_k, start=1):
            doc_aspects = doc_aspects_map.get(doc_id, set()) & q_aspect_set
            gain = 0.0
            for a in doc_aspects:
                gain += math.pow(1.0 - alpha, aspect_counts[a])
                aspect_counts[a] += 1
            alpha_dcg += gain / math.log2(i + 1)

        # Calculate Ideal alpha-DCG (greedy ideal selection of aspects)
        ideal_aspect_counts: Dict[str, int] = {a: 0 for a in query_aspects}
        ideal_alpha_dcg = 0.0
        
        # Build candidate pool of available document aspect sets
        available_docs = list(doc_aspects_map.keys())
        for i in range(1, min(k, len(available_docs)) + 1):
            # Pick doc that gives max incremental gain
            best_gain = -1.0
            best_doc = None
            for d in available_docs:
                d_aspects = doc_aspects_map.get(d, set()) & q_aspect_set
                g = sum(math.pow(1.0 - alpha, ideal_aspect_counts[a]) for a in d_aspects)
                if g > best_gain:
                    best_gain = g
                    best_doc = d
            
            if best_doc is not None and best_gain > 0:
                available_docs.remove(best_doc)
                for a in doc_aspects_map.get(best_doc, set()) & q_aspect_set:
                    ideal_aspect_counts[a] += 1
                ideal_alpha_dcg += best_gain / math.log2(i + 1)
            else:
                break

        if ideal_alpha_dcg <= 0.0:
            return 0.0

        return alpha_dcg / ideal_alpha_dcg

    @staticmethod
    def subtopic_recall_at_k(
        retrieved_docs: List[str],
        doc_aspects_map: Dict[str, Set[str]],
        query_aspects: List[str],
        k: int = 10
    ) -> float:
        """
        Computes Subtopic Recall (S-Recall@K):
        Fraction of query's desired compliance aspects present in the top-K retrieved docs.
        """
        if not query_aspects:
            return 1.0
        top_k = retrieved_docs[:k]
        retrieved_aspects = set()
        for doc_id in top_k:
            retrieved_aspects.update(doc_aspects_map.get(doc_id, set()))

        covered = len(retrieved_aspects & set(query_aspects))
        return covered / float(len(query_aspects))

    @staticmethod
    def intra_list_average_distance(doc_vectors: List[np.ndarray]) -> float:
        """
        Computes Intra-List Average Distance (ILAD):
        Average pairwise cosine distance (1 - similarity) across top-K result vectors.
        Higher value indicates greater diversity and less redundancy.
        """
        n = len(doc_vectors)
        if n <= 1:
            return 0.0

        total_distance = 0.0
        pairs_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                v_i = doc_vectors[i]
                v_j = doc_vectors[j]
                if v_i is not None and v_j is not None:
                    # Vectors are L2 normalized, so dot product is cosine similarity
                    sim = float(np.dot(v_i, v_j))
                    dist = max(0.0, 1.0 - sim)
                    total_distance += dist
                    pairs_count += 1

        if pairs_count == 0:
            return 0.0

        return total_distance / float(pairs_count)
