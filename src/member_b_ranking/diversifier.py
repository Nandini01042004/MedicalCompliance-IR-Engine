"""
Result Diversification Engine implementing MMR and xQuAD.
Balances relevance against novelty to guarantee multi-faceted regulatory compliance coverage.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from src.common.schemas import DocumentChunk, ScoredDocument
from src.member_b_ranking.dense_retriever import DenseSemanticRetriever
from src.common.config import DEFAULT_LAMBDA, XQUAD_ALPHA


class ResultDiversifier:
    """
    Diversifies ranked compliance search results using either:
    1. Maximal Marginal Relevance (MMR)
    2. Explicit Query Aspect Diversification (xQuAD)
    """
    def __init__(self, dense_retriever: DenseSemanticRetriever):
        self.dense_retriever = dense_retriever

    def diversify_mmr(
        self,
        candidate_pool: List[Dict[str, Any]],
        doc_store: Dict[int, DocumentChunk],
        top_k: int = 10,
        lambda_param: float = DEFAULT_LAMBDA
    ) -> List[ScoredDocument]:
        """
        Executes Maximal Marginal Relevance (MMR) greedy selection.
        MMR(d) = lambda * Sim1(d, Q) - (1 - lambda) * max_{dj in S} Sim2(d, dj)
        """
        if not candidate_pool:
            return []

        # Normalize relevance scores to [0, 1]
        raw_scores = [c["rrf_score"] for c in candidate_pool]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max(1e-6, max_s - min_s)

        candidates = []
        for c in candidate_pool:
            doc_id = c["doc_int_id"]
            norm_rel = (c["rrf_score"] - min_s) / score_range
            vec = self.dense_retriever.get_document_vector(doc_id)
            candidates.append({
                "doc_int_id": doc_id,
                "norm_rel": norm_rel,
                "raw_info": c,
                "vector": vec
            })

        selected_results: List[ScoredDocument] = []
        selected_indices = set()

        while len(selected_results) < top_k and len(selected_indices) < len(candidates):
            best_mmr = -float("inf")
            best_cand = None
            best_penalty = 0.0

            for cand in candidates:
                d_id = cand["doc_int_id"]
                if d_id in selected_indices:
                    continue

                rel = cand["norm_rel"]

                # Calculate max similarity with already selected documents
                if not selected_results:
                    redundancy = 0.0
                else:
                    max_sim = 0.0
                    for sel in selected_results:
                        sel_vec = self.dense_retriever.get_document_vector(sel.doc.doc_id_int if hasattr(sel.doc, 'doc_id_int') else cand["doc_int_id"])
                        if cand["vector"] is not None and sel_vec is not None:
                            # Cosine sim on normalized vectors = dot product
                            sim = float(np.dot(cand["vector"], sel_vec))
                            if sim > max_sim:
                                max_sim = sim
                    redundancy = max_sim

                mmr_score = (lambda_param * rel) - ((1.0 - lambda_param) * redundancy)

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cand = cand
                    best_penalty = redundancy

            if best_cand is None:
                break

            d_int = best_cand["doc_int_id"]
            selected_indices.add(d_int)
            chunk = doc_store[d_int]
            raw = best_cand["raw_info"]

            selected_results.append(ScoredDocument(
                rank=len(selected_results) + 1,
                doc=chunk,
                lexical_score=raw["lexical_score"],
                dense_score=raw["dense_score"],
                hybrid_rrf_score=raw["rrf_score"],
                final_score=round(best_mmr, 4),
                subtopic_aspect=chunk.subtopic_tag,
                diversity_penalty=round(best_penalty, 4)
            ))

        return selected_results

    def diversify_xquad(
        self,
        candidate_pool: List[Dict[str, Any]],
        doc_store: Dict[int, DocumentChunk],
        query_text: str,
        top_k: int = 10,
        alpha_weight: float = XQUAD_ALPHA
    ) -> List[ScoredDocument]:
        """
        Executes Explicit Query Aspect Diversification (xQuAD).
        Selects documents that satisfy multiple distinct compliance aspects.
        """
        if not candidate_pool:
            return []

        # Normalize relevance scores
        raw_scores = [c["rrf_score"] for c in candidate_pool]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max(1e-6, max_s - min_s)

        # Collect unique aspects present in the candidate pool
        aspects = list({doc_store[c["doc_int_id"]].subtopic_tag for c in candidate_pool})
        
        # Estimate P(aspect | Query) based on keyword overlaps
        q_lower = query_text.lower()
        aspect_prior = {}
        for a in aspects:
            # Tokenize aspect name, e.g. 'storage_temperature' -> ['storage', 'temperature']
            words = a.split("_")
            matches = sum(1 for w in words if w in q_lower)
            aspect_prior[a] = 1.0 + (matches * 2.0)
            
        # Normalize aspect priors
        total_p = sum(aspect_prior.values())
        for a in aspects:
            aspect_prior[a] /= total_p

        selected_results: List[ScoredDocument] = []
        selected_indices = set()
        covered_aspects_count: Dict[str, int] = {a: 0 for a in aspects}

        while len(selected_results) < top_k and len(selected_indices) < len(candidate_pool):
            best_xquad = -float("inf")
            best_cand = None

            for c in candidate_pool:
                d_id = c["doc_int_id"]
                if d_id in selected_indices:
                    continue

                chunk = doc_store[d_id]
                rel = (c["rrf_score"] - min_s) / score_range

                # Calculate aspect novelty bonus
                novelty_bonus = 0.0
                for a in aspects:
                    p_a_q = aspect_prior[a]
                    p_d_a = 1.0 if chunk.subtopic_tag == a else 0.0
                    decay = (1.0 - 0.7) ** covered_aspects_count[a]
                    novelty_bonus += p_a_q * p_d_a * decay

                xquad_score = ((1.0 - alpha_weight) * rel) + (alpha_weight * novelty_bonus)

                if xquad_score > best_xquad:
                    best_xquad = xquad_score
                    best_cand = c

            if best_cand is None:
                break

            d_int = best_cand["doc_int_id"]
            selected_indices.add(d_int)
            chunk = doc_store[d_int]
            covered_aspects_count[chunk.subtopic_tag] += 1

            selected_results.append(ScoredDocument(
                rank=len(selected_results) + 1,
                doc=chunk,
                lexical_score=best_cand["lexical_score"],
                dense_score=best_cand["dense_score"],
                hybrid_rrf_score=best_cand["rrf_score"],
                final_score=round(best_xquad, 4),
                subtopic_aspect=chunk.subtopic_tag,
                diversity_penalty=round(1.0 - (1.0 / (covered_aspects_count[chunk.subtopic_tag])), 4)
            ))

        return selected_results
