"""
WAND (Weak AND) Dynamic Query Pruning Engine.
Implements dynamic early-exit inverted list traversal with an adaptive score threshold
and term upper-bound skipping to deliver massive latency and computation reductions.
"""

import heapq
from typing import List, Tuple, Dict, Optional
from src.member_a_core.inverted_index import InvertedIndex, Posting
from src.member_a_core.bm25 import BM25Scorer


class PostingIterator:
    """Cursor iterating over a single term's posting list with fast skip navigation."""
    
    def __init__(self, term: str, postings: List[Posting], upper_bound: float):
        self.term = term
        self.postings = postings
        self.upper_bound = upper_bound
        self.cursor = 0
        self.length = len(postings)

    @property
    def is_exhausted(self) -> bool:
        return self.cursor >= self.length

    @property
    def current_doc(self) -> int:
        return self.postings[self.cursor].doc_int_id

    @property
    def current_posting(self) -> Posting:
        return self.postings[self.cursor]

    def advance(self):
        """Moves cursor to next posting."""
        self.cursor += 1

    def advance_to(self, target_doc_int_id: int):
        """Skips postings until current_doc >= target_doc_int_id using linear/binary skip."""
        while self.cursor < self.length and self.postings[self.cursor].doc_int_id < target_doc_int_id:
            self.cursor += 1


class WANDPruner:
    """
    Executes Weak-AND dynamic query pruning over the InvertedIndex.
    Guarantees top-K exactness while skipping uncompetitive postings.
    """
    def __init__(self, index: InvertedIndex, scorer: BM25Scorer):
        self.index = index
        self.scorer = scorer

    def wand_search(self, query_tokens: List[str], top_k: int = 50) -> Tuple[List[Tuple[int, float]], int]:
        """
        Executes WAND algorithm.
        Returns:
            - List of (doc_int_id, score) in descending score order.
            - Total number of postings actually evaluated.
        """
        # Filter query tokens to those present in vocabulary
        valid_terms = [t for t in query_tokens if t in self.index.postings and self.index.postings[t]]
        if not valid_terms:
            return [], 0

        # Initialize posting iterators with term upper bounds
        iterators: List[PostingIterator] = []
        for term in valid_terms:
            ub = self.index.get_upper_bound(term)
            postings = self.index.get_postings(term)
            iterators.append(PostingIterator(term, postings, ub))

        # Min-heap to maintain top-K candidates: elements are (score, doc_int_id)
        top_k_heap: List[Tuple[float, int]] = []
        theta: float = 0.0  # Dynamic threshold
        postings_evaluated: int = 0

        while True:
            # Step 1: Remove exhausted iterators
            iterators = [it for it in iterators if not it.is_exhausted]
            if not iterators:
                break

            # Step 2: Sort active iterators by current doc_int_id
            iterators.sort(key=lambda it: it.current_doc)

            # Step 3: Find pivot document by accumulating upper bounds
            accum_bound = 0.0
            pivot_idx = -1
            for idx, it in enumerate(iterators):
                accum_bound += it.upper_bound
                if accum_bound > theta:
                    pivot_idx = idx
                    break

            # If cumulative upper bounds cannot exceed threshold theta, terminate search!
            if pivot_idx == -1:
                break

            pivot_doc = iterators[pivot_idx].current_doc

            # Step 4: Check if leading iterator points to the pivot document
            if iterators[0].current_doc == pivot_doc:
                # Fully score pivot document across all matching terms
                full_score = 0.0
                for it in iterators:
                    if it.current_doc == pivot_doc:
                        p = it.current_posting
                        contrib = self.scorer.compute_term_score(it.term, p.term_freq, p.doc_len)
                        full_score += contrib
                        postings_evaluated += 1
                        it.advance()
                    elif it.current_doc > pivot_doc:
                        break

                # Update top-K heap if score qualifies
                if full_score > theta or len(top_k_heap) < top_k:
                    if len(top_k_heap) < top_k:
                        heapq.heappush(top_k_heap, (full_score, pivot_doc))
                    else:
                        heapq.heappushpop(top_k_heap, (full_score, pivot_doc))
                    
                    # Update dynamic threshold theta to the new minimum in top-K
                    if len(top_k_heap) >= top_k:
                        theta = top_k_heap[0][0]
            else:
                # Step 5: Advance iterator[0] forward directly to pivot_doc, skipping non-competitive postings!
                iterators[0].advance_to(pivot_doc)

        # Extract and sort results in descending order
        sorted_results = sorted([(doc_id, score) for score, doc_id in top_k_heap], key=lambda x: x[1], reverse=True)
        return sorted_results, postings_evaluated
