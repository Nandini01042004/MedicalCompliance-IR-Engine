"""
Query-Side Dynamic Pruner.
Analyzes conversational spoken queries, computes Term Discrimination Value (TDV)
and IDF weights, and prunes low-information / conversational noise tokens.
"""

from typing import List, Tuple, Set
from src.member_a_core.inverted_index import InvertedIndex
from src.member_a_core.tokenizer import MedicalLegalTokenizer, SPOKEN_FILLERS
from src.common.config import QUERY_PRUNE_RATIO


class QuerySidePruner:
    """
    Dynamically prunes conversational noise and low-discriminating terms
    from spoken queries prior to inverted list retrieval.
    """
    def __init__(self, index: InvertedIndex, tokenizer: MedicalLegalTokenizer, prune_ratio: float = QUERY_PRUNE_RATIO):
        self.index = index
        self.tokenizer = tokenizer
        self.prune_ratio = prune_ratio

    def prune_query(self, raw_query: str) -> Tuple[List[str], List[str]]:
        """
        Processes a raw spoken query transcript.
        Returns:
            - active_tokens: Informative search tokens retained for index traversal.
            - pruned_tokens: Discarded filler/low-discrimination tokens.
        """
        # Tokenize preserving raw tokens for analysis
        tokens = self.tokenizer.tokenize(raw_query)
        if not tokens:
            return [], []

        # Step 1: Compute term weights using IDF and length bonus
        term_weights = {}
        for t in set(tokens):
            idf = self.index.get_idf(t)
            # Penalize known spoken fillers that slipped through
            penalty = 0.1 if t in SPOKEN_FILLERS else 1.0
            term_weights[t] = idf * penalty

        max_weight = max(term_weights.values()) if term_weights else 0.0
        # Dynamic threshold: adaptive based on highest discriminating term in query
        cutoff_threshold = max_weight * self.prune_ratio

        active_tokens: List[str] = []
        pruned_tokens: List[str] = []

        for t in tokens:
            weight = term_weights.get(t, 0.0)
            if weight >= cutoff_threshold and weight > 0.1:
                active_tokens.append(t)
            else:
                pruned_tokens.append(t)

        # Safety fallback: if all terms were pruned, retain top 3 highest weight terms
        if not active_tokens and tokens:
            sorted_by_weight = sorted(set(tokens), key=lambda x: term_weights.get(x, 0.0), reverse=True)
            active_tokens = sorted_by_weight[:3]
            pruned_tokens = [t for t in tokens if t not in active_tokens]

        return active_tokens, pruned_tokens
