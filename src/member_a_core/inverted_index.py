"""
Inverted Index Implementation with Structural Metadata & WAND Upper-Bound Precomputation.
Provides efficient term posting lists, document frequency tracking, and skip navigation.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.common.schemas import DocumentChunk
from src.common.config import BM25_K1, BM25_B
from src.member_a_core.tokenizer import MedicalLegalTokenizer


@dataclass
class Posting:
    """Single posting entry for a term in a document."""
    doc_int_id: int
    doc_id: str
    term_freq: int
    doc_len: int


class InvertedIndex:
    """
    In-memory Inverted Index tailored for legal/regulatory information retrieval.
    Precomputes IDF and maximum term contribution bounds (U_t) for WAND dynamic pruning.
    """
    def __init__(self, tokenizer: Optional[MedicalLegalTokenizer] = None):
        self.tokenizer = tokenizer or MedicalLegalTokenizer()
        self.documents: Dict[int, DocumentChunk] = {}
        self.doc_id_to_int: Dict[str, int] = {}
        self.int_to_doc_id: Dict[int, str] = {}
        
        # Inverted index posting lists: term -> list of Posting sorted by doc_int_id
        self.postings: Dict[str, List[Posting]] = {}
        
        # Document statistics
        self.doc_lengths: Dict[int, int] = {}
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        
        # Term statistics
        self.doc_frequencies: Dict[str, int] = {}
        self.idf_cache: Dict[str, float] = {}
        self.term_upper_bounds: Dict[str, float] = {}  # U_t for WAND pruning

    def index_documents(self, chunks: List[DocumentChunk]):
        """Builds the complete inverted index and precomputes WAND statistics."""
        self.documents.clear()
        self.doc_id_to_int.clear()
        self.int_to_doc_id.clear()
        self.postings.clear()
        self.doc_lengths.clear()
        self.doc_frequencies.clear()
        self.idf_cache.clear()
        self.term_upper_bounds.clear()
        
        total_len = 0
        self.total_docs = len(chunks)
        
        # Step 1: Tokenize and build raw postings
        for int_id, chunk in enumerate(chunks):
            self.documents[int_id] = chunk
            self.doc_id_to_int[chunk.doc_id] = int_id
            self.int_to_doc_id[int_id] = chunk.doc_id
            
            # Combine title, text, and entities for fielded indexing
            combined_text = f"{chunk.title} {chunk.title} {chunk.text} {' '.join(chunk.entities)}"
            tokens = self.tokenizer.tokenize(combined_text)
            d_len = len(tokens)
            self.doc_lengths[int_id] = d_len
            total_len += d_len
            
            # Count term frequencies within this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
                
            # Append to posting lists
            for term, tf in tf_map.items():
                if term not in self.postings:
                    self.postings[term] = []
                    self.doc_frequencies[term] = 0
                self.postings[term].append(Posting(
                    doc_int_id=int_id,
                    doc_id=chunk.doc_id,
                    term_freq=tf,
                    doc_len=d_len
                ))
                self.doc_frequencies[term] += 1

        self.avg_doc_len = total_len / max(1, self.total_docs)
        
        # Step 2: Compute IDF and WAND Upper Bounds U_t for each term
        self._precompute_idf_and_upper_bounds()

    def _precompute_idf_and_upper_bounds(self):
        """
        Precomputes:
        1. Robertson-Spärck Jones IDF with smoothing.
        2. Maximum possible BM25 score contribution (U_t) for each term across all documents.
        """
        N = self.total_docs
        k1 = BM25_K1
        b = BM25_B
        avg_l = self.avg_doc_len
        
        for term, posting_list in self.postings.items():
            df = len(posting_list)
            # Standard BM25 IDF formulation with floor protection
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            idf = max(0.01, idf)
            self.idf_cache[term] = idf
            
            # Find max BM25 contribution in any posting for this term
            max_contrib = 0.0
            for p in posting_list:
                tf = p.term_freq
                doc_len = p.doc_len
                denom = tf + k1 * (1.0 - b + b * (doc_len / avg_l))
                contrib = idf * ((tf * (k1 + 1.0)) / max(1e-6, denom))
                if contrib > max_contrib:
                    max_contrib = contrib
                    
            self.term_upper_bounds[term] = max_contrib

    def get_idf(self, term: str) -> float:
        """Returns the precomputed IDF for a term, or an out-of-vocabulary penalty."""
        return self.idf_cache.get(term, 0.0)

    def get_upper_bound(self, term: str) -> float:
        """Returns precomputed maximum BM25 contribution U_t for WAND pruning."""
        return self.term_upper_bounds.get(term, 0.0)

    def get_postings(self, term: str) -> List[Posting]:
        """Returns the posting list for a term, sorted by doc_int_id."""
        return self.postings.get(term, [])

    def get_document(self, doc_int_id: int) -> Optional[DocumentChunk]:
        """Retrieves document chunk by integer ID."""
        return self.documents.get(doc_int_id)
