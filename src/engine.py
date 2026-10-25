"""
Master Compliance Search Engine Orchestrator.
Integrates Member A (Inverted Index, WAND, Query-Side Pruning) and
Member B (Phonetic Normalization, FAISS Dense Search, Hybrid Fusion, Diversification).
"""

import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.common.schemas import (
    DocumentChunk, VoiceQueryRequest, QueryMetadata,
    ScoredDocument, SearchResponse
)
from src.common.config import CORPUS_FILE, TOP_K_CANDIDATES
from src.member_a_core.tokenizer import MedicalLegalTokenizer
from src.member_a_core.inverted_index import InvertedIndex
from src.member_a_core.bm25 import BM25Scorer
from src.member_a_core.wand_pruner import WANDPruner
from src.member_a_core.query_pruner import QuerySidePruner
from src.member_b_ranking.normalizer import MedicalPhoneticNormalizer
from src.member_b_ranking.asr_pipeline import ASRPipeline
from src.member_b_ranking.dense_retriever import DenseSemanticRetriever
from src.member_b_ranking.hybrid_fusion import HybridRankFusion
from src.member_b_ranking.diversifier import ResultDiversifier


class ComplianceEngine:
    """Production-grade Medical & Pharmaceutical Compliance Information Retrieval Engine."""
    
    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = corpus_path or CORPUS_FILE
        self.chunks: List[DocumentChunk] = []
        self.doc_store: Dict[int, DocumentChunk] = {}
        
        # Member A Components
        self.tokenizer = MedicalLegalTokenizer()
        self.index = InvertedIndex(self.tokenizer)
        self.bm25 = BM25Scorer(self.index)
        self.wand = WANDPruner(self.index, self.bm25)
        self.query_pruner = QuerySidePruner(self.index, self.tokenizer)
        
        # Member B Components
        self.normalizer = MedicalPhoneticNormalizer()
        self.asr = ASRPipeline(self.normalizer)
        self.dense = DenseSemanticRetriever()
        self.fusion = HybridRankFusion()
        self.diversifier = ResultDiversifier(self.dense)
        
        self.is_initialized = False

    def initialize(self):
        """Loads corpus, builds inverted index with WAND bounds, and fits FAISS vector index."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_path}. Run build_corpus.py first.")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            
        self.chunks = [DocumentChunk(**item) for item in raw_data]
        self.doc_store = {idx: chunk for idx, chunk in enumerate(self.chunks)}

        # Build Inverted Index & WAND bounds
        self.index.index_documents(self.chunks)

        # Build Dense Semantic FAISS Index
        self.dense.fit_and_index(self.chunks)

        self.is_initialized = True
        print(f"ComplianceEngine successfully initialized with {len(self.chunks)} legal clauses.")

    def search(self, request: VoiceQueryRequest) -> SearchResponse:
        """Executes the complete end-to-end voice/text compliance retrieval pipeline."""
        if not self.is_initialized:
            self.initialize()

        latency_breakdown: Dict[str, float] = {}
        t_start = time.perf_counter()

        # Step 1: Voice / Speech Normalization
        t0 = time.perf_counter()
        asr_result = self.asr.process_spoken_input(
            raw_text_transcript=request.raw_query,
            confidence_score=request.confidence_score
        )
        normalized_query = asr_result["normalized_query"]
        latency_breakdown["speech_normalization"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 2: Dynamic Query-Side Pruning
        t0 = time.perf_counter()
        if request.enable_query_pruning:
            active_tokens, pruned_tokens = self.query_pruner.prune_query(normalized_query)
        else:
            active_tokens = self.tokenizer.tokenize(normalized_query)
            pruned_tokens = []
        latency_breakdown["query_pruning"] = round((time.perf_counter() - t0) * 1000, 2)

        # Query string formed from active tokens
        search_query_str = " ".join(active_tokens) if active_tokens else normalized_query

        # Step 3: Lexical Retrieval (WAND vs Exhaustive)
        t0 = time.perf_counter()
        # Always run exhaustive count to compute exact savings benchmark
        exhaustive_results, postings_exhaustive = self.bm25.exhaustive_search(active_tokens, top_k=TOP_K_CANDIDATES)
        
        if request.enable_wand_pruning:
            lexical_candidates, postings_wand = self.wand.wand_search(active_tokens, top_k=TOP_K_CANDIDATES)
        else:
            lexical_candidates = exhaustive_results
            postings_wand = postings_exhaustive
            
        latency_breakdown["lexical_retrieval"] = round((time.perf_counter() - t0) * 1000, 2)

        # Postings skipped percentage
        postings_skipped = 0.0
        if postings_exhaustive > 0:
            postings_skipped = round(((postings_exhaustive - postings_wand) / postings_exhaustive) * 100.0, 1)

        # Step 4: Dense Vector Search via FAISS
        t0 = time.perf_counter()
        dense_candidates = self.dense.search(search_query_str, top_k=TOP_K_CANDIDATES)
        latency_breakdown["dense_faiss_search"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 5: Hybrid Fusion (Reciprocal Rank Fusion)
        t0 = time.perf_counter()
        fused_candidates = self.fusion.fuse(lexical_candidates, dense_candidates, top_k=TOP_K_CANDIDATES)
        
        # Apply Jurisdiction Filter if requested
        if request.jurisdiction_filter:
            fused_candidates = [
                c for c in fused_candidates
                if self.doc_store[c["doc_int_id"]].jurisdiction == request.jurisdiction_filter
            ]
        latency_breakdown["hybrid_fusion"] = round((time.perf_counter() - t0) * 1000, 2)

        # Step 6: Result Diversification (MMR or xQuAD)
        t0 = time.perf_counter()
        if request.enable_diversification:
            if request.diversification_method.lower() == "xquad":
                final_results = self.diversifier.diversify_xquad(
                    fused_candidates,
                    self.doc_store,
                    query_text=search_query_str,
                    top_k=request.top_k
                )
            else:
                final_results = self.diversifier.diversify_mmr(
                    fused_candidates,
                    self.doc_store,
                    top_k=request.top_k,
                    lambda_param=request.lambda_param
                )
        else:
            # Standard un-diversified hybrid top-K
            final_results = []
            for rank, c in enumerate(fused_candidates[:request.top_k], start=1):
                chunk = self.doc_store[c["doc_int_id"]]
                final_results.append(ScoredDocument(
                    rank=rank,
                    doc=chunk,
                    lexical_score=c["lexical_score"],
                    dense_score=c["dense_score"],
                    hybrid_rrf_score=c["rrf_score"],
                    final_score=round(c["rrf_score"], 4),
                    subtopic_aspect=chunk.subtopic_tag,
                    diversity_penalty=0.0
                ))
        latency_breakdown["result_diversification"] = round((time.perf_counter() - t0) * 1000, 2)

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)
        latency_breakdown["total_pipeline"] = total_latency

        # Track unique subtopic facets covered in top results
        unique_aspects = list({res.subtopic_aspect for res in final_results})

        # Generate traceable legal audit summary
        audit_summary = self._generate_audit_summary(final_results)

        query_meta = QueryMetadata(
            raw_spoken_query=request.raw_query,
            normalized_query=normalized_query,
            pruned_tokens=pruned_tokens,
            active_query_tokens=active_tokens,
            postings_evaluated_exhaustive=postings_exhaustive,
            postings_evaluated_wand=postings_wand,
            postings_skipped_pct=postings_skipped,
            latency_breakdown_ms=latency_breakdown
        )

        return SearchResponse(
            status="success",
            query_metadata=query_meta,
            results=final_results,
            unique_subtopics_covered=unique_aspects,
            audit_summary=audit_summary
        )

    def _generate_audit_summary(self, results: List[ScoredDocument]) -> str:
        """Generates an auditable compliance explanation linking exact legal clauses."""
        if not results:
            return "No matching compliance regulations found for the specified query."

        top_doc = results[0].doc
        aspects_covered = {r.doc.subtopic_tag.replace("_", " ").title() for r in results[:4]}
        citations = [f"{r.doc.clause_reference} ({r.doc.statute})" for r in results[:3]]

        summary = (
            f"Regulatory analysis identified {len(results)} authoritative provisions spanning "
            f"facets: {', '.join(aspects_covered)}. "
            f"Primary compliance obligations mandate adherence to {top_doc.clause_reference} "
            f"({top_doc.statute}), with key governing references: {'; '.join(citations)}."
        )
        return summary
