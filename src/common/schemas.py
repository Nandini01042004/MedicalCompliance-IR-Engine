"""
Core Pydantic data contracts for the Voice-Enabled Medical Compliance Engine.
Defines standard schemas used across Member A (Indexing & Pruning) and Member B (Ranking & Diversification).
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete, auditable legal/regulatory compliance clause."""
    doc_id: str = Field(..., description="Unique deterministic identifier, e.g., HIPAA_164_514_B")
    jurisdiction: str = Field(..., description="Legal jurisdiction: US_HIPAA, US_FDA, IN_CDSCO")
    statute: str = Field(..., description="Parent statute or regulation title")
    part: str = Field(..., description="Specific subpart, rule, or schedule")
    clause_reference: str = Field(..., description="Exact legal section, e.g. § 164.514(b) or Schedule M Sec 4.1")
    subtopic_tag: str = Field(..., description="Compliance aspect/facet, e.g. de_identification, audit_trails")
    title: str = Field(..., description="Human readable clause title")
    text: str = Field(..., description="Full normative clause text")
    mandatory_level: str = Field("SHALL", description="Legal obligation: SHALL, MUST, RECOMMENDED, CONDITIONAL")
    entities: List[str] = Field(default_factory=list, description="Key medical/legal entity keywords")


class VoiceQueryRequest(BaseModel):
    """Incoming search request originating from voice transcription or direct text."""
    raw_query: str = Field(..., description="Spoken raw transcript or user typed text")
    confidence_score: float = Field(1.0, description="ASR acoustic confidence score [0.0 - 1.0]")
    enable_query_pruning: bool = Field(True, description="Enable query-side filler/noise removal")
    enable_wand_pruning: bool = Field(True, description="Enable index-side WAND early-exit traversal")
    enable_diversification: bool = Field(True, description="Apply result diversification (MMR / xQuAD)")
    diversification_method: str = Field("mmr", description="Diversification algorithm: 'mmr' or 'xquad'")
    lambda_param: float = Field(0.65, description="Relevance vs Novelty balance parameter [0.0 - 1.0]")
    top_k: int = Field(10, description="Number of diversified results to return")
    jurisdiction_filter: Optional[str] = Field(None, description="Optional filter: US_HIPAA, US_FDA, IN_CDSCO")


class QueryMetadata(BaseModel):
    """Auditing and profiling metadata for query execution."""
    raw_spoken_query: str
    normalized_query: str
    pruned_tokens: List[str] = Field(default_factory=list)
    active_query_tokens: List[str] = Field(default_factory=list)
    postings_evaluated_exhaustive: int = 0
    postings_evaluated_wand: int = 0
    postings_skipped_pct: float = 0.0
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)


class ScoredDocument(BaseModel):
    """Scored and ranked compliance document with aspect annotation."""
    rank: int
    doc: DocumentChunk
    lexical_score: float = 0.0
    dense_score: float = 0.0
    hybrid_rrf_score: float = 0.0
    final_score: float = 0.0
    subtopic_aspect: str
    diversity_penalty: float = 0.0


class SearchResponse(BaseModel):
    """Complete response returned by the compliance engine."""
    status: str = "success"
    query_metadata: QueryMetadata
    results: List[ScoredDocument]
    unique_subtopics_covered: List[str] = Field(default_factory=list)
    audit_summary: str = ""


class EvaluationResult(BaseModel):
    """Benchmark metrics for IR experiment runs."""
    system_variant: str
    map_score: float
    ndcg_at_10: float
    alpha_ndcg_at_10: float
    subtopic_recall_at_10: float
    ilad_score: float
    avg_latency_ms: float
    p99_latency_ms: float
    avg_postings_evaluated: float
