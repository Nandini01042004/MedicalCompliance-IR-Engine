"""
FastAPI Industry-Ready Application Server for Medical Compliance Engine.
Provides production async REST endpoints for search, speech normalization,
corpus querying, and live IR benchmarking.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

from src.common.schemas import VoiceQueryRequest, SearchResponse
from src.engine import ComplianceEngine
from src.evaluation.run_experiments import run_ablation_benchmarks

app = FastAPI(
    title="Voice-Enabled Medical Compliance Engine API",
    description="Scalable IR engine featuring Dynamic Query Pruning (WAND) and Result Diversification (MMR/xQuAD).",
    version="1.0.0"
)

# Enable CORS for web clients / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine globally
engine = ComplianceEngine()


@app.on_event("startup")
def startup_event():
    """Initializes corpus, inverted index, and vector space on startup."""
    engine.initialize()


@app.get("/")
def health_check():
    """Root health and status endpoint."""
    return {
        "status": "healthy",
        "service": "Voice-Enabled Medical Compliance Engine",
        "total_clauses_indexed": len(engine.chunks),
        "jurisdictions_supported": ["US_HIPAA", "US_FDA", "IN_CDSCO"],
        "algorithms": {
            "lexical": "BM25 with Robertson-Spärck Jones IDF",
            "pruning": "Dynamic Query-Side TDV + Index-Side WAND (Weak AND)",
            "dense": "FAISS IndexFlatIP with Latent Semantic Vectors",
            "fusion": "Reciprocal Rank Fusion (RRF)",
            "diversification": "Maximal Marginal Relevance (MMR) & xQuAD"
        }
    }


@app.post("/api/search", response_model=SearchResponse)
def execute_search(request: VoiceQueryRequest):
    """
    Executes full compliance retrieval:
    Speech Normalization -> Query Pruning -> WAND Lexical -> FAISS Dense -> RRF -> Diversification.
    """
    try:
        response = engine.search(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/normalize")
def normalize_voice_input(
    raw_transcript: str = Query(..., description="Raw text transcript from speech recognition"),
    simulate_asr_noise: bool = Query(False, description="Simulate realistic ASR phonetic corruptions")
):
    """Corrects ASR speech-to-text recognition errors using phonetic matching."""
    return engine.asr.process_spoken_input(
        raw_text_transcript=raw_transcript,
        simulate_asr_noise=simulate_asr_noise
    )


@app.get("/api/corpus")
def get_corpus(
    jurisdiction: Optional[str] = Query(None, description="Filter by US_HIPAA, US_FDA, or IN_CDSCO"),
    limit: int = Query(20, description="Max clauses to return")
):
    """Returns stored regulatory clauses with optional jurisdiction filter."""
    clauses = engine.chunks
    if jurisdiction:
        clauses = [c for c in clauses if c.jurisdiction == jurisdiction]
    return {
        "total_found": len(clauses),
        "clauses": clauses[:limit]
    }


@app.get("/api/benchmark/ablation")
def get_ablation_benchmark():
    """Executes live IR ablation experiments and returns evaluation metrics table."""
    try:
        results = run_ablation_benchmarks()
        return {
            "status": "success",
            "experiments": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
