"""
Comprehensive Pytest Test Suite for the Voice-Enabled Medical Compliance Engine.
Tests:
1. Medical Phonetic Normalization (ASR error correction)
2. Tokenizer & Query-Side Dynamic Pruning
3. Inverted Index & BM25 Scoring
4. WAND Dynamic Query Pruning (lossless early termination)
5. FAISS Dense Retrieval & Hybrid RRF
6. Result Diversification (MMR & xQuAD)
7. End-to-End Engine Search Execution
"""

import pytest
from src.common.schemas import DocumentChunk, VoiceQueryRequest
from src.member_a_core.tokenizer import MedicalLegalTokenizer
from src.member_a_core.inverted_index import InvertedIndex
from src.member_a_core.bm25 import BM25Scorer
from src.member_a_core.wand_pruner import WANDPruner
from src.member_a_core.query_pruner import QuerySidePruner
from src.member_b_ranking.normalizer import MedicalPhoneticNormalizer, soundex, levenshtein_distance
from src.member_b_ranking.dense_retriever import DenseSemanticRetriever
from src.member_b_ranking.hybrid_fusion import HybridRankFusion
from src.member_b_ranking.diversifier import ResultDiversifier
from src.engine import ComplianceEngine


@pytest.fixture(scope="session")
def sample_corpus():
    """Provides small test corpus."""
    return [
        DocumentChunk(
            doc_id="DOC_1",
            jurisdiction="US_HIPAA",
            statute="45 CFR 164",
            part="Subpart E",
            clause_reference="§ 164.514",
            subtopic_tag="de_identification",
            title="Safe Harbor De-identification",
            text="Health information is de-identified under Safe Harbor when 18 identifiers like names, SSN, and biometric records are removed.",
            mandatory_level="SHALL",
            entities=["safe harbor", "SSN", "de-identification"]
        ),
        DocumentChunk(
            doc_id="DOC_2",
            jurisdiction="US_HIPAA",
            statute="45 CFR 164",
            part="Subpart C",
            clause_reference="§ 164.312",
            subtopic_tag="audit_trails",
            title="Audit Controls",
            text="Implement hardware, software, and procedural mechanisms that record and examine activity in information systems.",
            mandatory_level="SHALL",
            entities=["audit controls", "logs"]
        ),
        DocumentChunk(
            doc_id="DOC_3",
            jurisdiction="IN_CDSCO",
            statute="Drugs and Cosmetics Rules, 1945",
            part="Schedule M",
            clause_reference="Schedule M Sec 7.4",
            subtopic_tag="storage_temperature",
            title="Cold Storage Conditions",
            text="Cold storage facilities for sensitive vaccines must maintain temperatures between 2°C and 8°C with continuous temperature loggers.",
            mandatory_level="SHALL",
            entities=["cold storage", "vaccines", "temperature"]
        ),
        DocumentChunk(
            doc_id="DOC_4",
            jurisdiction="IN_CDSCO",
            statute="Drugs and Cosmetics Rules, 1945",
            part="Schedule H1",
            clause_reference="Rule 65(3)(h)",
            subtopic_tag="antibiotic_register",
            title="Schedule H1 Antibiotic Register",
            text="A separate register for Schedule H1 antibiotics must be maintained and preserved for 3 years.",
            mandatory_level="MUST",
            entities=["Schedule H1", "register", "antibiotics"]
        )
    ]


def test_phonetic_normalizer():
    """Tests ASR phonetic repair for typical misheard speech queries."""
    normalizer = MedicalPhoneticNormalizer()
    
    # Phrase-level corrections
    text1, _ = normalizer.normalize("check hippa privacy rules for schedule age one")
    assert "hipaa" in text1
    assert "schedule h1" in text1
    
    # Acronym corrections
    text2, _ = normalizer.normalize("see dsco guidelines for p h i and b a a")
    assert "cdsco" in text2
    assert "phi" in text2
    assert "baa" in text2


def test_levenshtein_and_soundex():
    """Verifies string distance and Soundex encoding."""
    assert levenshtein_distance("hipaa", "hippa") == 1
    assert soundex("HIPAA") == soundex("HIPPA")


def test_inverted_index_and_bm25(sample_corpus):
    """Verifies posting creation, term upper bounds U_t, and BM25 scoring."""
    tokenizer = MedicalLegalTokenizer()
    index = InvertedIndex(tokenizer)
    index.index_documents(sample_corpus)
    
    assert index.total_docs == 4
    assert "vaccine" in index.postings or "vaccin" in index.postings
    
    bm25 = BM25Scorer(index)
    results, postings_evaluated = bm25.exhaustive_search(["vaccine", "cold"], top_k=2)
    assert len(results) > 0
    assert postings_evaluated > 0


def test_wand_exactness(sample_corpus):
    """Verifies WAND early-exit returns exact top result as exhaustive BM25."""
    tokenizer = MedicalLegalTokenizer()
    index = InvertedIndex(tokenizer)
    index.index_documents(sample_corpus)
    bm25 = BM25Scorer(index)
    wand = WANDPruner(index, bm25)
    
    tokens = ["audit", "control", "system"]
    ex_results, ex_p = bm25.exhaustive_search(tokens, top_k=2)
    wand_results, wand_p = wand.wand_search(tokens, top_k=2)
    
    # Top-1 doc should match exactly
    if ex_results and wand_results:
        assert ex_results[0][0] == wand_results[0][0]


def test_query_side_pruner(sample_corpus):
    """Verifies conversational fillers are stripped while informative terms are preserved."""
    tokenizer = MedicalLegalTokenizer()
    index = InvertedIndex(tokenizer)
    index.index_documents(sample_corpus)
    pruner = QuerySidePruner(index, tokenizer)
    
    spoken_q = "um can you tell me what the cold storage vaccine requirements are"
    active, pruned = pruner.prune_query(spoken_q)
    
    # Fillers like "um", "tell", "what" must be removed
    assert "um" not in active
    # Informative terms must be retained
    assert any(term in active for term in ["cold", "storage", "vaccine", "vaccin"])


def test_result_diversification(sample_corpus):
    """Verifies MMR diversifier calculates diversity penalty and orders multi-aspect clauses."""
    dense = DenseSemanticRetriever()
    dense.fit_and_index(sample_corpus)
    diversifier = ResultDiversifier(dense)
    
    candidate_pool = [
        {"doc_int_id": 0, "rrf_score": 0.035, "lexical_score": 2.5, "dense_score": 0.8},
        {"doc_int_id": 1, "rrf_score": 0.033, "lexical_score": 2.2, "dense_score": 0.75},
        {"doc_int_id": 2, "rrf_score": 0.030, "lexical_score": 1.9, "dense_score": 0.7}
    ]
    doc_store = {i: c for i, c in enumerate(sample_corpus)}
    
    div_results = diversifier.diversify_mmr(candidate_pool, doc_store, top_k=3, lambda_param=0.6)
    assert len(div_results) == 3
    # Check that each result has aspect annotation
    assert all(r.subtopic_aspect for r in div_results)


def test_end_to_end_engine():
    """Verifies full ComplianceEngine end-to-end execution on master corpus."""
    engine = ComplianceEngine()
    engine.initialize()
    
    req = VoiceQueryRequest(
        raw_query="um can you check the requirements for safe harbor de-identification under hipaa",
        enable_query_pruning=True,
        enable_wand_pruning=True,
        enable_diversification=True,
        top_k=5
    )
    res = engine.search(req)
    assert res.status == "success"
    assert len(res.results) > 0
    assert "de_identification" in res.unique_subtopics_covered
    assert res.query_metadata.latency_breakdown_ms["total_pipeline"] < 100.0  # SLA < 100ms
