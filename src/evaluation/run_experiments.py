"""
Ablation Study and Benchmark Evaluation Runner.
Executes systematic comparisons across:
- BM25 Exhaustive
- BM25 + WAND Pruning
- Dense FAISS
- Hybrid RRF (No Diversification)
- Hybrid + MMR Diversified
- Hybrid + xQuAD Diversified
Produces academic-grade IR evaluation tables with MAP, NDCG, alpha-NDCG, S-Recall, ILAD, and Latency.
"""

import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.config import QRELS_FILE, CORPUS_FILE
from src.common.schemas import VoiceQueryRequest
from src.engine import ComplianceEngine
from src.evaluation.ir_metrics import IRMetrics


def run_ablation_benchmarks() -> List[Dict[str, Any]]:
    """Runs complete comparative evaluation suite across all benchmark queries."""
    with open(QRELS_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        raw_corpus = json.load(f)

    # Build mapping from doc_id to subtopic aspects
    doc_aspects_map: Dict[str, set] = {}
    doc_id_to_int: Dict[str, int] = {}
    for idx, doc in enumerate(raw_corpus):
        doc_aspects_map[doc["doc_id"]] = {doc["subtopic_tag"]}
        doc_id_to_int[doc["doc_id"]] = idx

    engine = ComplianceEngine()
    engine.initialize()

    variants = [
        {
            "name": "1. BM25 (Exhaustive)",
            "wand": False,
            "query_prune": False,
            "diversify": False,
            "method": "none",
            "force_lexical_only": True
        },
        {
            "name": "2. BM25 + WAND Pruned",
            "wand": True,
            "query_prune": False,
            "diversify": False,
            "method": "none",
            "force_lexical_only": True
        },
        {
            "name": "3. Dense FAISS Only",
            "wand": False,
            "query_prune": False,
            "diversify": False,
            "method": "none",
            "force_dense_only": True
        },
        {
            "name": "4. Hybrid RRF (No Div)",
            "wand": True,
            "query_prune": True,
            "diversify": False,
            "method": "none",
            "force_lexical_only": False
        },
        {
            "name": "5. Hybrid + MMR Div",
            "wand": True,
            "query_prune": True,
            "diversify": True,
            "method": "mmr",
            "lambda_param": 0.65
        },
        {
            "name": "6. Hybrid + xQuAD Div",
            "wand": True,
            "query_prune": True,
            "diversify": True,
            "method": "xquad",
            "lambda_param": 0.65
        }
    ]

    ablation_results = []

    print("\n" + "=" * 90)
    print("RUNNING IR SYSTEM BENCHMARK AND ABLATION EXPERIMENTS")
    print(f"Total Benchmark Queries: {len(benchmarks)}")
    print("=" * 90 + "\n")

    for v in variants:
        variant_name = v["name"]
        print(f"Evaluating System Variant: {variant_name} ...")

        map_scores = []
        ndcg_scores = []
        alpha_ndcg_scores = []
        s_recall_scores = []
        ilad_scores = []
        latencies = []
        postings_counts = []

        for b in benchmarks:
            raw_q = b["raw_spoken_query"]
            qrels = b["qrels"]
            subtopics = b["subtopics"]

            t0 = time.perf_counter()

            if v.get("force_lexical_only"):
                # Pure BM25
                tokens = engine.tokenizer.tokenize(raw_q)
                if v["wand"]:
                    results, p_eval = engine.wand.wand_search(tokens, top_k=10)
                else:
                    results, p_eval = engine.bm25.exhaustive_search(tokens, top_k=10)
                retrieved_ids = [engine.index.int_to_doc_id[d_id] for d_id, _ in results]
                postings_counts.append(p_eval)
            elif v.get("force_dense_only"):
                # Pure Dense FAISS
                results = engine.dense.search(raw_q, top_k=10)
                retrieved_ids = [engine.dense.doc_int_to_id[d_id] for d_id, _ in results]
                postings_counts.append(0)
            else:
                req = VoiceQueryRequest(
                    raw_query=raw_q,
                    enable_query_pruning=v["query_prune"],
                    enable_wand_pruning=v["wand"],
                    enable_diversification=v["diversify"],
                    diversification_method=v.get("method", "mmr"),
                    lambda_param=v.get("lambda_param", 0.65),
                    top_k=10
                )
                res = engine.search(req)
                retrieved_ids = [r.doc.doc_id for r in res.results]
                postings_counts.append(res.query_metadata.postings_evaluated_wand)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Evaluate Metrics
            map_scores.append(IRMetrics.average_precision(retrieved_ids, qrels))
            ndcg_scores.append(IRMetrics.ndcg_at_k(retrieved_ids, qrels, k=10))
            alpha_ndcg_scores.append(IRMetrics.alpha_ndcg_at_k(retrieved_ids, doc_aspects_map, subtopics, k=10))
            s_recall_scores.append(IRMetrics.subtopic_recall_at_k(retrieved_ids, doc_aspects_map, subtopics, k=10))

            # Retrieve vectors for ILAD
            vectors = []
            for d_id in retrieved_ids:
                if d_id in doc_id_to_int:
                    vec = engine.dense.get_document_vector(doc_id_to_int[d_id])
                    if vec is not None:
                        vectors.append(vec)
            ilad_scores.append(IRMetrics.intra_list_average_distance(vectors))

        summary = {
            "variant": variant_name,
            "MAP": round(float(np.mean(map_scores)), 4),
            "NDCG@10": round(float(np.mean(ndcg_scores)), 4),
            "alpha_NDCG@10": round(float(np.mean(alpha_ndcg_scores)), 4),
            "S_Recall@10": round(float(np.mean(s_recall_scores)), 4),
            "ILAD": round(float(np.mean(ilad_scores)), 4),
            "Avg_Latency_ms": round(float(np.mean(latencies)), 2),
            "P99_Latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "Avg_Postings_Eval": round(float(np.mean(postings_counts)), 1)
        }
        ablation_results.append(summary)

    # Print Table
    print("\n" + "=" * 115)
    print("EXPERIMENTAL ABLATION TABLE (BENCHMARK RESULTS)")
    print("=" * 115)
    header = f"{'System Variant':<26} | {'MAP':<7} | {'NDCG@10':<7} | {'a-NDCG@10':<9} | {'S-Rec@10':<8} | {'ILAD':<7} | {'Avg Lat (ms)':<12} | {'P99 (ms)':<8}"
    print(header)
    print("-" * 115)
    for r in ablation_results:
        row = (
            f"{r['variant']:<26} | "
            f"{r['MAP']:<7.4f} | "
            f"{r['NDCG@10']:<7.4f} | "
            f"{r['alpha_NDCG@10']:<9.4f} | "
            f"{r['S_Recall@10']:<8.4f} | "
            f"{r['ILAD']:<7.4f} | "
            f"{r['Avg_Latency_ms']:<12.2f} | "
            f"{r['P99_Latency_ms']:<8.2f}"
        )
        print(row)
    print("=" * 115 + "\n")

    return ablation_results


if __name__ == "__main__":
    run_ablation_benchmarks()
