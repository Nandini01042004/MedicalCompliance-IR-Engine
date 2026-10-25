"""
Scalability and Dynamic Pruning Benchmark Suite.
Evaluates WAND vs Exhaustive scoring across scaling corpus sizes (100 -> 1,000 -> 5,000 -> 10,000 chunks)
to demonstrate computational efficiency, postings skipped, and P99 latency characteristics.
"""

import time
import sys
import copy
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.common.schemas import DocumentChunk
from src.member_a_core.tokenizer import MedicalLegalTokenizer
from src.member_a_core.inverted_index import InvertedIndex
from src.member_a_core.bm25 import BM25Scorer
from src.member_a_core.wand_pruner import WANDPruner
from src.common.config import CORPUS_FILE
import json


def generate_scaled_corpus(base_chunks: List[DocumentChunk], target_size: int) -> List[DocumentChunk]:
    """Generates a realistic scaled corpus by simulating institutional SOPs and site amendments."""
    scaled = []
    base_len = len(base_chunks)
    sites = ["Hospital_NY", "Pharma_Site_BOM", "Trial_Site_DEL", "Lab_Apex_BLR", "Clinic_TX", "Bio_Facility_LON"]
    
    for i in range(target_size):
        base_item = base_chunks[i % base_len]
        site = sites[i % len(sites)]
        doc_copy = DocumentChunk(
            doc_id=f"{base_item.doc_id}_INST_{i}",
            jurisdiction=base_item.jurisdiction,
            statute=f"{base_item.statute} (Institution Policy Revision {i // base_len + 1})",
            part=base_item.part,
            clause_reference=f"{base_item.clause_reference} [Site: {site}]",
            subtopic_tag=base_item.subtopic_tag,
            title=f"{base_item.title} - Operational Policy {site}",
            text=f"{base_item.text} Standard Operating Procedure executed at facility {site} under supervisor {i % 12}.",
            mandatory_level=base_item.mandatory_level,
            entities=base_item.entities + [site, f"policy_{i}"]
        )
        scaled.append(doc_copy)
    return scaled


def run_scalability_test():
    """Runs scalability tests across multiple corpus sizes."""
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        base_data = [DocumentChunk(**item) for item in json.load(f)]

    tokenizer = MedicalLegalTokenizer()
    test_queries = [
        "cold storage temperature requirements sensitive vaccines data logger",
        "patient health records breach notification sixty days media reporting",
        "cleanroom air lock positive pressure differential HEPA ventilation",
        "electronic signature password two distinct components validation",
        "schedule H1 prescription register retention three years warning box"
    ]

    corpus_sizes = [100, 500, 2000, 5000]
    results = []

    print("\n" + "=" * 110)
    print("SCALABILITY BENCHMARK: EXHAUSTIVE BM25 vs WAND DYNAMIC PRUNING")
    print("=" * 110)

    for size in corpus_sizes:
        print(f"\nBuilding & Indexing Corpus of size N = {size} chunks ...")
        scaled_corpus = generate_scaled_corpus(base_data, size)
        
        index = InvertedIndex(tokenizer)
        index.index_documents(scaled_corpus)
        bm25 = BM25Scorer(index)
        wand = WANDPruner(index, bm25)

        ex_postings_list = []
        wand_postings_list = []
        ex_latencies = []
        wand_latencies = []

        for q_text in test_queries:
            tokens = tokenizer.tokenize(q_text)
            
            # 1. Exhaustive BM25
            t0 = time.perf_counter()
            _, p_ex = bm25.exhaustive_search(tokens, top_k=10)
            ex_lat = (time.perf_counter() - t0) * 1000
            ex_postings_list.append(p_ex)
            ex_latencies.append(ex_lat)

            # 2. WAND Pruning
            t0 = time.perf_counter()
            _, p_wand = wand.wand_search(tokens, top_k=10)
            wand_lat = (time.perf_counter() - t0) * 1000
            wand_postings_list.append(p_wand)
            wand_latencies.append(wand_lat)

        avg_ex_p = np.mean(ex_postings_list)
        avg_wand_p = np.mean(wand_postings_list)
        pct_skipped = ((avg_ex_p - avg_wand_p) / max(1, avg_ex_p)) * 100.0

        results.append({
            "size": size,
            "avg_ex_postings": round(avg_ex_p, 1),
            "avg_wand_postings": round(avg_wand_p, 1),
            "pct_skipped": round(pct_skipped, 1),
            "ex_latency_ms": round(float(np.mean(ex_latencies)), 2),
            "wand_latency_ms": round(float(np.mean(wand_latencies)), 2),
            "speedup": round(float(np.mean(ex_latencies)) / max(0.01, float(np.mean(wand_latencies))), 2)
        })

    print("\n" + "=" * 110)
    print("SCALABILITY BENCHMARK RESULTS TABLE")
    print("=" * 110)
    header = f"{'Corpus Size (N)':<16} | {'Exhaustive Postings':<20} | {'WAND Postings':<15} | {'Postings Skipped %':<20} | {'Ex Lat (ms)':<12} | {'WAND Lat (ms)':<13}"
    print(header)
    print("-" * 110)
    for r in results:
        row = (
            f"{r['size']:<16} | "
            f"{r['avg_ex_postings']:<20} | "
            f"{r['avg_wand_postings']:<15} | "
            f"{r['pct_skipped']:<19.1f}% | "
            f"{r['ex_latency_ms']:<12.2f} | "
            f"{r['wand_latency_ms']:<13.2f}"
        )
        print(row)
    print("=" * 110 + "\n")


if __name__ == "__main__":
    run_scalability_test()
