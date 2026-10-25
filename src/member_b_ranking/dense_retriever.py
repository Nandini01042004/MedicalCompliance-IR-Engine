"""
Dense Semantic Vector Engine backed by FAISS and Latent Semantic Projection.
Embeds regulatory text into normalized dense vectors and executes sub-millisecond
cosine similarity vector search using FAISS IndexFlatIP.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.common.schemas import DocumentChunk
from src.common.config import EMBEDDING_DIM


class DenseSemanticRetriever:
    """
    Computes dense semantic embeddings for documents and queries,
    and indexes them into a high-performance FAISS vector index.
    """
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=2500,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.doc_int_to_id: Dict[int, str] = {}
        self.doc_vectors: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit_and_index(self, chunks: List[DocumentChunk]):
        """Fits semantic latent projection and builds FAISS IndexFlatIP."""
        corpus_texts = []
        self.doc_int_to_id.clear()

        for idx, chunk in enumerate(chunks):
            self.doc_int_to_id[idx] = chunk.doc_id
            text = f"{chunk.title} {chunk.clause_reference} {chunk.text} {' '.join(chunk.entities)}"
            corpus_texts.append(text)

        # Step 1: Compute sparse TF-IDF n-gram matrix
        tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

        # Step 2: Dense latent semantic decomposition
        # Adjust component count if corpus has fewer samples
        n_comp = min(self.embedding_dim, tfidf_matrix.shape[1] - 1, len(corpus_texts) - 1)
        n_comp = max(4, n_comp)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        dense_vecs = self.svd.fit_transform(tfidf_matrix).astype(np.float32)

        # Step 3: L2 Normalize vectors for Cosine Similarity via Inner Product
        faiss.normalize_L2(dense_vecs)
        self.doc_vectors = dense_vecs

        # Step 4: Build FAISS Index
        actual_dim = dense_vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(actual_dim)
        self.faiss_index.add(dense_vecs)
        self.is_fitted = True

    def encode(self, text: str) -> np.ndarray:
        """Encodes arbitrary query string into dense normalized vector."""
        if not self.is_fitted:
            raise RuntimeError("Dense retriever has not been fitted on corpus yet.")
        tfidf_vec = self.vectorizer.transform([text])
        dense_vec = self.svd.transform(tfidf_vec).astype(np.float32)
        faiss.normalize_L2(dense_vec)
        return dense_vec

    def search(self, query_text: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        Executes dense vector search using FAISS.
        Returns list of (doc_int_id, cosine_similarity_score).
        """
        if not self.is_fitted or self.faiss_index is None:
            return []

        query_vec = self.encode(query_text)
        scores, indices = self.faiss_index.search(query_vec, top_k)

        results: List[Tuple[int, float]] = []
        for doc_int_id, score in zip(indices[0], scores[0]):
            if doc_int_id >= 0:  # valid document index
                results.append((int(doc_int_id), float(score)))

        return results

    def get_document_vector(self, doc_int_id: int) -> Optional[np.ndarray]:
        """Returns the precomputed dense vector for a document."""
        if self.doc_vectors is not None and 0 <= doc_int_id < len(self.doc_vectors):
            return self.doc_vectors[doc_int_id]
        return None
