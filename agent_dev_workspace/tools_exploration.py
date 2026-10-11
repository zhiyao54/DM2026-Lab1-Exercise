
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_idx1: int, doc_idx2: int) -> dict:
        """Computes cosine similarity between two documents using raw feature counts.

        If either document index is out of bounds or has zero norm (e.g., empty text),
        handles it cleanly and returns 0.0 similarity.

        Args:
            doc_idx1 (int): Index of the first document (0-based).
            doc_idx2 (int): Index of the second document (0-based).

        Returns:
            dict: Summary containing result_id, doc_idx1, doc_idx2, and similarity score.
        """
        if session.feature_matrix is None or session.dataframe is None:
            return {"error": "Feature matrix not found. Please run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_idx1 < 0 or doc_idx1 >= n_docs or doc_idx2 < 0 or doc_idx2 >= n_docs:
            return {"error": f"Document index out of bounds. Must be between 0 and {n_docs - 1}."}

        vec1 = session.feature_matrix[doc_idx1]
        vec2 = session.feature_matrix[doc_idx2]

        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0 or norm2 == 0:
            sim_val = 0.0
        else:
            sim_matrix = cosine_similarity(vec1, vec2)
            sim_val = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": float(sim_val),
        }

        session.store_result("cosine_similarity_tool", {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2}, summary)
        return summary

    return [cosine_similarity_tool]
