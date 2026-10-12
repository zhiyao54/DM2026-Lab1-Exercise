import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes Pearson feature correlation heatmap for the top-20 terms by variance.

        Extracts up to top 20 highest-variance features from session.feature_matrix,
        computes their Pearson correlation matrix, builds an annotated heatmap, and
        stores the plot in session.pending_figure.

        Returns:
            dict: Summary containing result_id, top_terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Feature matrix not found. Please run build_dtm_tool first."}

        dtm_dense = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # Calculate variance across all documents for each feature
        variances = np.var(dtm_dense, axis=0)

        # Sort indices by variance in descending order
        top_indices = np.argsort(variances)[::-1][:20]

        top_terms = feature_names[top_indices].tolist()
        top_dtm = dtm_dense[:, top_indices]

        # Compute Pearson correlation matrix across features (columns)
        corr_matrix = np.corrcoef(top_dtm, rowvar=False)

        # If only 1 feature, handle scalar corrcoef output gracefully
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[float(corr_matrix)]])

        # Replace any potential NaN correlation values with 0.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Build figure and heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=top_terms,
            yticklabels=top_terms,
            cmap="vlag",
            ax=ax,
        )
        ax.set_title("Top 20 Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        # Convert matrix to python list of floats
        corr_list = [[float(v) for v in row] for row in corr_matrix]

        result_id = session.next_result_id("feat_corr")
        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
