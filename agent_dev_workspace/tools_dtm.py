from typing import Any, Dict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function creating DTM tools bound to the given pipeline session."""

    @tool
    def build_dtm_tool() -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from the text column of session.dataframe.

        Fits a standard CountVectorizer on session.dataframe['text'], saves the sparse
        feature matrix to session.feature_matrix, vocabulary to session.feature_names,
        fitted vectorizer to session.artifacts['count_vectorizer'], and sets session labels.
        Returns matrix statistics including size and sparsity percentage.

        Returns:
            Dict containing status, result_id, n_terms, non_zero, total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe found in session. Please load a dataset first."
            }

        text_series = session.dataframe["text"].fillna("")
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(text_series)

        session.feature_matrix = X
        session.feature_names = list(vectorizer.get_feature_names_out())
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())

        n_rows, n_terms = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_rows * n_terms)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "status": "success",
            "result_id": result_id,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {}, summary)
        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates term frequencies across all documents from the document-term matrix.

        Sums counts for each term across all documents using session.feature_matrix and
        session.feature_names. Stores a full_report DataFrame containing term and frequency,
        sorted in descending order by frequency.

        Returns:
            Dict containing status, result_id, n_terms, and top_20 list of term frequencies.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-term matrix not found in session. Please run build_dtm_tool first."
            }

        counts = session.feature_matrix.sum(axis=0).A1.tolist()
        df = pd.DataFrame({
            "term": session.feature_names,
            "frequency": [int(c) for c in counts]
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        n_terms = len(df)
        top_20 = df.head(20).to_dict(orient="records")

        result_id = session.next_result_id("term_frequency")
        summary = {
            "status": "success",
            "result_id": result_id,
            "n_terms": n_terms,
            "top_20": top_20
        }

        session.store_result("term_frequency_tool", {}, summary, full_report=df)
        return summary

    return [build_dtm_tool, term_frequency_tool]
