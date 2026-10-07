import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters feature terms based on variance across documents.
        
        Args:
            threshold: Minimum variance threshold. Terms with variance <= threshold are removed.
            
        Returns:
            Dictionary containing result summary (n_kept, n_removed, kept_terms, result_id).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}
        
        # Convert sparse matrix to dense array or compute variance directly per column
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
            variances = np.var(X_dense, axis=0)
        else:
            variances = np.var(X, axis=0)
            
        feature_names = list(session.feature_names)
        
        # Build full report DataFrame with term as first column, variance as second column
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })
        
        # Sort full report by variance descending
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)
        
        # Filter terms based on threshold
        kept_mask = report_df["variance"] > threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]
        
        all_kept_terms = kept_df["term"].tolist()
        n_kept = len(all_kept_terms)
        n_removed = len(removed_df)
        
        # Cap kept_terms to top 20 in returned summary for JSON serialization conciseness
        kept_terms_summary = all_kept_terms[:20]
        
        res_id = session.next_result_id("variance_filter")
        
        summary = {
            "result_id": res_id,
            "n_kept": n_kept,
            "n_removed": n_removed,
            "kept_terms": kept_terms_summary
        }
        
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )
        
        return summary

    @tool
    def pearson_filter_tool(category: str) -> dict:
        """Computes Pearson correlation between each term's frequency and a binary target category indicator.

        Args:
            category: Target category label to correlate terms against (one-vs-rest).

        Returns:
            Dictionary containing result_id, n_terms, category, and top 20 kept_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix or feature names found in session. Run build_dtm_tool first."}

        # Resolve labels from session
        labels = session.labels
        if labels is None and session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            elif "label" in session.dataframe.columns:
                labels = session.dataframe["label"].values

        if labels is None:
            return {"error": "No labels found in session to compute category correlation."}

        # Create binary target vector (1 for matching category, 0 otherwise)
        target = (np.array(labels) == category).astype(float)

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        feature_names = list(session.feature_names)
        pearson_rs = []

        for col in range(X_dense.shape[1]):
            col_vals = X_dense[:, col]
            # Handle zero variance edge case to avoid NaN
            if np.std(col_vals) == 0 or np.std(target) == 0:
                r_val = 0.0
            else:
                r_val, _ = pearsonr(col_vals, target)
                if np.isnan(r_val):
                    r_val = 0.0
            pearson_rs.append(r_val)

        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs
        })

        # Sort by absolute correlation value descending, highest absolute correlation first
        report_df["abs_r"] = report_df["pearson_r"].abs()
        report_df = report_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        all_terms = report_df["term"].tolist()
        kept_terms_summary = all_terms[:20]

        res_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": res_id,
            "category": category,
            "n_terms": len(feature_names),
            "kept_terms": kept_terms_summary
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"category": category},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool]
