from typing import Literal, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> dict:
        """Reduces feature_matrix dimensionality to 2D using PCA, t-SNE, or UMAP.

        Generates a 2D scatter plot colored by category, assigns it to
        session.pending_figure, and stores the reduced coordinates in artifacts.

        Args:
            method: Reduction technique ('pca', 'tsne', or 'umap'). Default 'pca'.
            perplexity: Perplexity parameter for t-SNE. Default 30.0.
            n_neighbors: Number of neighbors parameter for UMAP. Default 15.
            random_state: Random seed for reproducibility in t-SNE / UMAP / PCA.

        Returns:
            dict: Summary containing result_id, method, coordinates (first 20 rows), and explained_variance_ratio (if PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found – call build_dtm_tool first."}

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X = X.toarray()

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Unsupported method: {method}"}

        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        # Get category names for plotting labels
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            categories = list(session.labels)
        else:
            categories = ["Unassigned"] * len(coords)

        # Plot coordinates colored by category
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1],
            "Category": categories,
        })
        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=100,
            ax=ax,
        )
        ax.set_title(f"2D Projection via {method.upper()}")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method,
            "n_documents": len(coords),
            "coordinates": coords_list[:20],
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary=summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes category_name from session.dataframe into a 2D binary matrix.

        Stores the binary matrix in session.artifacts['binarized_labels'] and
        returns category column order, counts, and example row vectors.

        Returns:
            dict: Summary containing status, column_order, counts_per_category,
            and example_vectors, or an error message if no dataset is loaded.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded – call load_dataset_tool first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "'category_name' column not found in session.dataframe."}

        categories_series = session.dataframe["category_name"]

        # Instantiate OneHotEncoder with sparse_output=False for a dense 2D array
        encoder = OneHotEncoder(sparse_output=False)
        category_array = categories_series.values.reshape(-1, 1)
        matrix = encoder.fit_transform(category_array)

        # Get column names corresponding to categories in fitted order
        column_order = [str(cat) for cat in encoder.categories_[0]]

        # Store the complete 2D matrix and column names in artifacts
        session.artifacts["binarized_labels"] = matrix
        session.artifacts["binarized_label_names"] = column_order

        # Compute count per category
        counts_per_category = categories_series.value_counts().to_dict()

        # Build example vectors for the first few rows (up to 3)
        n_examples = min(3, len(matrix))
        example_vectors = []
        for i in range(n_examples):
            row_dict = {
                col: int(matrix[i, j]) for j, col in enumerate(column_order)
            }
            example_vectors.append(row_dict)

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "column_order": column_order,
            "counts_per_category": counts_per_category,
            "example_vectors": example_vectors,
            "matrix_shape": list(matrix.shape),
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
