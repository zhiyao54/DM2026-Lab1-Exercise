import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from langchain_core.tools import tool


def make_tools(session):
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

    return [binarize_labels_tool]
