import pandas as pd
from typing import Dict, Any, List
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def inspect_data_tool(offset: int = 0, n_rows: int = 5, max_text_len: int = 100) -> Dict[str, Any]:
        """Peeks at a slice of the working DataFrame (session.dataframe) after loading.

        Args:
            offset: The zero-based row index to start viewing from. Defaults to 0.
            n_rows: The number of rows to return. Defaults to 5.
            max_text_len: Maximum character length for text strings before truncation. Defaults to 100.

        Returns:
            A dictionary containing dataset summary and sliced preview rows, or an error status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe
        total_docs = len(df)

        if total_docs == 0:
            return {
                "status": "error",
                "message": "The loaded dataset is empty."
            }

        if offset < 0 or offset >= total_docs:
            return {
                "status": "error",
                "message": f"Invalid offset {offset}. Dataset has {total_docs} rows (valid offsets: 0 to {total_docs - 1})."
            }

        if n_rows <= 0:
            return {
                "status": "error",
                "message": f"n_rows must be greater than 0, got {n_rows}."
            }

        # Dynamically inspect columns from the loaded dataframe
        dynamic_columns = list(df.columns)

        # Slice requested window
        df_slice = df.iloc[offset : offset + n_rows].copy()

        # Fill missing values and track truncation on text column specifically if it exists
        text_truncated_flags = []
        if "text" in df_slice.columns:
            df_slice["text"] = df_slice["text"].fillna("")
            new_texts = []
            for text_val in df_slice["text"]:
                s_val = str(text_val)
                if len(s_val) > max_text_len:
                    new_texts.append(s_val[:max_text_len] + "...")
                    text_truncated_flags.append(True)
                else:
                    new_texts.append(s_val)
                    text_truncated_flags.append(False)
            df_slice["text"] = new_texts

        # Explicitly cast category column to standard Python int if present
        if "category" in df_slice.columns:
            df_slice["category"] = df_slice["category"].astype(int)

        # Convert records to python primitives for clean JSON serialization
        sample_rows = df_slice.to_dict(orient="records")

        # Explicitly ensure all integer/category values in records are standard Python int/types and add text_truncated key
        for idx, row in enumerate(sample_rows):
            if text_truncated_flags:
                row["text_truncated"] = text_truncated_flags[idx]
            for k, v in row.items():
                if pd.isna(v):
                    row[k] = ""
                elif hasattr(v, "item"):  # converts numpy/pandas scalar types (e.g., int64) to native Python types
                    row[k] = v.item()

        result_id = session.next_result_id("inspect")
        summary = {
            "status": "success",
            "result_id": result_id,
            "total_documents": total_docs,
            "columns": dynamic_columns,
            "offset": offset,
            "n_rows_returned": len(sample_rows),
            "sample_rows": sample_rows,
        }

        session.store_result("inspect_data_tool", {"offset": offset, "n_rows": n_rows, "max_text_len": max_text_len}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing or empty text values in the loaded working DataFrame.

        Returns:
            A dictionary summary containing total documents, missing count, missing percentage, and indices of missing rows.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe
        total_rows = len(df)

        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "DataFrame does not contain a 'text' column."
            }

        # Combine checks with OR directly on the original text column
        missing_mask = df["text"].isna() | (df["text"] == "")
        missing_count = int(missing_mask.sum())
        missing_pct = float(100.0 * missing_count / total_rows) if total_rows > 0 else 0.0
        missing_indices = [int(i) for i in df.index[missing_mask].tolist()[:10]]

        result_id = session.next_result_id("check_missing")
        summary = {
            "status": "success",
            "result_id": result_id,
            "total_documents": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices,
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate rows in the loaded working DataFrame and optionally drops them.

        Args:
            drop: If True, drops duplicate rows using keep=False (removing all duplicate copies). Defaults to False.

        Returns:
            A dictionary summary containing total documents, duplicate count, duplicate indices, and remaining rows.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe
        total_rows = len(df)

        # Duplicate detection across all columns using default keep='first'
        dup_mask = df.duplicated()
        dup_indices = [int(i) for i in df[dup_mask].index.tolist()[:10]]
        dup_count = int(dup_mask.sum())

        if drop and dup_count > 0:
            df.drop_duplicates(keep=False, inplace=True)

        rows_remaining = int(len(session.dataframe))

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "status": "success",
            "result_id": result_id,
            "total_documents": total_rows,
            "duplicate_count": dup_count,
            "duplicate_indices": dup_indices,
            "rows_remaining": rows_remaining,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 1) -> Dict[str, Any]:
        """Subsamples n rows from the working DataFrame using random sampling.

        Args:
            n: Number of documents to sample. Must be between 1 and the total number of rows.
            random_state: Seed for the random number generator. Defaults to 1.

        Returns:
            A summary dictionary containing result_id, sampled row indices, and counts per category.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe
        total_rows = len(df)

        if n < 1 or n > total_rows:
            return {
                "status": "error",
                "message": f"n must be between 1 and total rows ({total_rows}), got {n}."
            }

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]

        counts_per_category = {}
        if "category_name" in sampled_df.columns:
            cat_counts = sampled_df["category_name"].value_counts()
            counts_per_category = {str(cat): int(count) for cat, count in cat_counts.items()}

        session.artifacts["sample_data"] = sampled_df

        result_id = session.next_result_id("sample_data")
        summary = {
            "status": "success",
            "result_id": result_id,
            "sampled_indices": sampled_indices,
            "counts_per_category": counts_per_category,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    return [inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool]
