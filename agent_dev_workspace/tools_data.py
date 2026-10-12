import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from typing import Dict, Any, List
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists directory folders and files in a specified subdirectory.

        Args:
            subdirectory: The relative path of the directory to list. Defaults to '.'.

        Returns:
            A dictionary summary containing result_id, subdirectory, and lists of directories and files.
        """
        if not os.path.exists(subdirectory):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' does not exist."
            }

        if not os.path.isdir(subdirectory):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory."
            }

        entries = os.listdir(subdirectory)
        directories = sorted([e for e in entries if os.path.isdir(os.path.join(subdirectory, e))])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(subdirectory, e))])

        result_id = session.next_result_id("list_files")
        summary = {
            "status": "success",
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

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

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text for each document in session.dataframe using nltk.word_tokenize on lowercased text.

        Saves tokenized unigrams into session.dataframe['unigrams']. Handles empty or missing text as [].

        Returns:
            A summary dictionary containing result_id, n_documents, and first_3_documents.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe

        def _tokenize_val(val):
            if pd.isna(val) or not str(val).strip():
                return []
            return nltk.word_tokenize(str(val).lower())

        df["unigrams"] = df["text"].apply(_tokenize_val)

        unigrams_list = df["unigrams"].tolist()
        first_3_documents = unigrams_list[:3]
        n_documents = len(df)

        result_id = session.next_result_id("tokenize")
        summary = {
            "status": "success",
            "result_id": result_id,
            "n_documents": n_documents,
            "first_3_documents": first_3_documents,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes summary statistics on document text length (overall and per category_name) and generates a boxplot.

        Returns:
            A dictionary containing result_id, overall statistics, and statistics per category_name.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session.dataframe. Call load_dataset_tool first."
            }

        df = session.dataframe

        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "DataFrame does not contain a 'text' column."
            }

        # Compute text character length, counting missing/NA as 0
        text_length = df["text"].apply(lambda x: len(x) if pd.notna(x) else 0)
        df_temp = df.copy()
        df_temp["text_length"] = text_length

        def _clean_describe_dict(series: pd.Series) -> Dict[str, float]:
            desc = series.describe().to_dict()
            return {k: float(v) for k, v in desc.items()}

        overall_stats = _clean_describe_dict(df_temp["text_length"])

        per_category_stats = {}
        if "category_name" in df_temp.columns:
            for cat_name, group in df_temp.groupby("category_name"):
                per_category_stats[str(cat_name)] = _clean_describe_dict(group["text_length"])

        # Plot boxplot using Seaborn
        fig, ax = plt.subplots(figsize=(8, 6))
        if "category_name" in df_temp.columns:
            sns.boxplot(data=df_temp, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df_temp, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "status": "success",
            "result_id": result_id,
            "overall": overall_stats,
            "per_category": per_category_stats,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        describe_data_tool,
    ]
