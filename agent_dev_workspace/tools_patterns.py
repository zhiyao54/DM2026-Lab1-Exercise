import os
import tempfile
import pandas as pd
import numpy as np
from typing import Optional, Literal, List, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent itemsets/patterns for a specific document category using PAMI algorithms.

        Args:
            category_name: Name of category to filter and mine.
            filtering_method: Vocabulary filter technique ("variance", "tfidf", "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", "maxfpgrowth").
            min_sup: Minimum support count (required for fpgrowth and maxfpgrowth).
            k: Top-K count (required for topk algorithm).

        Returns:
            Dict containing mined patterns, support counts, and status.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No DataFrame loaded in session."}

        # Filter dataframe to category
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"Category '{category_name}' not found or has no documents."}

        # Step 1: Fresh CountVectorizer with English stop words on local category text
        texts = df_cat["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(texts)
        except ValueError:
            # Vocabulary empty (e.g. all stop words or empty strings)
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        dtm = dtm_sparse.toarray()

        if len(feature_names) == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Apply chosen vocabulary filtering
        keep_mask = np.ones(len(feature_names), dtype=bool)

        if filtering_method == "variance":
            variances = np.var(dtm, axis=0)
            low, high = np.percentile(variances, 5), np.percentile(variances, 95)
            keep_mask = (variances > low) & (variances < high)
        elif filtering_method == "term_frequency":
            counts = np.sum(dtm, axis=0)
            low, high = np.percentile(counts, 5), np.percentile(counts, 95)
            keep_mask = (counts > low) & (counts < high)
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(dtm_sparse).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            threshold = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= threshold

        surviving_terms = feature_names[keep_mask]
        surviving_dtm = dtm[:, keep_mask]

        # Step 3: Check if terms survived
        if len(surviving_terms) == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 4: Convert to transactional DB and mine via PAMI
        # Convert surviving DTM matrix to binary dataframe with surviving term column names
        df_surviving = pd.DataFrame(surviving_dtm, columns=surviving_terms)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
            tmp_db_path = tmp_file.name

        try:
            converter = db_converter.DF2DB(df_surviving)
            converter.convert2TransactionalDatabase(tmp_db_path, ">=", 1)

            if algorithm in ("fpgrowth", "maxfpgrowth"):
                if min_sup is None:
                    return {"error": f"min_sup is required for algorithm '{algorithm}'."}

                if algorithm == "fpgrowth":
                    miner = FPGrowth.FPGrowth(iFile=tmp_db_path, minSup=min_sup, sep="\t")
                else:
                    miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                if k is None:
                    return {"error": "k is required for algorithm 'topk'."}
                miner = FAE.FAE(iFile=tmp_db_path, k=k, sep="\t")
            else:
                return {"error": f"Unsupported algorithm '{algorithm}'."}

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

            # Format patterns
            patterns_list = []
            if patterns_df is not None and not patterns_df.empty:
                # PAMI returns columns like 'Itemset' / 'Patterns' / 'itemSets' and 'Support' / 'support'
                col_item = [c for c in patterns_df.columns if "item" in c.lower() or "pattern" in c.lower()]
                col_sup = [c for c in patterns_df.columns if "sup" in c.lower()]

                item_col = col_item[0] if col_item else patterns_df.columns[0]
                sup_col = col_sup[0] if col_sup else patterns_df.columns[1]

                for _, row in patterns_df.head(20).iterrows():
                    raw_items = row[item_col]
                    raw_sup = row[sup_col]

                    if isinstance(raw_items, str):
                        items = [x.strip() for x in raw_items.split("\t") if x.strip()]
                        if not items:
                            items = [x.strip() for x in raw_items.split(" ") if x.strip()]
                    elif isinstance(raw_items, (list, tuple)):
                        items = [str(x) for x in raw_items]
                    else:
                        items = [str(raw_items)]

                    patterns_list.append({
                        "items": items,
                        "support": int(float(raw_sup)),
                    })

            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": patterns_list,
                "note": "success" if patterns_list else "no patterns found",
            }

            artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
            session.artifacts[artifact_key] = patterns_list
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)

            return summary

        finally:
            if os.path.exists(tmp_db_path):
                try:
                    os.remove(tmp_db_path)
                except OSError:
                    pass

    return [mine_patterns_tool]
