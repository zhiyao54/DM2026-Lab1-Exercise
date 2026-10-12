import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools
from tools_dtm import make_tools as make_dtm_tools


def test_binarize_labels_counts_and_types():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Instantiate student's tools_reduction tools
    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert res.get("status") == "success"
    assert res.get("column_order") == ["catA", "catB"]

    counts = res.get("counts_per_category", {})
    assert counts.get("catA") == 4
    assert counts.get("catB") == 4

    # Check whether counts are native Python int or numpy int64
    for cat, cnt in counts.items():
        assert type(cnt) is int, f"Count for '{cat}' is {type(cnt)}, expected native int"
        assert not isinstance(cnt, np.integer), f"Count for '{cat}' is numpy type {type(cnt)}, expected native int"

    # Check shape of stored binarized labels artifact
    assert session.artifacts["binarized_labels"].shape == (8, 2)


def test_reduce_dimensions_tool_pca_tsne_umap():
    session = SessionState()

    # 1. Load premade tools and load fixture dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # 2. Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Load reduce_dimensions_tool
    reduction_tools = make_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Test PCA (Known Answer #7)
    pca_res = reduce_tool.invoke({"method": "pca"})
    assert pca_res["status"] == "success"
    assert pca_res["method"] == "pca"

    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    pca_coords = np.array(pca_res["coordinates"])
    assert np.allclose(pca_coords, expected_pca_coords, atol=1e-3)

    expected_var_ratio = [0.7532, 0.1965]
    assert np.allclose(pca_res["explained_variance_ratio"], expected_var_ratio, atol=1e-3)

    # Verify pending figure exists
    assert session.pending_figure is not None

    # Test t-SNE
    session.pending_figure = None
    tsne_res = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42,
    })
    assert tsne_res["status"] == "success"
    tsne_coords = np.array(tsne_res["coordinates"])
    assert tsne_coords.shape == (8, 2)
    # Row 5 and Row 6 are exact duplicates in the fixture dataset
    assert np.allclose(tsne_coords[5], tsne_coords[6], atol=1e-3)
    assert session.pending_figure is not None

    # Test UMAP
    session.pending_figure = None
    umap_res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })
    assert umap_res["status"] == "success"
    umap_coords = np.array(umap_res["coordinates"])
    assert umap_coords.shape == (8, 2)
    assert session.pending_figure is not None
