import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get student-built build_dtm_tool bound to same session
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    res = build_dtm_tool.invoke({})

    # Assertions based on Known Answer 4 in TEST_FIXTURE.md
    assert res["status"] == "success"
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5, atol=1e-3)

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    dtm_res = build_dtm_tool.invoke({})
    assert dtm_res["status"] == "success"

    tf_res = term_frequency_tool.invoke({})
    assert tf_res["status"] == "success"
    assert tf_res["n_terms"] == 5

    # Known answer #14 expected term frequencies
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    top_20 = tf_res["top_20"]
    actual_freqs = {item["term"]: item["frequency"] for item in top_20}
    assert actual_freqs == expected_freqs


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Test full matrix slice default (20x20 on 8x5 matrix)
    res_default = dtm_heatmap_tool.invoke({})
    assert res_default["status"] == "success"
    assert res_default["document_indices"] == list(range(8))
    assert res_default["term_names"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_default["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test custom slice (n_terms=2, n_documents=3)
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_sliced["status"] == "success"
    assert res_sliced["document_indices"] == [0, 1, 2]
    assert res_sliced["term_names"] == ["alpha", "always"]
    assert res_sliced["matrix"] == [[3, 1], [2, 1], [1, 1]]
