import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Create a fresh session and load dataset fixture
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    assert load_result["n_documents"] == 8

    # 2. Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_result = build_dtm_tool.invoke({})
    assert dtm_result["status"] == "success"

    # 3. Test variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Check delta variance from session.results_store via full_report DataFrame
    res_id = result["result_id"]
    full_report = session.results_store[res_id]["full_report"]
    delta_variance = full_report.loc[full_report["term"] == "delta", "variance"].iloc[0]
    assert pytest.approx(delta_variance, abs=1e-3) == 0.25

    # Check kept_terms as a set: threshold=0.15 keeps alpha, beta, delta, gamma and drops always
    kept_terms = set(result["kept_terms"])
    expected_kept = {"alpha", "beta", "delta", "gamma"}
    assert kept_terms == expected_kept


def test_pearson_filter_tool():
    # 1. Fresh session per test function to prevent state pollution
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Test pearson_filter_tool against target_class / category "catB"
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    result = pearson_filter_tool.invoke({"category": "catB"})

    res_id = result["result_id"]
    full_report = session.results_store[res_id]["full_report"]

    # Known answer #6 Pearson r values for target "catB":
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson.items():
        val = full_report.loc[full_report["term"] == term, "pearson_r"].iloc[0]
        assert pytest.approx(val, abs=1e-3) == exp_r
