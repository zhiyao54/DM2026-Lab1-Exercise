import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_inspect_data_tool_spec_cases():
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

    # Get student-built inspect_data_tool bound to same session
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 1) offset=7, n_rows=1 -> check text == "" and text_truncated == False
    res_7 = inspect_data_tool.invoke({"offset": 7, "n_rows": 1})
    assert res_7["status"] == "success"
    assert len(res_7["sample_rows"]) == 1
    row_7 = res_7["sample_rows"][0]
    assert row_7["text"] == ""
    assert row_7["text_truncated"] == False

    # 2) max_text_len=5 on row 0 -> text is truncated ("always alpha alpha alpha beta" -> "alway...") and text_truncated == True
    res_0 = inspect_data_tool.invoke({"offset": 0, "n_rows": 1, "max_text_len": 5})
    assert res_0["status"] == "success"
    assert len(res_0["sample_rows"]) == 1
    row_0 = res_0["sample_rows"][0]
    assert row_0["text"] == "alway..."
    assert row_0["text_truncated"] == True

    # 3) offset=10 -> out of bounds -> returns status == "error"
    res_err = inspect_data_tool.invoke({"offset": 10, "n_rows": 1})
    assert res_err["status"] == "error"
    assert "Invalid offset" in res_err["message"]

    # 4) category is isinstance(..., int)
    res_type = inspect_data_tool.invoke({"offset": 0, "n_rows": 1})
    assert res_type["status"] == "success"
    cat_val = res_type["sample_rows"][0]["category"]
    assert isinstance(cat_val, int)
    assert type(cat_val) is int


def test_check_missing_tool_known_answer_2():
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

    # Get student-built check_missing_tool bound to same session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Call check_missing_tool
    res = check_missing_tool.invoke({})

    # Known answer 2 assertions:
    assert res["status"] == "success"
    assert res["missing_count"] == 1
    assert type(res["missing_count"]) is int
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
    # Test 1: drop=False (default)
    session1 = SessionState()
    premade_tools1, _ = load_workspace_tools("premade_tools", session1)
    load_dataset_tool1 = next(t for t in premade_tools1 if t.name == "load_dataset_tool")
    load_dataset_tool1.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools1 = make_tools(session1)
    check_duplicates_tool1 = next(t for t in tools1 if t.name == "check_duplicates_tool")

    res1 = check_duplicates_tool1.invoke({"drop": False})
    assert res1["status"] == "success"
    assert res1["duplicate_count"] == 1
    assert type(res1["duplicate_count"]) is int
    assert res1["duplicate_indices"] == [6]
    assert res1["rows_remaining"] == 8

    # Test 2: drop=True in a fresh session
    session2 = SessionState()
    premade_tools2, _ = load_workspace_tools("premade_tools", session2)
    load_dataset_tool2 = next(t for t in premade_tools2 if t.name == "load_dataset_tool")
    load_dataset_tool2.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools2 = make_tools(session2)
    check_duplicates_tool2 = next(t for t in tools2 if t.name == "check_duplicates_tool")

    res2 = check_duplicates_tool2.invoke({"drop": True})
    assert res2["status"] == "success"
    assert res2["duplicate_count"] == 1
    assert res2["duplicate_indices"] == [6]
    assert res2["rows_remaining"] == 6
    assert len(session2.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset with label_column="label"
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Get student-built sample_data_tool bound to same session
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Invoke with n=4, random_state=42
    res = sample_data_tool.invoke({
        "n": 4,
        "random_state": 42
    })

    # Assertions per Known Answer 13 & actual sample_data_tool schema:
    assert res["status"] == "success"
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert isinstance(res["counts_per_category"], dict)
    # Ensure original session.dataframe remains intact (8 rows)
    assert len(session.dataframe) == 8
