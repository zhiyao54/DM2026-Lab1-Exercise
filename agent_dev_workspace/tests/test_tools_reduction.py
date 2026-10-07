import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def test_binarize_labels_counts_and_types():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
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
