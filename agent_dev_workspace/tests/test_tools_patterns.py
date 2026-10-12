from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_tool_fpgrowth_and_maxfpgrowth():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #17: filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    # catA keeps alpha (support 3), catB keeps gamma (support 4)
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA["note"] == "success"
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["items"] == ["alpha"]
    assert res_catA["patterns"][0]["support"] == 3

    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catB["note"] == "success"
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["items"] == ["gamma"]
    assert res_catB["patterns"][0]["support"] == 4

    # Known answer #17: variance and term_frequency keep zero terms for both categories
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["patterns"] == []
    assert res_var["note"] == "no terms survived filtering"

    # Known answer #20: maxfpgrowth returns zero patterns for both categories even with tfidf
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["patterns"] == []


def test_mine_patterns_tool_topk():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Known answer #20: algorithm="topk", k=1
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk["note"] == "success"
    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["items"] == ["alpha"]
    assert res_topk["patterns"][0]["support"] == 3
