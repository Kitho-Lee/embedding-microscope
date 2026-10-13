import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.engine import evaluate, encode
client=TestClient(app)

def config():
    return dict(query="alpha",variant="beta",kind="condition_change",
                documents=[dict(id="A",text="alpha"),dict(id="B",text="beta")],
                relevant_before=["A"],relevant_after=["B"],models=["demo-word","demo-char"],k=1)

def test_paired_retrieval():
    response=client.post("/api/experiment",json=config())
    assert response.status_code==200
    for result in response.json()["results"]:
        assert result["before"]["recall"]==1
        assert result["after"]["recall"]==1
        assert result["top_k_overlap"]==0
        assert not result["after"]["failure"]

def test_failure_and_metrics():
    result=evaluate("alpha",config()["documents"],["B"],"demo-word",1)
    assert result["failure"]
    assert result["recall"]==0
    assert result["mrr"]==.5
    assert result["ndcg"]==0

def test_label_validation():
    for change in [dict(relevant_after=["X"]),dict(kind="meaning_preserving"),dict(k=3),dict(models=["demo-word"]*2)]:
        assert client.post("/api/experiment",json=config()|change).status_code==422

def test_demo_deterministic_and_normalized():
    x=encode(["alpha beta",""],"demo-word","query")
    assert np.array_equal(x,encode(["alpha beta",""],"demo-word","query"))
    assert np.isclose(np.linalg.norm(x[0]),1)
    assert np.all(x[1]==0)

def test_index():
    assert client.get("/").status_code==200
