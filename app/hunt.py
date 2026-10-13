"""Exhaustive cardinality-ordered search over disjoint, user-approved edits."""
from itertools import combinations
import hashlib
import json
from .engine import encode, rank_scores, MODEL_REVISIONS, REAL_MODELS, tokenization_report

def search(data):
    query=data["query"]
    spans=[]
    for rule in data["rules"]:
        source=rule["source"]
        if query.count(source)!=1:
            raise ValueError("Each source must occur exactly once in the original query")
        start=query.index(source);end=start+len(source)
        if any(start < b and end > a for a,b,_ in spans):
            raise ValueError("Rule sources must not overlap")
        spans.append((start,end,rule["replacement"]))
    def apply(indices):
        result=query
        for i in sorted(indices,key=lambda i:spans[i][0],reverse=True):
            start,end,replacement=spans[i]
            result=result[:start]+replacement+result[end:]
        return result
    candidates=[((),query)]
    for size in range(1,len(spans)+1):
        candidates += [(indices,apply(indices)) for indices in combinations(range(len(spans)),size)]
    total=len(candidates)-1
    chosen=candidates[:data["budget"]+1]
    vectors={}
    for model in data["models"]:
        documents=encode([d["text"] for d in data["documents"]],model,"document")
        queries=encode([text for _,text in chosen],model,"query")
        vectors[model]=[rank_scores(scores,data["documents"],data["relevant"],data["k"]) for scores in queries@documents.T]
    baseline={m:v[0] for m,v in vectors.items()}
    eligible=[m for m,e in baseline.items() if not e["failure"]]
    failures=[]
    for index,(indices,text) in enumerate(chosen[1:],1):
        failing=[m for m in eligible if vectors[m][index]["failure"]]
        if failing:
            failures.append({"query":text,"rule_indices":list(indices),"edit_count":len(indices),
                             "failing_models":failing,"results":{m:v[index] for m,v in vectors.items()}})
    best=failures[0] if failures else None
    return {"schema_version":1,"config":data,"baseline":baseline,"eligible_models":eligible,
            "evaluated":len(chosen)-1,"search_space":total,"exhausted":len(chosen)==len(candidates),
            "minimality":"Minimum number of approved rules among tested combinations; cardinality-ordered exhaustive prefixes. No claim about arbitrary text edits.",
            "semantic_validity":"User attests each rule and all combinations preserve relevance; not independently verified.",
            "tokenization":{m:tokenization_report({"query":query,"variant":best["query"] if best else query,"documents":data["documents"]},m) for m in data["models"]},
            "model_revisions":{m:MODEL_REVISIONS[m] for m in data["models"] if m in REAL_MODELS},
            "failure_count":len(failures),"best":best,"failures":failures}

def regression_bundle(result):
    if not result.get("best"):
        raise ValueError("No newly failing case to export")
    cfg=result["config"];best=result["best"]
    payload={"query":best["query"],"baseline_query":cfg["query"],"documents":cfg["documents"],
             "relevant":cfg["relevant"],"models":best["failing_models"],"k":cfg["k"],
             "expected":"At least one relevant document in top-k for baseline and perturbed queries",
             "rules":cfg["rules"],"rule_indices":best["rule_indices"],
             "model_revisions":result["model_revisions"],"semantic_validity":result["semantic_validity"]}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:12]
    test="""# Generated desired-behavior regression: expected to FAIL until retrieval is repaired.
import json
from pathlib import Path
import pytest
from app.engine import evaluate, MODEL_REVISIONS
CASE = json.loads(Path(__file__).with_name('case.json').read_text())
@pytest.mark.parametrize('model', CASE['models'])
def test_relevance_survives_approved_edits(model):
    if model in CASE['model_revisions']:
        assert MODEL_REVISIONS[model] == CASE['model_revisions'][model], 'Model revision changed; revalidate case'
    for query in [CASE['baseline_query'], CASE['query']]:
        result = evaluate(query, CASE['documents'], CASE['relevant'], model, CASE['k'])
        assert not result['failure'], f"No relevant document in top-{CASE['k']}: {query}"
"""
    return {"case_id":digest,"case":payload,"pytest":test,
            "instructions":"Save case.json and test_regression.py together. Run pytest test_regression.py. This desired-behavior test intentionally fails on the discovered bug; demo models run without downloads."}
