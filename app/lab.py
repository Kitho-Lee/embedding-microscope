"""Corpus-level differential retrieval, auditable review and finite edit reduction."""
import hashlib
import json
import sqlite3
import os
from pathlib import Path
from itertools import combinations
from uuid import uuid4
from datetime import datetime, timezone
from .engine import encode, rank_scores, MODEL_REVISIONS, REAL_MODELS, load_model, package_version
from .hunt import regression_bundle

def database():
    path=Path(os.environ.get("EM_LAB_DB", "work/lab.sqlite"))
    path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
    return conn

def save(run):
    with database() as db:
        db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)",(run["id"],json.dumps(run,ensure_ascii=False)))

def get(run_id):
    with database() as db: row=db.execute("SELECT data FROM runs WHERE id=?",(run_id,)).fetchone()
    if not row: raise ValueError("Run not found; import and search again")
    return json.loads(row[0])

def edits(query,rules):
    spans=[]
    for rule in rules:
        src=rule["source"]
        if query.count(src)!=1: raise ValueError("Each rule source must occur exactly once")
        start=query.index(src);end=start+len(src)
        if any(start<b and end>a for a,b,_ in spans): raise ValueError("Edit sources overlap")
        spans.append((start,end,rule["replacement"]))
    return spans

def apply(query,rules,indices):
    spans=edits(query,rules)
    for i in sorted(indices,key=lambda i:spans[i][0],reverse=True):
        a,b,replacement=spans[i];query=query[:a]+replacement+query[b:]
    return query

def identity(qid,text):
    return hashlib.sha256((qid+"\0"+text).encode()).hexdigest()[:20]

def inspect(text,results,baseline,qid,indices,origin="search"):
    models=list(results)
    winners=[m for m,e in results.items() if not e["failure"]]
    losers=[m for m,e in results.items() if e["failure"]]
    ranks={m:min(d["rank"] for d in e["ranking"] if d["relevant"]) for m,e in results.items()}
    differential=len(winners)==1
    return {"id":identity(qid,text),"query_id":qid,"text":text,"rule_indices":list(indices),
            "edit_count":len(indices),"results":results,"winners":winners,"losers":losers,
            "differential":differential,"introduced_failures":[m for m in losers if not baseline[m]["failure"]],
            "priority":(1000 if differential else 0)+abs(ranks[models[0]]-ranks[models[1]]),
            "review":{"status":"pending","note":"","reviewed_at":None},"origin":origin}

def truncation_warnings(model,texts,role):
    if model not in REAL_MODELS:return [False]*len(texts)
    encoder=load_model(model)
    if model=="intfloat/multilingual-e5-small":
        texts=[("query: " if role=="query" else "passage: ")+t for t in texts]
    tokens=encoder.tokenizer(texts,truncation=False,verbose=False)["input_ids"]
    return [len(ids)>encoder.max_seq_length for ids in tokens]

def search(data):
    vectors={m:encode([d["text"] for d in data["documents"]],m,"document") for m in data["models"]}
    corpus_warnings={m:[d["id"] for d,w in zip(data["documents"],truncation_warnings(m,[d["text"] for d in data["documents"]],"document")) if w] for m in data["models"]}
    candidates=[];baselines={};evaluated=0;possible=0
    for query in data["queries"]:
        edits(query["text"],query["rules"])
        options=[((),query["text"])];seen={query["text"]}
        for size in range(1,len(query["rules"])+1):
            for indices in combinations(range(len(query["rules"])),size):
                text=apply(query["text"],query["rules"],indices)
                if text not in seen: options.append((indices,text));seen.add(text)
        possible+=len(options)
        options=options[:data["budget_per_query"]+1];evaluated+=len(options)
        outputs={};query_warnings={}
        for m in data["models"]:
            query_warnings[m]=truncation_warnings(m,[text for _,text in options],"query")
            scores=encode([text for _,text in options],m,"query")@vectors[m].T
            outputs[m]=[rank_scores(s,data["documents"],query["relevant"],data["k"]) for s in scores]
        baseline={m:out[0] for m,out in outputs.items()};baselines[query["id"]]=baseline
        for idx,(indices,text) in enumerate(options):
            candidate=inspect(text,{m:out[idx] for m,out in outputs.items()},baseline,query["id"],indices)
            candidate["truncation"]={m:{"query":query_warnings[m][idx],"documents":corpus_warnings[m]} for m in data["models"]}
            if candidate["differential"] or candidate["introduced_failures"]:
                candidates.append(candidate)
    provenance={"dataset_sha256":hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest(),
                "packages":{name:package_version(name) for name in ["numpy","torch","sentence-transformers","transformers"]},
                "retrieval":"exact normalized cosine", "device":"cpu", "search":"cardinality-ordered finite edit subsets"}
    run={"id":uuid4().hex,"schema_version":2,"created_at":datetime.now(timezone.utc).isoformat(),
         "config":data,"baselines":baselines,"candidates":sorted(candidates,key=lambda c:(-c["priority"],c["edit_count"],c["id"])),
         "evaluated":evaluated,"search_space":possible,"exhausted":evaluated==possible,
         "model_revisions":{m:MODEL_REVISIONS[m] for m in data["models"] if m in REAL_MODELS},
         "provenance":provenance,"review_log":[],"reductions":[],"limitations":"Candidate semantics and labels need review. Priority uses rank differences, not cross-model cosine calibration."}
    save(run);return run

def candidate(run,cid):
    for c in run["candidates"]:
        if c["id"]==cid:return c
    raise ValueError("Candidate not found")

def review(run_id,cid,status,note):
    run=get(run_id);c=candidate(run,cid)
    record={"candidate_id":cid,"status":status,"note":note,"reviewed_at":datetime.now(timezone.utc).isoformat()}
    c["review"]={k:record[k] for k in ["status","note","reviewed_at"]}
    run["review_log"].append(record);save(run);return run

def reduce(run_id,cid):
    run=get(run_id);original=candidate(run,cid)
    if original["review"]["status"]!="accepted":raise ValueError("Accept the candidate before reducing edits")
    if not original["differential"]:raise ValueError("Reduction requires one successful and one failed model")
    cfg=run["config"];q=next(q for q in cfg["queries"] if q["id"]==original["query_id"])
    options=[]
    for size in range(len(original["rule_indices"])+1):
        options.extend((indices,apply(q["text"],q["rules"],indices)) for indices in combinations(original["rule_indices"],size))
    outputs={}
    for m in cfg["models"]:
        docs=encode([d["text"] for d in cfg["documents"]],m,"document")
        scores=encode([text for _,text in options],m,"query")@docs.T
        outputs[m]=[rank_scores(s,cfg["documents"],q["relevant"],cfg["k"]) for s in scores]
    trace=[];best=None
    for i,(indices,text) in enumerate(options):
        result={m:out[i] for m,out in outputs.items()}
        preserves=all(not result[m]["failure"] for m in original["winners"]) and all(result[m]["failure"] for m in original["losers"])
        trace.append({"indices":list(indices),"text":text,"preserves_differential":preserves})
        if preserves and best is None:
            best=inspect(text,result,run["baselines"][q["id"]],q["id"],indices,"reduction")
    if best is None:raise ValueError("Failure no longer reproduces")
    # Existing approved text retains review. New smaller text must be reviewed anew.
    existing=next((c for c in run["candidates"] if c["id"]==best["id"]),None)
    if existing:best=existing
    else:
        best["truncation"]={m:{"query":truncation_warnings(m,[best["text"]],"query")[0],"documents":original.get("truncation",{}).get(m,{}).get("documents",[])} for m in cfg["models"]}
        run["candidates"].append(best)
    record={"source_id":cid,"result_id":best["id"],"before":original["edit_count"],"after":best["edit_count"],
            "trace":trace,"minimality":"Exact minimum rule cardinality among subsets of the selected candidate's edits; no claim about all rewrites or character minimality.",
            "semantic_status":best["review"]["status"]}
    run["reductions"].append(record);save(run)
    return {"run":run,"reduction":record,"candidate":best}

def export(run_id,cid):
    run=get(run_id);c=candidate(run,cid)
    if c["review"]["status"]!="accepted":raise ValueError("Only accepted candidates may be exported")
    if not c["losers"]:raise ValueError("No failing model")
    cfg=run["config"];q=next(q for q in cfg["queries"] if q["id"]==c["query_id"])
    result={"config":{"query":q["text"],"documents":cfg["documents"],"relevant":q["relevant"],"models":cfg["models"],"k":cfg["k"],"rules":q["rules"]},
            "best":{"query":c["text"],"failing_models":c["losers"],"rule_indices":c["rule_indices"]},
            "model_revisions":run["model_revisions"],"semantic_validity":"Accepted by user review; not independently verified"}
    bundle=regression_bundle(result)
    bundle["case"]["provenance"]=run.get("provenance",{})
    bundle["case"]["truncation"]=c.get("truncation",{})
    bundle["case"]["review"]=c["review"]
    bundle["case"]["query_id"]=q["id"]
    bundle["case"]["winner_models"]=c["winners"]
    bundle["case"]["baseline_failures"]=[m for m,e in run["baselines"][q["id"]].items() if e["failure"]]
    bundle["case"]["reduction"]=next((r for r in reversed(run["reductions"]) if r["result_id"]==cid),None)
    return bundle
