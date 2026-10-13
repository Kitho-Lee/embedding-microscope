"""Exact retrieval and paired experiments. Scores are never compared across models."""
import importlib.metadata
import platform
import json
from pathlib import Path
import hashlib
import re
from functools import lru_cache
import numpy as np

DEMO_MODELS = {"demo-word", "demo-char"}
REAL_MODELS = {"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "intfloat/multilingual-e5-small"}

MODEL_REVISIONS = json.loads((Path(__file__).parent / "model-revisions.json").read_text())

@lru_cache(maxsize=2)
def load_model(name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, revision=MODEL_REVISIONS[name], trust_remote_code=False, device="cpu")

def encode(texts, model, role):
    if model in DEMO_MODELS:
        vectors = np.zeros((len(texts), 512))
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())
            if model == "demo-char":
                tokens += [text[j:j+2] for j in range(len(text)-1)]
            for token in tokens:
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 512
                vectors[i,index] += 1
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    if model not in REAL_MODELS:
        raise ValueError("Unsupported model")
    if model == "intfloat/multilingual-e5-small":
        texts = [("query: " if role == "query" else "passage: ") + t for t in texts]
    return load_model(model).encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def evaluate(query, documents, relevant, model, k):
    q = encode([query], model, "query")[0]
    scores = encode([d["text"] for d in documents], model, "document") @ q
    return rank_scores(scores, documents, relevant, k)

def rank_scores(scores, documents, relevant, k):
    order = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), documents[i]["id"]))
    ranked = [{**documents[i], "score": float(scores[i]), "rank": rank+1,
               "relevant": documents[i]["id"] in relevant} for rank,i in enumerate(order)]
    hits = [r["rank"] for r in ranked if r["relevant"]]
    top = [r for r in ranked[:k] if r["relevant"]]
    dcg = sum(1/np.log2(r["rank"]+1) for r in top)
    ideal = sum(1/np.log2(i+2) for i in range(min(k,len(relevant))))
    return {"ranking": ranked, "recall": len(top)/len(relevant),
            "mrr": 1/min(hits) if hits else 0, "ndcg": float(dcg/ideal) if ideal else 0,
            "failure": not bool(top)}

def run_experiment(data):
    results=[]
    for model in data["models"]:
        before = evaluate(data["query"], data["documents"], data["relevant_before"], model,data["k"])
        after = evaluate(data["variant"], data["documents"], data["relevant_after"], model,data["k"])
        a={r["id"] for r in before["ranking"][:data["k"]]}
        b={r["id"] for r in after["ranking"][:data["k"]]}
        results.append({"model":model,"demo":model in DEMO_MODELS,"before":before,"after":after,
                        "top_k_overlap":len(a&b)/len(a),
                        "regression":after["recall"] < before["recall"],
                        "tokenization":tokenization_report(data,model)})
    return {"config":data,"results":results,"schema_version":1,
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,
                       "sentence_transformers":package_version("sentence-transformers"),
                       "encoding":"normalized cosine; E5 query/passage prefixes; CPU float32",
                       "model_revisions":{m:MODEL_REVISIONS[m] for m in data["models"] if m in REAL_MODELS},
                       "torch":package_version("torch"),"transformers":package_version("transformers")},
            "failure_definition":"No labeled relevant document in top-k; regression means recall@k decreased.",
            "label_source":"User-supplied labels; not independently verified."}

def package_version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None

def tokenization_report(data, model):
    if model in DEMO_MODELS:
        return {"max_tokens":None,"truncated":[]}
    encoder=load_model(model)
    entries=[("query_before",data["query"],"query"),("query_after",data["variant"],"query")]
    entries += [("document:"+d["id"],d["text"],"document") for d in data["documents"]]
    truncated=[]
    for name,text,role in entries:
        if model=="intfloat/multilingual-e5-small":
            text=("query: " if role=="query" else "passage: ")+text
        count=len(encoder.tokenizer(text,truncation=False)["input_ids"])
        if count>encoder.max_seq_length:
            truncated.append({"input":name,"tokens":count})
    return {"max_tokens":encoder.max_seq_length,"truncated":truncated}
