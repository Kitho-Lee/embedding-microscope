# Embedding Microscope · Embedding 显微镜

Controlled retrieval failure discovery: import labeled corpora, compare two models, review semantic edits, reduce failing cases, and export executable regression tests.


## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 for the corpus differential lab, or `/experiment` for the single-query experiment. The two demo encoders run without model downloads. They are deterministic hashed bag-of-features baselines, **not semantic models and not evidence of real model quality**.

For real embeddings:

```bash
pip install -e '.[models]'
```

Select multilingual MiniLM and multilingual E5 in the UI. First use downloads weights from Hugging Face; subsequent runs use the local model cache. E5 uses `query:` / `passage:` prefixes. Both use normalized vectors and exact cosine retrieval. Model revisions are pinned in `app/model-revisions.json`; inference uses CPU float32 for the verified baseline. Model memory and download size depend on the selected models. The server runs synchronously and is intended for one local user.

## Experiment workflow

1. Enter original and perturbed queries and a corpus of `{id,text}` documents.
2. Choose meaning-preserving or condition-changing perturbation.
3. Supply relevant document IDs for **both** queries. Meaning-preserving cases require the same labels.
4. Run two models and inspect full rankings, Recall@k, MRR, nDCG@k and top-k overlap.
5. Filter history for retrieval failures or recall regressions; load any case back into the editor.
6. Export the selected experiment and reproduce it:

```bash
python -m app.cli embedding-experiment.json --output result.json
pytest
```

The single-factor replacement tool changes the first occurrence of a selected string and records the experiment factor (negation, numeric threshold, entity, relation direction, paraphrase or custom).

The negation button creates an unverified candidate, not a validated semantic transformation. Review meaning and labels. The rewrite button loads a curated example.

## Interpretation

- Failure means **zero relevant documents in top-k**. Regression means recall@k fell after perturbation; it is not necessarily a model defect.
- Labels are user supplied, binary, and not independently validated. Condition changes can legitimately change relevant documents.
- MRR is computed over the complete corpus ranking; Recall and nDCG are at k.
- Raw cosine scores are not calibrated across models. Compare task outcomes, not raw score magnitude.
- MiniLM uses a 128-token maximum sequence length and E5 uses its model-defined limit. Longer inputs are truncated by the encoder. Pair exports report truncation for queries and documents; hunt reports baseline/best queries and corpus, not every intermediate candidate. Truncation warnings must be considered before attributing failures to semantic reasoning.
- A document's contradictory meaning does not by itself imply low topical similarity.
- Ties are broken by document ID, deterministically. Zero demo vectors yield zero scores.
- History is capped at 50 runs and stored in browser localStorage; export important cases.
- Model revisions and key runtime versions are recorded in exports. Rankings can still vary across library versions and hardware; the verified environment is described below. Very close scores should be treated as fragile.

## Minimal failing cases → executable regression tests

Use your own corpus in the document editor (up to 200 documents). Provide relevant IDs and up to six disjoint replacement rules. Each source must occur exactly once in the original query. **Review that every rule and every tested combination preserve the relevance labels** before checking approval. The tool does not verify semantics itself.

Search enumerates rule subsets by edit count (up to 63 combinations), encodes candidates in batches, and reports the smallest newly failing case. Only models that retrieve a relevant document at baseline are eligible. Minimality is restricted to the supplied finite rule set, not arbitrary edits or character distance. Zero relevant documents in top-k defines failure; a partial recall decrease does not count in this search.

Export creates a ZIP containing `case.json`, `test_regression.py` and instructions. Save them together and run:

```bash
pytest path/to/test_regression.py
```

The generated test expresses **desired behavior**, so it should FAIL on the known failure until a repair restores relevance. It also checks the baseline and pinned model revision. This is distinct from the project's passing implementation tests. Generated examples are outside the default test discovery path.

The saved human-readable case is the regression artifact; no LLM judgment or semantic similarity threshold is used as an oracle. Rules are attested by the user, not automatically verified.

## Related work and positioning

The individual ideas are established; this project makes no first-of-its-kind claim.

| Prior work | Established capability | Our focus |
| --- | --- | --- |
| [WizMap](https://github.com/poloclub/wizmap) | Interactive exploration of large embedding spaces | Small, labeled retrieval failures rather than projection maps |
| [Embedding Comparator](https://vis.mit.edu/pubs/embedding-comparator.pdf) | Comparing global structure and local neighborhoods | Query edits and executable desired-behavior tests |
| [EmbedAudit](https://github.com/dakshjain-1616/Embedding-Evaluator) | Embedding space auditing and reporting | Relevance-labeled, corpus-specific minimal failure cases |
| [Aquin CLI](https://www.aquin.app/docs/inspection/embedding) | Token perturbation diagnostics and cosine-ranked retrieval | Relevance-labeled finite failure search and regression bundles |
| [CheckList](https://github.com/marcotcr/checklist) | Behavioral NLP testing with controlled changes | Local retriever integration and a failure-to-regression workflow |
| [TextAttack](https://github.com/QData/TextAttack) | Constrained transformations and adversarial search | Transparent, finite approved edits and relevance labels; not a new search algorithm |
| [ALEE](https://github.com/Andrian0s/any-lang-embed-eval) | AMR-controlled semantic foils for multilingual embedding evaluation | User corpus debugging rather than cross-lingual benchmark ranking |

Research checked 2026-10-13. These overlaps are based on official repositories/papers, not a full code audit or proof that no other equivalent tool exists. Potential differentiation is the focused workflow **real corpus → approved edits → minimal failure → pytest**, not model comparison or perturbation testing alone.

## Verification (2026-10-13)

Both pinned real models were downloaded and executed on CPU with Python 3.12. The checked package versions are in `requirements-verified.txt` (macOS ARM64 version snapshot, not a portable lockfile). Six hand-authored small-corpus cases test negation, paraphrase, numeric thresholds, entities, relation direction and an English control. These are integration probes, **not a representative benchmark or user production corpus**.

| Case | E5 Recall@1 before → after | MiniLM Recall@1 before → after |
| --- | --- | --- |
| Negation | 1 → 0.5 | 1 → 0.5 |
| Paraphrase | 1 → 1 | 1 → 1 |
| Number | 1 → 0 | 0 → 1 |
| Entity | 1 → 1 | 1 → 1 |
| Relation direction | 0 → 1 | 0 → 1 |
| English control | 1 → 1 | 1 → 1 |

The negation case has TWO relevant documents after the edit, so perfect top-1 recall is impossible; 0.5 is not itself a defect. This illustrates why changed-label recall deltas require interpretation. Numeric and relation failures are based on explicit document labels, not cosine thresholds.

Metrics were independently recomputed from returned ranks; embedding dimensions (384), finiteness and unit norm were checked. Re-running all six configs produced matching rankings and scores within 1e-6.

A further same-meaning probe found a genuine labeled failure:

- Baseline: `李明向谁借钱？` → both models rank `李明向王芳借了100元。` first.
- Approved paraphrase: `谁把钱借给了李明？` → E5 still ranks the correct document first; MiniLM instead ranks `张伟向李明借了100元。` first.
- The latter describes Zhang borrowing FROM Li, so it does not answer who lent TO Li.

The finite search uses a single whole-query rewrite rule in this example. It demonstrates the workflow; it does not establish a character-minimal counterexample. Additional multi-rule interaction tests validate cardinality-ordered search.

Reproduce:

```bash
python -m app.cli examples/experiment.json --output result.json
python -m app.hunt_cli examples/hunt.json --output hunt-result.json --regression-dir work/regression
pytest work/regression/test_regression.py  # expected to fail on the pinned MiniLM model
pytest  # implementation tests should pass
```

Run `python scripts/verify_real_models.py` to repeat the six paired model/API/metric/replay checks (requires model dependencies and downloads if not cached).

Machine-readable results are in `verification/`. `verification/regression-example/test_regression.py` is intentionally failing and excluded from default discovery. A model-quality failure does not mean the evaluation software is broken. No model fine-tuning or retrieval repair is included yet.

## Architecture

`app/engine.py`: normalized encoding, exact retrieval, paired experiment metrics.

`app/main.py`: validated FastAPI API, model allowlist, static frontend.

`app/static/`: dependency-free responsive UI, browser history, JSON export.

`app/cli.py`: repeat exported experiments without the UI.

`tests/`: metrics, validation and deterministic demo regression tests.

No vector database is required for the 200-document limit. No external LLM generates judgments. The implementation intentionally exposes labels rather than inventing correctness scores.

## API

`GET /api/models` lists supported encoders. `POST /api/experiment` takes the config shown in `examples/experiment.json`. Interactive schema: http://127.0.0.1:8000/docs.

## Roadmap

- Verified Chinese minimal-pair dataset and grouped bootstrap intervals.
- Corpus-level batch failure discovery and held-out repair validation.
- Embedding cache and portable dependency lockfiles.
- BM25 / reranker repair comparison on held-out cases.

## References

- [MTEB](https://arxiv.org/abs/2210.07316)
- [BEIR](https://arxiv.org/abs/2104.08663)
- [Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/model.html)

MIT licensed. Model weights retain their own licenses.

## v0.2 — Corpus Differential Lab

The home page opens the corpus lab; `/static/lab.html` is an equivalent direct link. The lab adds file import, query-level differential discovery, recorded review, exact reduction over selected edit subsets, and reviewed regression export. The single-query v0.1 experiment remains available.

### Import format

Import a UTF-8 JSON file or paste JSON; see `examples/dataset.json`:

```json
{
  "documents": [{"id":"d1","text":"Relevant passage"},{"id":"d2","text":"Distractor"}],
  "queries": [{"id":"q1","text":"Original query","relevant":["d1"],"rules":[{"source":"Original","replacement":"Initial"}]}],
  "models": ["demo-word","demo-char"],
  "k": 1,
  "budget_per_query": 63
}
```

Relevance is binary. Document/query IDs must be unique, every relevant ID must exist, sources must occur exactly once and must not overlap. Rules propose relevance-preserving rewrites, but **search does not certify semantic equivalence**. Empty rules allow baseline model comparison. File import populates model, k and budget controls; these controls can then override the JSON fields. Current limits: 200 documents, 20 queries, 6 rules/query, 63 variants/query. This is exact local retrieval, not a large-scale vector serving system.

### Differential search and visuals

Document vectors are computed once per model for the run; candidate queries are encoded in batches. Search includes the original query and cardinality-ordered edit combinations. Identical generated texts are deduplicated within a query. It surfaces single-model failures and perturbations introducing failures in both models. Candidate priority uses a single-model-failure indicator and the gap in best relevant document rank; raw cosine values are not calibrated across models.

The lab shows:

- Workflow steps and run/review counters.
- A query × model baseline relevance-rank matrix.
- Each candidate's before/after relevant-rank graph, with a Top-k boundary.
- Removed/inserted text for each active edit.
- A reduction trace showing which subsets preserve the same winner/loser outcome.

The displayed matrix is baseline-only; graphs show each selected perturbation. Green/red encode Top-k success/failure, not general semantic quality.

### Review and reduction

Every candidate starts pending. Accept, reject or mark uncertain with a written reason after inspecting text, labels and full rankings. Only accepted cases can be exported or reduced. Reduction exhaustively tests all subsets of the candidate's edits while preserving **the same successful model and the same failed model**. The smallest result is exact only in that finite subset space. It may be zero edits if the original query already exhibits that differential. A new reduced text remains pending until reviewed; existing reviews are retained for identical query-ID/text pairs.

Review history and reduction traces persist in local SQLite (`work/lab.sqlite` by default; override `EM_LAB_DB`). Browser reload restores the last run ID. Export the complete run JSON to archive inputs, results and audit records. The local database and imported data are excluded from Git. This is a single-user local workflow, without authentication or multi-user review isolation.

Regression ZIPs contain the reviewed case and executable desired-behavior pytest. They intentionally fail on known retrieval failures. If the baseline was already failing, the exported test will fail on the baseline too; the case records that fact and should not be described as a perturbation regression. Rejected, uncertain and pending cases cannot be exported by this lab.

### Validation

Implementation tests include invalid imports, unknown labels, overlapping edits, budget truncation, review/export gates, persisted audit trails, and a synthetic three-edit case reduced to one edit. Synthetic tests check mechanics rather than semantic validity.

`verification/lab-real-results.json` records a run using both pinned real models and the hand-authored borrowing-direction corpus, with accepted review and exact subset reduction from two edits to one. The known MiniLM failure is reproduced and exported in `verification/lab-regression.zip`. These small examples validate integration; they are not a representative production benchmark.

Repeat v0.2 integration checks with `python scripts/verify_lab.py` (real model dependencies required). Candidate exports record input truncation warnings; those warnings can explain apparent failures and must be considered during review.
