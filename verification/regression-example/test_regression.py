# Generated desired-behavior regression: expected to FAIL until retrieval is repaired.
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
