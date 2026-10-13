import io,json,zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.hunt import search,regression_bundle
client=TestClient(app)

def config():
    return dict(query='alpha beta',documents=[dict(id='A',text='alpha beta'),dict(id='B',text='gamma delta')],relevant=['A'],models=['demo-word','demo-char'],k=1,rules=[dict(source='alpha',replacement='gamma'),dict(source='beta',replacement='delta')],approved=True,budget=63)

def test_search_exhaustive_minimal_and_export():
    c=config();result=search(c)
    assert result['exhausted'] and result['evaluated']==3
    assert result['best']
    assert all(f['edit_count']>=result['best']['edit_count'] for f in result['failures'])
    assert result['best']['query']!=c['query']
    # synthetic lexical replacements test search plumbing, NOT semantic validity
    response=client.post('/api/hunt/export',json=c)
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        case=json.loads(z.read('case.json'))
        assert case['query']==result['best']['query']
        compile(z.read('test_regression.py'),'test_regression.py','exec')

def test_overlap_and_missing_source_rejected():
    for rules in [[dict(source='alpha',replacement='x'),dict(source='alpha beta',replacement='y')],[dict(source='missing',replacement='x')]]:
        assert client.post('/api/hunt',json=config()|dict(rules=rules)).status_code==422
    assert client.post('/api/hunt',json=config()|dict(approved=False)).status_code==422

def test_budget_and_baseline_failure():
    result=search(config()|dict(budget=1))
    assert result['evaluated']==1 and not result['exhausted']
    result=search(config()|dict(relevant=['B']))
    assert not result['eligible_models'] and result['best'] is None
    with pytest.raises(ValueError): regression_bundle(result)

def test_pair_interaction_requires_two_edits(monkeypatch):
    import app.hunt as h
    def fake_encode(texts,model,role):
        if role=='document': return np.eye(2)
        return np.array([[0,1] if 'gamma' in t and 'delta' in t else [1,0] for t in texts])
    monkeypatch.setattr(h,'encode',fake_encode)
    result=search(config())
    assert result['best']['edit_count']==2
    assert result['failure_count']==1
