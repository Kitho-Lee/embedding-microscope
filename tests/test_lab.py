import io,json,zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import lab
client=TestClient(app)
@pytest.fixture(autouse=True)
def local_db(tmp_path,monkeypatch):
    monkeypatch.setenv('EM_LAB_DB',str(tmp_path/'lab.sqlite'))

def dataset():
    return dict(documents=[dict(id='A',text='correct'),dict(id='B',text='wrong')],
                queries=[dict(id='q',text='red blue green',relevant=['A'],rules=[dict(source='red',replacement='rouge'),dict(source='blue',replacement='bleu'),dict(source='green',replacement='vert')])],
                models=['demo-word','demo-char'],k=1,budget_per_query=63)

def fake(texts,model,role):
    if role=='document':return np.eye(2)
    return np.array([[0.,1.] if model=='demo-word' and 'rouge' in t else [1.,0.] for t in texts])

def test_import_validation():
    assert client.post('/api/lab/validate',json=dataset()).status_code==200
    d=dataset();d['queries'][0]['relevant']=['missing']
    assert client.post('/api/lab/validate',json=d).status_code==422
    d=dataset();d['queries']*=2
    assert client.post('/api/lab/validate',json=d).status_code==422
    d=dataset();d['queries'][0]['rules'].append(dict(source='red blue',replacement='x'))
    assert client.post('/api/lab/validate',json=d).status_code==422

def test_review_reduction_export_and_persistence(monkeypatch):
    monkeypatch.setattr(lab,'encode',fake)
    response=client.post('/api/lab/search',json=dataset());assert response.status_code==200,response.text
    run=response.json();assert run['evaluated']==8 and run['exhausted']
    assert len(run['candidates'])==4
    original=next(c for c in run['candidates'] if c['edit_count']==3)
    base=f"/api/lab/runs/{run['id']}/candidates/{original['id']}"
    assert client.get(base+'/export').status_code==422
    assert client.post(base+'/reduce').status_code==422
    assert client.post(base+'/review',json=dict(status='accepted',note='Reviewed synthetic fixture')).status_code==200
    reduced=client.post(base+'/reduce').json()
    assert reduced['reduction']['before']==3 and reduced['reduction']['after']==1
    best=reduced['candidate'];assert best['text']=='rouge blue green'
    assert best['review']['status']=='pending'
    target=f"/api/lab/runs/{run['id']}/candidates/{best['id']}"
    assert client.get(target+'/export').status_code==422
    client.post(target+'/review',json=dict(status='rejected',note='Not equivalent'))
    assert client.get(target+'/export').status_code==422
    client.post(target+'/review',json=dict(status='accepted',note='Test mechanics only; not a semantic claim'))
    response=client.get(target+'/export');assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        case=json.loads(archive.read('case.json'))
        assert case['models']==['demo-word']
        assert case['review']['status']=='accepted'
        assert case['reduction']['after']==1
        compile(archive.read('test_regression.py'),'test_regression.py','exec')
    loaded=client.get('/api/lab/runs/'+run['id']).json()
    assert len(loaded['review_log'])==3
    assert loaded['reductions']

def test_budget_and_identical_output_dedup(monkeypatch):
    monkeypatch.setattr(lab,'encode',fake)
    d=dataset();d['budget_per_query']=1
    result=client.post('/api/lab/search',json=d).json()
    assert result['evaluated']==2 and not result['exhausted']
    assert len(result['candidates'])==1

def test_missing_run_and_empty_note():
    assert client.get('/api/lab/runs/missing').status_code==422
    assert client.post('/api/lab/runs/missing/candidates/x/review',json=dict(status='accepted',note='')).status_code==422

def test_primary_and_single_query_pages():
    assert '检索差分实验室' in client.get('/').text
    assert '控制实验' in client.get('/experiment').text

def test_provenance_is_config_bound(monkeypatch):
    monkeypatch.setattr(lab,'encode',fake)
    first=client.post('/api/lab/search',json=dataset()).json()
    second=client.post('/api/lab/search',json=dataset()).json()
    assert first['provenance']['dataset_sha256']==second['provenance']['dataset_sha256']
    modified=dataset();modified['k']=2
    third=client.post('/api/lab/search',json=modified).json()
    assert first['provenance']['dataset_sha256']!=third['provenance']['dataset_sha256']

def test_bilingual_pages_and_no_screenshots():
    from pathlib import Path
    static=Path(__file__).parents[1]/'app/static'
    for name in ['lab.html','index.html','lab.en.html','index.en.html']:
        page=(static/name).read_text()
        assert 'id="languageSwitch"' in page
        assert '<img' not in page
        assert '/static/language.js' in page
    assert 'lang="en"' in (static/'lab.en.html').read_text()
    assert 'Review candidates' in (static/'lab.en.html').read_text()
    assert '李明向谁借钱？' in (static/'lab.en.html').read_text()

def test_traditional_chinese_preserves_case_data():
    from pathlib import Path
    static=Path(__file__).parents[1]/'app/static'
    html=(static/'lab.zh-Hant.html').read_text()
    assert 'lang="zh-Hant"' in html and '導入語料與標籤' in html
    assert '李明向谁借钱？' in html
    assert 'value="zh-Hant"' in html
    assert '<img' not in html
