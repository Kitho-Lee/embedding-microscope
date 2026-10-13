import json,io,zipfile,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.environ['EM_LAB_DB']=str(root/'work/verify-lab.sqlite')
from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app);p=root
data=json.loads((p/'examples/dataset.json').read_text())
assert client.post('/api/lab/validate',json=data).status_code==200
response=client.post('/api/lab/search',json=data);assert response.status_code==200,response.text
run=response.json();print('evaluated',run['evaluated'],'candidates',len(run['candidates']),flush=True)
c=next(c for c in run['candidates'] if c['query_id']=='borrow-direction' and c['edit_count']==2)
assert c['losers']==['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']
base=f"/api/lab/runs/{run['id']}/candidates/{c['id']}"
assert client.get(base+'/export').status_code==422
r=client.post(base+'/review',json={'status':'accepted','note':'两个问句均询问李明的出借人；文档1说明出借人为王芳。'});assert r.status_code==200
r=client.post(base+'/reduce');assert r.status_code==200,r.text
assert r.json()['reduction']['before']==2 and r.json()['reduction']['after']==1
reduced=r.json()['candidate']
base=f"/api/lab/runs/{run['id']}/candidates/{reduced['id']}"
assert client.get(base+'/export').status_code==422
assert client.post(base+'/review',json={'status':'accepted','note':'删除问号替换后，同义改写保持相关文档1，分歧仍复现。'}).status_code==200
export=client.get(base+'/export');assert export.status_code==200
with zipfile.ZipFile(io.BytesIO(export.content)) as z:
    folder=root/'work/lab-regression';folder.mkdir(exist_ok=True);z.extractall(folder)
    case=json.loads(z.read('case.json'));assert case['review']['status']=='accepted'
    assert case['models']==c['losers']
run=client.get('/api/lab/runs/'+run['id']).json()
(p/'verification/lab-real-results.json').write_text(json.dumps(run,ensure_ascii=False,indent=2))
(p/'verification/lab-regression.zip').write_bytes(export.content)
print('Real import/search/review/reduction/export verified',flush=True)
