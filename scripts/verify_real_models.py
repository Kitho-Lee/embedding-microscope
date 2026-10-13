import json,time
from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.engine import REAL_MODELS,encode,load_model
p=Path(__file__).resolve().parents[1];models=sorted(REAL_MODELS)
cases=[]
def add(name,q,v,kind,docs,b,a,factor):
    cases.append(dict(name=name,query=q,variant=v,kind=kind,factor=factor,documents=[dict(id=str(i),text=t) for i,t in enumerate(docs)],relevant_before=b,relevant_after=a,models=models,k=1))
notes=['本地笔记软件，断网可用，所有数据保存在设备上，不上传数据。','云端笔记软件，需要联网才能使用，所有数据自动上传。','笔记软件支持离线编辑，联网后自动上传数据。']
add('negation','寻找不上传数据的笔记软件','寻找会上传数据的笔记软件','condition_change',notes,['0'],['1','2'],'否定')
add('paraphrase','寻找不上传数据的笔记软件','找一款笔记应用，资料只保留在设备本地','meaning_preserving',notes,['0'],['0'],'同义改写')
add('number','寻找价格低于500元的显示器','寻找价格高于500元的显示器','condition_change',['显示器价格为300元。','显示器价格为800元。','键盘价格为200元。'],['0'],['1'],'数字阈值')
add('entity','查询北京的天气','查询上海的天气','condition_change',['北京今天晴，最高气温25度。','上海今天雨，最高气温23度。','广州今天阴，最高气温28度。'],['0'],['1'],'实体')
add('direction','谁向李明借钱？','李明向谁借钱？','condition_change',['张伟向李明借了100元。','李明向王芳借了100元。','王芳向张伟借了100元。'],['0'],['1'],'关系方向')
add('english_control','A cat sleeping on a sofa','A feline resting on a couch','meaning_preserving',['A cat is sleeping on the sofa.','A car drives on the highway.','A fish swims underwater.'],['0'],['0'],'同义改写')
client=TestClient(app);results=[]
for model in models:
    start=time.time();print('Loading '+model,flush=True)
    vectors=encode(['寻找本地笔记应用','查询北京天气'],model,'query')
    assert vectors.shape==(2,384),vectors.shape
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    print('Loaded',model,'seconds',round(time.time()-start,2),'dimension',vectors.shape[1],flush=True)
for case in cases:
    name=case.pop('name');start=time.time()
    response=client.post('/api/experiment',json=case)
    assert response.status_code==200,response.text
    result=response.json()
    # independent metric computation from returned ranks and provided labels
    for r in result['results']:
        for side,labels in [('before',case['relevant_before']),('after',case['relevant_after'])]:
            e=r[side];ranks=[d['rank'] for d in e['ranking'] if d['id'] in labels]
            recall=sum(rank<=case['k'] for rank in ranks)/len(labels)
            dcg=sum(1/np.log2(rank+1) for rank in ranks if rank<=case['k'])
            ideal=sum(1/np.log2(i+2) for i in range(min(case['k'],len(labels))))
            assert np.isclose(e['recall'],recall)
            assert np.isclose(e['mrr'],1/min(ranks))
            assert np.isclose(e['ndcg'],dcg/ideal)
            assert e['failure']==(recall==0)
    repeated=client.post('/api/experiment',json=result['config']).json()
    for a,b in zip(result['results'],repeated['results']):
        for side in ['before','after']:
            assert [d['id'] for d in a[side]['ranking']]==[d['id'] for d in b[side]['ranking']]
            assert np.allclose([d['score'] for d in a[side]['ranking']],[d['score'] for d in b[side]['ranking']],atol=1e-6)
    results.append(dict(name=name,experiment=result))
    print(name,[(r['model'].split('/')[-1],r['before']['recall'],r['after']['recall']) for r in result['results']],flush=True)
folder=p/'verification';folder.mkdir(exist_ok=True)
(folder/'real-model-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
(folder/'real-model-cases.json').write_text(json.dumps([r['experiment']['config'] for r in results],ensure_ascii=False,indent=2))
print('All model integration, metric and replay checks passed',flush=True)
