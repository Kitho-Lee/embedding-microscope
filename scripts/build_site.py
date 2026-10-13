"""Package the real browser-inference edition for GitHub Pages."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1];src=root/'app/static';dest=root/'docs/site';dest.mkdir(exist_ok=True)
for suffix in ['', '.en', '.zh-Hant']:
    html=(src/('lab'+suffix+'.html')).read_text()
    html=html.replace('/static/style.css','./style.css').replace('/static/language.js','./language.js').replace('href="/"','href="./index.html"')
    html=html.replace('href="/experiment"','href="https://github.com/Kitho-Lee/embedding-microscope#run-locally"')
    note={'':'在线完整版：输入语料在浏览器中运行，首次下载 ONNX q8 模型。无需 Python 或 API Key。结果可能与 Python fp32 版不同；模型最多读取 128 / 512 tokens，长文本可能截断。',' .en':'',' .zh-Hant':''}
    if suffix=='.en':text='Full online edition: run your own corpus in-browser. First use downloads ONNX q8 weights. No Python or API key required. Results may differ from Python fp32; model limits are 128 / 512 tokens and long inputs may be truncated.'
    elif suffix=='.zh-Hant':text='線上完整版：輸入語料在瀏覽器中運行，首次下載 ONNX q8 模型。無需 Python 或 API Key。結果可能與 Python fp32 版不同；模型最多讀取 128 / 512 tokens，長文字可能截斷。'
    else:text=note['']
    for old,new in {
      '文件仅发给本地服务。':'语料只在浏览器内处理。',
      'Files are sent only to this local server.':'Corpus text stays in this browser.',
      '檔案僅發給本地服務。':'語料只在瀏覽器內處理。',
      '文件僅發給本地服務。':'語料只在瀏覽器內處理。',
      '安装模型依赖后，载入真实模型示例。首次运行需要联网下载权重。':'载入真实模型示例即可运行。首次需要下载模型，之后浏览器会缓存权重。',
      'Install the model dependencies and load the real-model example. First use downloads weights.':'Load the real-model example directly. First use downloads weights; subsequent visits use the browser cache.',
      '安裝模型依賴後，載入真實模型示例。首次運行需要聯網下載權重。':'載入真實模型示例即可運行。首次下載模型，之後瀏覽器會快取權重。',
      '将 ZIP 解压，在已安装项目的环境中运行 pytest test_regression.py。已知失败在修复前应当报错。':'真实模型案例导出 ZIP，内含 Node.js 回归测试与运行说明。演示案例导出 JSON。已知失败在修复前应报错。',
      'Extract the ZIP and run pytest test_regression.py in an environment with this project installed. Known failures should fail until repaired.':'Real-model cases export a ZIP with a Node.js regression test and instructions. Demo cases export JSON. Known failures should fail until repaired.',
      '將 ZIP 解壓，在已安裝項目的環境中運行 pytest test_regression.py。已知失敗在修復前應當報錯。':'真實模型案例匯出 ZIP，內含 Node.js 回歸測試與說明。演示案例匯出 JSON。已知失敗在修復前應報錯。'
    }.items():html=html.replace(old,new)
    html=html.replace('>单查询实验<','>源码与教程<').replace('>單查詢實驗<','>原始碼與教學<').replace('>Single-query lab<','>Source and guide<')
    html=html.replace('<main>' ,'<main><section class="wide"><p class="hint">'+text+'</p></section>',1)
    name='index.html' if not suffix else 'lab'+suffix+'.html';(dest/name).write_text(html)
    js=(src/('lab'+suffix+'.js')).read_text();a=js.index('async function api(');b=js.index('function config()',a)
    js=js[:a]+"const browserEngine=import('./browser-engine.js');\nasync function api(path,method='GET',body){return (await browserEngine).request(path,method,body)}\n"+js[b:]
    js=js.replace("fetch('/api/models')","fetch('./models.json')").replace("fetch('/static/lab-example.json')","fetch('./lab-example.json')")
    a=js.index("const response=await fetch('/api/lab'+base+'/export');");b=js.index("const a=document.createElement('a');",a)
    js=js[:a]+"const blob=await api(base+'/export');const u=URL.createObjectURL(blob);"+js[b:]
    js=js.replace("a.download='retrieval-regression.zip'","a.download=blob.type==='application/json'?'retrieval-case.json':'retrieval-regression.zip'")
    (dest/('lab'+suffix+'.js')).write_text(js)
for name in ['style.css','lab-example.json']:(dest/name).write_bytes((src/name).read_bytes())
(dest/'models.json').write_text(json.dumps({'demo':['demo-word','demo-char'],'local':['intfloat/multilingual-e5-small','sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']}))
language=(src/'language.js').read_text();a=language.index('const isSingle=');b=language.index('const pageLanguage=',a);language=language[:a]+'const isSingle=false;'+language[b:]
a=language.index('const target=');b=language.index('const switcher',a);language=language[:a]+"const target=language=>language==='zh'?'./index.html':'./lab.'+language+'.html';"+language[b:]
language=language.replace("script.src='/static/'","script.src='./'")
(dest/'language.js').write_text(language)
(dest/'.nojekyll').write_text('')
