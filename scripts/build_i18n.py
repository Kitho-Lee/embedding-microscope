"""Build English authored UI strings. Runtime corpus data is never translated."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'app/static'
pairs=json.loads((p/'ui-en.json').read_text())
pattern=re.compile('|'.join(re.escape(k) for k in sorted(pairs,key=len,reverse=True)))
def translate(text):return pattern.sub(lambda m:pairs[m.group()],text)
for base in ['lab','index']:
    text=translate((p/(base+'.html')).read_text()).replace('lang="zh-CN"','lang="en"')
    if base=='index':
        for zh in ['否定','数字阈值','实体','关系方向','同义改写','自定义']:
            text=text.replace('<option>'+pairs[zh]+'</option>','<option value="'+zh+'">'+pairs[zh]+'</option>')
    (p/(base+'.en.html')).write_text(text)
for base in ['lab','app']:
    text=translate((p/(base+'.js')).read_text())
    if base=='app':
        for zh in ['同义改写','自定义']:
            text=text.replace(".value='"+pairs[zh]+"'", ".value='"+zh+"'")
        text=text.replace("||'Custom'", "||'自定义'")
    if base=='lab':
        text=text.replace('${c.edit_count} edits', "${c.edit_count} ${c.edit_count===1?'edit':'edits'}")
        text=text.replace('${t.indices.length} edits', "${t.indices.length} ${t.indices.length===1?'edit':'edits'}")
    (p/(base+'.en.js')).write_text(text)

# Traditional Chinese converts authored UI strings only, never experiment data.
from opencc import OpenCC
converter=OpenCC('s2t')
def traditional(text):return pattern.sub(lambda m:converter.convert(m.group()),text)
for base in ['lab','index']:
    text=traditional((p/(base+'.html')).read_text()).replace('lang="zh-CN"','lang="zh-Hant"')
    if base=='index':
        for zh in ['否定','数字阈值','实体','关系方向','同义改写','自定义']:
            text=text.replace('<option>'+converter.convert(zh)+'</option>','<option value="'+zh+'">'+converter.convert(zh)+'</option>')
    (p/(base+'.zh-Hant.html')).write_text(text)
for base in ['lab','app']:
    text=traditional((p/(base+'.js')).read_text())
    if base=='app':
        for zh in ['同义改写','自定义']:
            text=text.replace(".value='"+converter.convert(zh)+"'", ".value='"+zh+"'")
        text=text.replace("||'"+converter.convert('自定义')+"'", "||'自定义'")
    (p/(base+'.zh-Hant.js')).write_text(text)
