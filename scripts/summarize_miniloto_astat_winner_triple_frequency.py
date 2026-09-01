import json
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'data'/'miniloto-astat-top100-winner-triple-frequency-fast.json'
d=json.loads(p.read_text())
out={'definition':d['definition']}
for k in ['development_800_999','validation_1000_1199','diagnostic_1200_1399']:
    v=d[k]
    out[k]={x:y for x,y in v.items() if x!='rows'}
q=p.with_name('miniloto-astat-top100-winner-triple-frequency-summary.json')
q.write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
