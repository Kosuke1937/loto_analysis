import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data'/'miniloto-a-dynamic-density-fast-800-1399.json'
o=json.loads(p.read_text(encoding='utf-8'))
out={'protocol':o['protocol'],'theoretical_null':o['theoretical_null'],'blocks':o['blocks']}
q=ROOT/'data'/'miniloto-a-dynamic-density-summary.json'
q.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
