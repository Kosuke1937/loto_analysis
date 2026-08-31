import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
o=json.loads((ROOT/'data'/'miniloto-a-dynamic-promotion-summary.json').read_text(encoding='utf-8'))
keep=['raw_dynamic','dynamic_tiebreak_astat','astat_raw','dynpool100_astat','dynpool500_astat','dynpool1000_astat','blend_0.20','blend_0.50','blend_1.00']
out={'selected_on_development':o['selected_on_development'],'blocks':{}}
for b,v in o['blocks'].items():
    out['blocks'][b]={'A_draws':v['A_draws'],'tie_diagnostics':v['tie_diagnostics'],'methods':{k:v['methods'][k] for k in keep}}
q=ROOT/'data'/'miniloto-a-dynamic-promotion-slim.json'; q.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
