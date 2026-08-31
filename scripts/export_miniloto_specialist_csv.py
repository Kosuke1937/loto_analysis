import json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
j=json.loads((ROOT/'data'/'miniloto-score-traces-1200-1399.json').read_text(encoding='utf-8'))
out=ROOT/'data'/'miniloto-specialist-traces-1200-1399.csv'
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f)
    w.writerow(['draw','layer','Committee015','A_Stat','A_Dynamic','B_Stat'])
    for r in j['rows']:
        w.writerow([r['draw'],r.get('layer',''),r.get('Committee015',''),r.get('A_Stat',''),r.get('A_Dynamic',''),r.get('B_Stat','')])
print(out)
