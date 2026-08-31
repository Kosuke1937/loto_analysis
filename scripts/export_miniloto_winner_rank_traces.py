import json, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'research_miniloto_committee_v2.py'))

draws=ns['draws']; combo_index=ns['combo_index']; stat_score=ns['stat_score']; z=ns['z']; core_from_pairmetric=ns['core_from_pairmetric']; pair_metric=ns['pair_metric']; rank_of=ns['rank_of']; draw_shapes=ns['draw_shapes']; A_SHAPES=ns['A_SHAPES']; B_SHAPES=ns['B_SHAPES']

rows=[]
for rr in range(1200,1400):
    t=rr-1
    wi=combo_index[tuple(map(int,draws[t]))]
    s1=stat_score(t); z1=z(s1)
    c300=core_from_pairmetric(pair_metric(t,'count300'))
    current=z1+0.15*z(c300)
    shape=draw_shapes[t]
    layer='A' if shape in A_SHAPES else 'B' if shape in B_SHAPES else 'other'
    row={
        'draw':rr,
        'layer':layer,
        'A1_rank':rank_of(s1,wi),
        'Committee_rank':rank_of(current,wi),
        'A_Stat_rank':None,
        'A_Dynamic_rank':None,
        'B_Stat_rank':None,
    }
    if layer=='A':
        row['A_Stat_rank']=rank_of(stat_score(t,role='A'),wi)
        row['A_Dynamic_rank']=rank_of(stat_score(t,role='A',dynamic_only=True),wi)
    elif layer=='B':
        row['B_Stat_rank']=rank_of(stat_score(t,role='B'),wi)
    rows.append(row)
    if rr%25==0: print('rank done',rr,flush=True)

out={'note':'Exact winner ranks among 169,911 combinations for A1/Committee. A/B specialist ranks are within their role subset because non-role candidates are assigned -1e9. Lower rank is better.', 'rows':rows}
path=ROOT/'data'/'miniloto-winner-rank-traces-1200-1399.json'
path.write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
print('WROTE',path)
