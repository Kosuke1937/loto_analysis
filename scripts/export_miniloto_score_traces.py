import json, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'research_miniloto_committee_v2.py'))

draws=m['draws']; combos=m['combos']; combo_index=m['combo_index']; stat_score=m['stat_score']; z=m['z']
core_from_pairmetric=m['core_from_pairmetric']; pair_metric=m['pair_metric']; maskA=m['maskA']; maskB=m['maskB']; draw_shapes=m['draw_shapes']; A_SHAPES=m['A_SHAPES']; B_SHAPES=m['B_SHAPES']

rows=[]
for rr in range(1200,1400):
    t=rr-1; wi=combo_index[tuple(map(int,draws[t]))]
    s1=stat_score(t); z1=z(s1)
    cores={name:core_from_pairmetric(pair_metric(t,name)) for name in ['count100','count200','count300','count500','countall','lift300','lift500','under300','under500']}
    zc={k:z(v) for k,v in cores.items()}
    z300=zc['count300']
    a=z1-z1.min()+1e-3; c=z300-z300.min()+1e-3
    row={'draw':rr,'layer':('A' if draw_shapes[t] in A_SHAPES else 'B' if draw_shapes[t] in B_SHAPES else 'other'),
         'A1':float(z1[wi]),
         'core100':float(zc['count100'][wi]),'core200':float(zc['count200'][wi]),'core300':float(z300[wi]),
         'core500':float(zc['count500'][wi]),'coreAll':float(zc['countall'][wi]),
         'lift300':float(zc['lift300'][wi]),'lift500':float(zc['lift500'][wi]),
         'under300':float(zc['under300'][wi]),'under500':float(zc['under500'][wi]),
         'Committee015':float((z1+0.15*z300)[wi]),
         'Count200_015':float((z1+0.15*zc['count200'])[wi]),
         'Count300_030':float((z1+0.30*z300)[wi]),
         'Lift300_050':float((z1+0.50*zc['lift300'])[wi]),
         'Under500_030':float((z1+0.30*zc['under500'])[wi]),
         'Product025':float((a*np.power(c,0.25))[wi]),
         'Product050':float((a*np.power(c,0.50))[wi]),
         'Product100':float((a*c)[wi]),
         'Interact005':float((z1+0.15*z300+0.05*(z1*z300))[wi]),
         'Interact010':float((z1+0.15*z300+0.10*(z1*z300))[wi]),
         'Interact020':float((z1+0.15*z300+0.20*(z1*z300))[wi])}
    for lam in [0.0,0.05,0.10,0.15,0.20,0.30,0.50,1.0]:
        row[f'lambda_{lam:.2f}']=float((z1+lam*z300)[wi])
    if row['layer']=='A':
        sa=stat_score(t,role='A'); sad=stat_score(t,role='A',dynamic_only=True)
        za=z(sa,maskA); zad=z(sad,maskA)
        row['A_Stat']=float(za[wi]); row['A_Dynamic']=float(zad[wi])
    else:
        row['A_Stat']=None; row['A_Dynamic']=None
    if row['layer']=='B':
        sb=stat_score(t,role='B'); zb=z(sb,maskB); row['B_Stat']=float(zb[wi])
    else:
        row['B_Stat']=None
    rows.append(row)

path=ROOT/'data'/'miniloto-score-traces-1200-1399.json'
path.write_text(json.dumps({'note':'Winner score traces. Z-based series are standardized within each draw candidate universe; A/B specialist Z scores are within their role subset. Product series are shifted-positive diagnostics and not on the same Z scale.','rows':rows},ensure_ascii=False),encoding='utf-8')
print('WROTE',path,len(rows))