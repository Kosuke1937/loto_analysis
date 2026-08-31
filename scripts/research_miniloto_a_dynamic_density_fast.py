import runpy, json, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'backtest_miniloto_a_dynamic_top10.py'))
combos=ns['combos']; draws=ns['draws']; maskA=ns['maskA']; A_idx=np.where(maskA)[0]
a_dynamic_score=ns['a_dynamic_score']; A_SHAPES=ns['A_SHAPES']; draw_shapes=ns['draw_shapes']
Ks=[100,300,500,1000,3000,5000]
number_cuts=[10,12,15]; pair_cuts=[30,50,100]
Acomb=combos[A_idx]
# background probabilities inside A layer
bg_num=np.bincount(Acomb.ravel(),minlength=32).astype(float)/len(Acomb)
pairs=list(itertools.combinations(range(1,32),2)); p2i={p:i for i,p in enumerate(pairs)}
pair_cols=np.array([[p2i[p] for p in itertools.combinations(map(int,c),2)] for c in combos],dtype=np.int16)
bg_pair=np.bincount(pair_cols[A_idx].ravel(),minlength=len(pairs)).astype(float)/len(A_idx)

def metrics(sel,winner):
    cc=combos[sel]
    nsup=np.bincount(cc.ravel(),minlength=32).astype(float)/len(sel)
    nlift=nsup/(bg_num+1e-12); order=np.lexsort((np.arange(32),-nlift)); rank=np.empty(32,int); rank[order]=np.arange(1,33)
    wr=[int(rank[int(n)]) for n in winner]
    psup=np.bincount(pair_cols[sel].ravel(),minlength=len(pairs)).astype(float)/len(sel)
    plift=psup/(bg_pair+1e-12); po=np.lexsort((np.arange(len(pairs)),-plift)); prank=np.empty(len(pairs),int); prank[po]=np.arange(1,len(pairs)+1)
    wpr=[int(prank[p2i[p]]) for p in itertools.combinations(sorted(map(int,winner)),2)]
    return {
      'winner_num_mean_rank':float(np.mean(wr)),
      'winner_num_median_rank':float(np.median(wr)),
      **{f'winner_num_in_top{c}':int(sum(r<=c for r in wr)) for c in number_cuts},
      'winner_pair_mean_rank':float(np.mean(wpr)),
      'winner_pair_median_rank':float(np.median(wpr)),
      **{f'winner_pair_in_top{c}':int(sum(r<=c for r in wpr)) for c in pair_cuts},
    }

def summarize(rr):
    out={}
    for K in Ks:
      xs=[r[str(K)] for r in rr]; out[str(K)]={'n':len(xs)}
      for key in xs[0]: out[str(K)][key]=float(np.mean([x[key] for x in xs]))
    return out

records=[]
for rr in range(800,1400):
    t=rr-1
    if draw_shapes[t] not in A_SHAPES: continue
    s=a_dynamic_score(t)
    order=A_idx[np.lexsort((A_idx,-s[A_idx]))]
    r={'draw':rr,'winner':list(map(int,draws[t]))}
    for K in Ks: r[str(K)]=metrics(order[:K],draws[t])
    records.append(r)

blocks={}
for name,lo,hi in [('development_800_999',800,999),('validation_1000_1199',1000,1199),('diagnostic_1200_1399',1200,1399),('all_800_1399',800,1399)]:
    rr=[r for r in records if lo<=r['draw']<=hi]; blocks[name]=summarize(rr)

null={
 'winner_num_mean_rank':16.0,
 'winner_num_median_rank_approx':16.0,
 **{f'winner_num_in_top{c}':5*c/31 for c in number_cuts},
 'winner_pair_mean_rank':233.0,
 'winner_pair_median_rank_approx':233.0,
 **{f'winner_pair_in_top{c}':10*c/465 for c in pair_cuts},
}
# enrichment ratios versus theoretical null for intuitive comparison
for b in blocks.values():
    for K,v in b.items():
      v['enrichment']={
       **{f'num_top{c}_x':v[f'winner_num_in_top{c}']/null[f'winner_num_in_top{c}'] for c in number_cuts},
       **{f'pair_top{c}_x':v[f'winner_pair_in_top{c}']/null[f'winner_pair_in_top{c}'] for c in pair_cuts},
       'num_mean_rank_improvement_x':null['winner_num_mean_rank']/v['winner_num_mean_rank'],
       'pair_mean_rank_improvement_x':null['winner_pair_mean_rank']/v['winner_pair_mean_rank'],
      }

out={'protocol':{
 'period':[800,1399],'A_layer_only':True,'Ks':Ks,
 'density':'Top-K A-Dynamic candidate number/pair support divided by full A-layer baseline support; lower support rank is better.',
 'blocks':'800-999 development, 1000-1199 validation, 1200-1399 diagnostic only; 1400/1401 excluded.',
 'null':'Theoretical rank-uniform reference after baseline normalization: number mean rank 16 among 31, pair mean rank 233 among 465.'},
 'A_candidate_count':int(len(A_idx)),'theoretical_null':null,'blocks':blocks,'records':records}
path=ROOT/'data'/'miniloto-a-dynamic-density-fast-800-1399.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'protocol':out['protocol'],'theoretical_null':null,'blocks':blocks},ensure_ascii=False,indent=2))
print('WROTE',path)
