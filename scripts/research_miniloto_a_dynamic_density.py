import runpy, json, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'backtest_miniloto_a_dynamic_top10.py'))
combos=ns['combos']; draws=ns['draws']; maskA=ns['maskA']; A_idx=np.where(maskA)[0]
a_dynamic_score=ns['a_dynamic_score']; A_SHAPES=ns['A_SHAPES']; draw_shapes=ns['draw_shapes']

Ks=[100,300,500,1000,3000,5000]
number_cuts=[10,12,15]
pair_cuts=[30,50,100]
# A-layer background incidence, used for lift normalization
Acomb=combos[A_idx]
bg_num=np.bincount(Acomb.ravel(),minlength=32).astype(float)/len(Acomb)
all_pairs=list(itertools.combinations(range(1,32),2)); pair_to_i={p:i for i,p in enumerate(all_pairs)}
bg_pair=np.zeros(len(all_pairs),float)
for c in Acomb:
    for p in itertools.combinations(map(int,c),2): bg_pair[pair_to_i[p]]+=1
bg_pair/=len(Acomb)

def dense_metrics(sel,winner):
    cc=combos[sel]
    num=np.bincount(cc.ravel(),minlength=32).astype(float)
    # relative lift vs A-layer background; tiny floor avoids division by zero
    num_lift=(num/len(sel))/(bg_num+1e-12)
    num_order=np.lexsort((np.arange(32),-num_lift))
    num_rank=np.empty(32,int); num_rank[num_order]=np.arange(1,33)
    w=list(map(int,winner))
    nr=[int(num_rank[n]) for n in w]
    nout={'winner_num_mean_rank':float(np.mean(nr)), 'winner_num_median_rank':float(np.median(nr)),
          **{f'winner_num_in_top{c}':int(sum(r<=c for r in nr)) for c in number_cuts}}
    ps=np.zeros(len(all_pairs),float)
    for c in cc:
        for p in itertools.combinations(map(int,c),2): ps[pair_to_i[p]]+=1
    plift=(ps/len(sel))/(bg_pair+1e-12)
    po=np.lexsort((np.arange(len(all_pairs)),-plift)); prank=np.empty(len(all_pairs),int); prank[po]=np.arange(1,len(all_pairs)+1)
    wr=[int(prank[pair_to_i[p]]) for p in itertools.combinations(sorted(w),2)]
    pout={'winner_pair_mean_rank':float(np.mean(wr)), 'winner_pair_median_rank':float(np.median(wr)),
          **{f'winner_pair_in_top{c}':int(sum(r<=c for r in wr)) for c in pair_cuts}}
    return {**nout,**pout}

def block_summary(recs):
    out={}
    for K in Ks:
        rr=[r[str(K)] for r in recs]
        out[str(K)]={'n':len(rr)}
        for key in rr[0]: out[str(K)][key]=float(np.mean([x[key] for x in rr]))
    return out

records=[]
for rr in range(800,1400):
    t=rr-1
    if draw_shapes[t] not in A_SHAPES: continue
    s=a_dynamic_score(t)
    order=A_idx[np.lexsort((A_idx,-s[A_idx]))]
    rec={'draw':rr,'winner':list(map(int,draws[t]))}
    for K in Ks: rec[str(K)]=dense_metrics(order[:K],draws[t])
    records.append(rec)

blocks={
 'development_800_999':block_summary([r for r in records if 800<=r['draw']<=999]),
 'validation_1000_1199':block_summary([r for r in records if 1000<=r['draw']<=1199]),
 'diagnostic_1200_1399':block_summary([r for r in records if 1200<=r['draw']<=1399]),
 'all_800_1399':block_summary(records),
}
# null reference from uniform A-layer candidate set: fixed RNG, same number of A draws as each block, 25 replicates per draw/K
rng=np.random.default_rng(20260901)
def null_summary(n_draws):
    acc={K:[] for K in Ks}
    # use random synthetic winners from A-layer so null respects layer geometry
    for _ in range(n_draws*25):
        wi=int(rng.choice(A_idx)); w=combos[wi]
        for K in Ks:
            sel=rng.choice(A_idx,size=K,replace=False)
            acc[K].append(dense_metrics(sel,w))
    out={}
    for K in Ks:
        rr=acc[K]; out[str(K)]={}
        for key in rr[0]: out[str(K)][key]=float(np.mean([x[key] for x in rr]))
    return out
for name,lo,hi in [('development_800_999',800,999),('validation_1000_1199',1000,1199),('diagnostic_1200_1399',1200,1399)]:
    n=sum(lo<=r['draw']<=hi for r in records); blocks[name]['random_A_null']=null_summary(n)

out={
 'protocol':{'period':[800,1399], 'A_layer_only':True, 'Ks':Ks,
  'density':'Support is normalized by each number/pair baseline frequency in the full A-layer candidate universe. Lower support rank is better.',
  'blocks':'800-999 development; 1000-1199 validation; 1200-1399 diagnostic only. 1400/1401 excluded.',
  'null':'Uniform random A-layer candidate sets, 25 replications per actual A-draw count.'},
 'A_candidate_count':int(len(A_idx)), 'blocks':blocks, 'records':records
}
path=ROOT/'data'/'miniloto-a-dynamic-density-800-1399.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'protocol':out['protocol'],'blocks':blocks},ensure_ascii=False,indent=2))
print('WROTE',path)
