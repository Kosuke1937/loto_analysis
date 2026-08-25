import json, math, runpy, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']; a1_score=m['a1_score']; a2_score=m['a2_score']; z=m['z']
DEV=range(1000,1200); TEST=range(1200,1400)

def committee(t): return z(a1_score(t))+0.15*z(a2_score(t))
def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def random_exact5(pool):
    return math.comb(pool,5)/math.comb(31,5)
def random_4plus(pool):
    # P(X>=4), X hypergeom drawing pool digits from 31 against 5 winning digits
    den=math.comb(31,pool)
    p=0.0
    for x in (4,5):
        if x<=pool and pool-x<=26:
            p += math.comb(5,x)*math.comb(26,pool-x)/den
    return p

def num_scores(top,score,method,N):
    ids=top[:N]
    if method=='support':
        s=inc[ids].sum(0).astype(float)
    elif method=='rankweighted':
        w=1.0/np.log2(np.arange(2,N+2,dtype=float))
        s=(inc[ids]*w[:,None]).sum(0)
    elif method=='multiscale':
        s=np.zeros(31,float)
        for n,wgt in [(100,1.0),(500,0.8),(1500,0.5),(5000,0.25)]:
            x=inc[top[:n]].sum(0).astype(float)
            x=(x-x.mean())/(x.std()+1e-9)
            s += wgt*x
    else: raise ValueError(method)
    return s

def pool_eval(period,method,N,size):
    r5=r4=0; total=0
    for rr in period:
        t=rr-1; win=set(map(int,draws[t])); sc=committee(t); top=top_indices(sc,5000)
        ns=num_scores(top,sc,method,N)
        pick=set((np.argsort(-ns)[:size]+1).tolist())
        h=len(win&pick); r5+=int(h==5); r4+=int(h>=4); total+=1
    return {'n':total,'winner5':r5,'winner5_rate':round(r5/total,4),'winner4plus':r4,'winner4plus_rate':round(r4/total,4),
            'random5_rate':round(random_exact5(size),4),'random4plus_rate':round(random_4plus(size),4),
            'enrichment5':round((r5/total)/max(random_exact5(size),1e-12),3),'enrichment4plus':round((r4/total)/max(random_4plus(size),1e-12),3)}

def overlap_diag(period,K):
    cand=np.zeros(6,dtype=np.int64); maxc=np.zeros(6,dtype=np.int64); n=0
    for rr in period:
        t=rr-1; win=set(map(int,draws[t])); sc=committee(t); top=top_indices(sc,K)
        ov=np.array([len(win&set(map(int,combos[i]))) for i in top],dtype=int)
        for j in range(6): cand[j]+=int((ov==j).sum())
        maxc[int(ov.max())]+=1; n+=1
    tot=int(cand.sum())
    return {'draws':n,'K':K,'candidate_overlap_counts':cand.tolist(),
            'candidate_overlap_rates':[round(int(x)/tot,5) for x in cand],
            'mean_overlap':round(sum(i*int(cand[i]) for i in range(6))/tot,4),
            'max_overlap_draw_counts':maxc.tolist(),
            'draws_max3plus':int(maxc[3:].sum()),'draws_max4plus':int(maxc[4:].sum()),'draws_max5':int(maxc[5])}

def random_overlap_dist():
    den=math.comb(31,5); out=[]
    for x in range(6):
        out.append(math.comb(5,x)*math.comb(26,5-x)/den)
    return out

methods=['support','rankweighted','multiscale']; Ns=[100,500,1500,5000]; sizes=[12,14,16,18,20]
out={'protocol':{'development':[1000,1199],'fixed_test':[1200,1399],'excluded':[1400,1401]},
     'random_single_candidate_overlap_rates':[round(x,6) for x in random_overlap_dist()],
     'compression':{'development':{},'fixed_test':{}},'overlap':{'development':{},'fixed_test':{}}}
for method in methods:
    useNs=[5000] if method=='multiscale' else Ns
    for N in useNs:
        for size in sizes:
            key=f'{method}_N{N}_P{size}'
            out['compression']['development'][key]=pool_eval(DEV,method,N,size)
            out['compression']['fixed_test'][key]=pool_eval(TEST,method,N,size)
for K in [10,100,500,1000,5000]:
    out['overlap']['development'][str(K)]=overlap_diag(DEV,K)
    out['overlap']['fixed_test'][str(K)]=overlap_diag(TEST,K)

# rank the most interesting fixed-test compression settings by 5/5 enrichment, then absolute 5/5, then 4+
ranked=sorted(out['compression']['fixed_test'].items(),key=lambda kv:(-kv[1]['enrichment5'],-kv[1]['winner5_rate'],-kv[1]['winner4plus_rate']))
out['best_fixed_test_by_enrichment']=[{'setting':k,**v} for k,v in ranked[:10]]
path=ROOT/'data'/'miniloto-committee-compression-diagnostic.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
print('WROTE',path)
