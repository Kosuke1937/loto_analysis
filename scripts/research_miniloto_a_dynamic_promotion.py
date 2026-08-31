import runpy, json, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'backtest_miniloto_a_dynamic_top10.py'))
combos=ns['combos']; draws=ns['draws']; bonuses=ns['bonuses']; T=ns['T']; N=ns['N']; inc=ns['inc']; maskA=ns['maskA']; A_idx=np.where(maskA)[0]
a_dynamic_score=ns['a_dynamic_score']; A_SHAPES=ns['A_SHAPES']; draw_shapes=ns['draw_shapes']; actual_dyn=ns['actual']; rnum=ns['rnum']

# Reconstruct A-Stat exactly from documented A1-style role model
sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')
# actual categories
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]
    d={f:int(static[f][wi]) for f in static}; d.update(actual_dyn[t]); actual.append(d)

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
       'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
       'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

def a_stat_score(t,hist=500,alpha=75):
    dyn=dyn_cats(t); score=np.zeros(N,np.float32); lo=max(0,t-hist)
    hist_idx=[u for u in range(lo,t) if draw_shapes[u] in A_SHAPES]; n=max(1,len(hist_idx))
    for f in FEATURES:
        cats=static[f] if f in static else dyn[f]
        cc=np.bincount(cats[maskA],minlength=sizes[f]).astype(float); p=cc/max(cc.sum(),1)
        wins=np.bincount([actual[u][f] for u in hist_idx],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha)
        w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        score+=w[cats].astype(np.float32)
    score[~maskA]=-1e9
    return score

def z_role(x):
    y=x[maskA]; out=np.full(N,-1e9,np.float32); out[maskA]=(y-y.mean())/(y.std()+1e-9); return out

def top10_order(primary,secondary=None):
    if secondary is None:
        order=A_idx[np.lexsort((A_idx,-primary[A_idx]))]
    else:
        # primary desc, then secondary desc, then index asc
        order=A_idx[np.lexsort((A_idx,-secondary[A_idx],-primary[A_idx]))]
    return order[:10]

def pool_rerank(dyn,ast,K):
    ordD=A_idx[np.lexsort((A_idx,-dyn[A_idx]))]
    kth=dyn[ordD[min(K-1,len(ordD)-1)]]
    pool=A_idx[dyn[A_idx]>=kth]  # expand ties at cutoff
    order=pool[np.lexsort((pool,-ast[pool]))]
    return order[:10],len(pool)

def overlap(a,b): return len(set(map(int,a))&set(map(int,b)))
def eval_sel(sel,winner,wi):
    matches=[overlap(combos[i],winner) for i in sel]
    uni=sorted(set(combos[sel].ravel().tolist())); wset=set(map(int,winner)); rec=len(wset&set(uni))
    return {'exact':int(wi in set(map(int,sel))),'best':int(max(matches)),'recall':rec,'union_size':len(uni)}

def summarize(xs):
    return {'n':len(xs),'exact5':sum(x['exact'] for x in xs),'best3plus':sum(x['best']>=3 for x in xs),
            'best4plus':sum(x['best']>=4 for x in xs),'union5':sum(x['recall']==5 for x in xs),
            'union4plus':sum(x['recall']>=4 for x in xs),'avg_best':float(np.mean([x['best'] for x in xs])),
            'avg_recall':float(np.mean([x['recall'] for x in xs])),'avg_union_size':float(np.mean([x['union_size'] for x in xs]))}

Ks=[100,300,500,1000,3000,5000]; lambdas=[0.05,0.10,0.20,0.50,1.0]
records=[]
for rr in range(800,1400):
    t=rr-1
    if draw_shapes[t] not in A_SHAPES: continue
    winner=draws[t]; wi=combo_index[tuple(map(int,winner))]
    dyn=a_dynamic_score(t); ast=a_stat_score(t); zd=z_role(dyn); za=z_role(ast)
    # tie diagnostics for winner
    sw=float(dyn[wi]); higher=int(np.sum(dyn[maskA]>sw)); equal=int(np.sum(dyn[maskA]==sw))
    topv=float(np.max(dyn[maskA])); top_tie=int(np.sum(dyn[maskA]==topv))
    rec={'draw':rr,'winner':list(map(int,winner)),'dyn_winner_rank_start':higher+1,'dyn_winner_tie_size':equal,'dyn_top_tie_size':top_tie,'methods':{},'pool_sizes':{}}
    rec['methods']['raw_dynamic']=eval_sel(top10_order(dyn),winner,wi)
    rec['methods']['dynamic_tiebreak_astat']=eval_sel(top10_order(dyn,ast),winner,wi)
    rec['methods']['astat_raw']=eval_sel(top10_order(ast),winner,wi)
    for K in Ks:
        sel,psz=pool_rerank(dyn,ast,K); rec['pool_sizes'][str(K)]=psz
        rec['methods'][f'dynpool{K}_astat']=eval_sel(sel,winner,wi)
    for lam in lambdas:
        sc=zd+lam*za; rec['methods'][f'blend_{lam:.2f}']=eval_sel(top10_order(sc),winner,wi)
    records.append(rec)

method_names=list(records[0]['methods'])
def block(lo,hi):
    rr=[r for r in records if lo<=r['draw']<=hi]
    out={'A_draws':len(rr),'tie_diagnostics':{
      'avg_winner_tie_size':float(np.mean([r['dyn_winner_tie_size'] for r in rr])),
      'median_winner_tie_size':float(np.median([r['dyn_winner_tie_size'] for r in rr])),
      'avg_top_tie_size':float(np.mean([r['dyn_top_tie_size'] for r in rr])),
      'median_top_tie_size':float(np.median([r['dyn_top_tie_size'] for r in rr]))}}
    out['methods']={m:summarize([r['methods'][m] for r in rr]) for m in method_names}
    out['avg_expanded_pool_size']={str(K):float(np.mean([r['pool_sizes'][str(K)] for r in rr])) for K in Ks}
    return out
blocks={'development_800_999':block(800,999),'validation_1000_1199':block(1000,1199),'diagnostic_1200_1399':block(1200,1399)}
# Select on development lexicographically: exact5, best4+, best3+, union5, union4+, then smaller union size
scores=blocks['development_800_999']['methods']
def key(m):
    x=scores[m]; return (x['exact5'],x['best4plus'],x['best3plus'],x['union5'],x['union4plus'],-x['avg_union_size'])
selected=max(method_names,key=key)
out={'protocol':{'A_layer_only':True,'development':[800,999],'validation':[1000,1199],'diagnostic':[1200,1399],'excluded':[1400,1401],
 'methods':'raw A-Dynamic; A-Stat tie-break; A-Dynamic tie-expanded TopK pool then A-Stat rerank; weak z-score blends.',
 'selection':'method chosen using development only by exact5, 4+, 3+, union5, union4+; validation and diagnostic not used to choose.'},
 'selected_on_development':selected,'blocks':blocks,'records':records}
path=ROOT/'data'/'miniloto-a-dynamic-promotion-800-1399.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
compact={'protocol':out['protocol'],'selected_on_development':selected,'blocks':blocks}
(ROOT/'data'/'miniloto-a-dynamic-promotion-summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(compact,ensure_ascii=False,indent=2)); print('WROTE',path)
