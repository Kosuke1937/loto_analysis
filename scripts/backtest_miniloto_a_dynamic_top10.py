import itertools, json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    mm=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if mm: rows.extend(json.loads(mm.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]))
draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16)
bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16)
T=len(draws)
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16); N=len(combos)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1

# prefixes
num_prefix=np.zeros((T+1,32),dtype=np.int32)
for t,row in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]

# A-layer shapes from documented role research
A_SHAPES={'1-2-2-0','2-2-1-0','2-1-2-0','1-1-3-0','1-3-1-0','1-2-1-1','1-1-2-1'}
def shape_arr(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
combo_shapes=np.array([shape_arr(c) for c in combos],dtype=object)
maskA=np.isin(combo_shapes,list(A_SHAPES))
draw_shapes=[shape_arr(a) for a in draws]
A_idx=np.where(maskA)[0]

# dynamic feature categories over all candidates
sizes={'prev':6,'prev2':6,'pbonus':2,'hot':6}
actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({
        'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),
        'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
        'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),
        'hot':sum(int(n) in hot for n in draws[t])})

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16)
    prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={
      'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
      'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
      'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

def a_dynamic_score(t,hist=500,alpha=75):
    dyn=dyn_cats(t); score=np.full(N,-1e9,np.float32)
    s=np.zeros(N,np.float32); lo=max(0,t-hist)
    hist_idx=[u for u in range(lo,t) if draw_shapes[u] in A_SHAPES]
    n=max(1,len(hist_idx))
    for f in ('prev','prev2','pbonus','hot'):
        cats=dyn[f]
        cc=np.bincount(cats[maskA],minlength=sizes[f]).astype(float)
        p=cc/max(cc.sum(),1)
        wins=np.bincount([actual[u][f] for u in hist_idx],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha)
        w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        s += w[cats].astype(np.float32)
    score[maskA]=s[maskA]
    return score

def top10(score):
    # deterministic tie break by combination index
    idx=np.argpartition(-score,9)[:10]
    return idx[np.lexsort((idx,-score[idx]))]

def overlap(a,b): return len(set(map(int,a)) & set(map(int,b)))

def summarize(records):
    n=len(records)
    if not n: return {}
    best=[r['best_match'] for r in records]
    recall=[r['winner_number_recall_union'] for r in records]
    return {
      'n':n,
      'exact5_top10':sum(r['exact_winner_in_top10'] for r in records),
      'best_match_counts':{str(k):sum(x==k for x in best) for k in range(6)},
      'draws_best3plus':sum(x>=3 for x in best),
      'draws_best4plus':sum(x>=4 for x in best),
      'union_recall_counts':{str(k):sum(x==k for x in recall) for k in range(6)},
      'union5':sum(x==5 for x in recall),
      'union4plus':sum(x>=4 for x in recall),
      'assembly_fail_union5_but_no_exact':sum((r['winner_number_recall_union']==5 and not r['exact_winner_in_top10']) for r in records),
      'avg_union_size':float(np.mean([r['union_size'] for r in records])),
      'avg_best_match':float(np.mean(best)),
      'avg_winner_number_recall_union':float(np.mean(recall)),
    }

records=[]
for rr in range(1200,1400):
    t=rr-1; winner=draws[t]; wi=combo_index[tuple(map(int,winner))]
    score=a_dynamic_score(t); sel=top10(score)
    tickets=[list(map(int,combos[i])) for i in sel]
    matches=[overlap(combos[i],winner) for i in sel]
    uni=sorted(set().union(*[set(x) for x in tickets]))
    wset=set(map(int,winner))
    rec=len(wset & set(uni))
    exact=wi in set(map(int,sel))
    records.append({
      'draw':rr,'layer':'A' if draw_shapes[t] in A_SHAPES else 'nonA','winner':list(map(int,winner)),
      'top10':tickets,'exact_winner_in_top10':bool(exact),'winner_rank_within_A':(1+int(np.sum(score[maskA]>score[wi]))) if maskA[wi] else None,
      'best_match':int(max(matches)),'match_counts':matches,'union':uni,'union_size':len(uni),
      'winner_numbers_in_union':sorted(wset & set(uni)),'winner_numbers_missing_union':sorted(wset-set(uni)),
      'winner_number_recall_union':rec
    })

A_records=[r for r in records if r['layer']=='A']
nonA=[r for r in records if r['layer']=='nonA']
out={
 'protocol':{
   'period':[1200,1399],
   'definition':'A-Dynamic = A-layer candidates only; dynamic features prev, prev2, previous bonus, Hot15; 500-draw history; alpha75; clip ±1.5. Top10 = raw score top10, no diversification.',
   'important':'A-layer membership of the future winner is not knowable pre-draw. A-layer-only summary is an oracle/conditional diagnostic. All-200 summary is what happens if A-Dynamic is blindly used every draw.'
 },
 'A_candidate_count':int(maskA.sum()),
 'A_layer_actual_draws':len(A_records),
 'nonA_actual_draws':len(nonA),
 'A_layer_only':summarize(A_records),
 'all_200_blind_use':summarize(records),
 'records':records
}
path=ROOT/'data'/'miniloto-a-dynamic-top10-backtest-1200-1399.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='records'},ensure_ascii=False,indent=2))
print('WROTE',path)
