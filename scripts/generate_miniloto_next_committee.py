import collections, itertools, json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if not m: raise RuntimeError(f'cannot parse {p}')
    rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]))
draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16)
bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16)
T=len(draws); target_draw=int(rows[-1][0])+1
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16); N=len(combos)
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1
num_prefix=np.zeros((T+1,32),dtype=np.int32); pair_prefix=np.zeros((T+1,32,32),dtype=np.int16)
for t,row in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1
    pair_prefix[t+1]=pair_prefix[t]
    for a,b in itertools.combinations(map(int,row),2):
        pair_prefix[t+1,a,b]+=1; pair_prefix[t+1,b,a]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]
def rpair(t,w): return pair_prefix[t]-pair_prefix[max(0,t-w)]
sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
static_prior={k:np.bincount(v,minlength=sizes[k]).astype(float) for k,v in static.items()}
actual=[]
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),
      'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())), 'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
      'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')
def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8),'prev2':inc[:,prev2].sum(1).astype(np.int8),'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]
    d['hot']=inc[:,hot].sum(1).astype(np.int8); return d
def a1_score(t,hist=500,alpha=75):
    dyn=dyn_cats(t); score=np.zeros(N,np.float32); lo=max(0,t-hist); n=max(1,t-lo)
    for f in FEATURES:
        cats=static[f] if f in static else dyn[f]
        cc=static_prior[f] if f in static_prior else np.bincount(cats,minlength=sizes[f]).astype(float)
        p=cc/cc.sum(); wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha); w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        score+=w[cats].astype(np.float32)
    return score
def a2_score(t):
    pc=rpair(t,300); total=np.zeros(N,np.float32); incident=np.zeros((N,5),np.float32)
    for i,j in itertools.combinations(range(5),2):
        v=pc[combos[:,i],combos[:,j]].astype(np.float32); total+=v; incident[:,i]+=v; incident[:,j]+=v
    return np.max(total[:,None]-incident,axis=1)
def z(x): return (x-x.mean())/(x.std()+1e-9)
def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]; return idx[np.lexsort((idx,-score[idx]))]
def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        idx=int(idx); nums=tuple(map(int,combos[idx])); trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
        if any(tc[x]>=triple_cap for x in trs) or any(pc[x]>=pair_cap for x in prs) or any(nc[x]>=num_cap for x in nums): continue
        sel.append(idx)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in nums: nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for idx in indices:
            idx=int(idx)
            if idx not in used: sel.append(idx); used.add(idx)
            if len(sel)>=K: break
    return sel
def anti_support(score,top5000):
    base=[]; used_cap=None
    for cap in (2,3,4):
        trial=greedy(top5000[:1500],10,1,2,cap,False)
        if len(trial)>=10: base=trial[:10]; used_cap=cap; break
    if len(base)<10: base=greedy(top5000[:1500],10,1,2,4,True); used_cap=4
    support=inc[top5000[:500]].sum(0).astype(float); union=set()
    for idx in base: union.update(map(int,combos[idx]))
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=top5000[allowed[combos[top5000]].all(axis=1)]; bset=set(base); pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
    if len(pool)==0: return base,used_cap,None
    supp=support[combos[pool]].mean(1); zz=(supp-supp.mean())/(supp.std()+1e-9); add=int(pool[np.argmax(score[pool]-0.10*zz)])
    best=None; bestkey=None; removed=None
    for rem in base:
        keep=[x for x in base if x!=rem]+[add]; nu=set(); pu=set()
        for idx in keep:
            nums=tuple(map(int,combos[idx])); nu.update(nums); pu.update(itertools.combinations(nums,2))
        key=(len(nu),len(pu),float(np.sum(score[keep])))
        if bestkey is None or key>bestkey: bestkey=key; best=keep; removed=rem
    return best,used_cap,{'added':add,'removed':int(removed)}
def shape(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))

t=T; s1=a1_score(t); s2=a2_score(t); cs=z(s1)+0.15*z(s2); top=top_indices(cs,5000); ids,cap,anti=anti_support(cs,top)
rank=np.empty(N,dtype=np.int32); rank[top_indices(cs,N)]=np.arange(1,N+1,dtype=np.int32)
prev=set(map(int,draws[-1])); prev2=set(map(int,draws[-2]))
def rel(a):
    A=set(map(int,a)); prev_pm=sum(any(abs(n-p)==1 for p in prev) for n in A); prev2_pm=sum(any(abs(n-p)<=1 for p in prev2) for n in A)
    return len(A&prev),prev_pm,prev2_pm
records=[]
for pos,idx in enumerate(ids,1):
    a=tuple(map(int,combos[idx])); same,pm1,p2=rel(a)
    records.append({'position':pos,'numbers':list(a),'sum':sum(a),'shape':shape(a),'odd_even':f"{sum(n%2 for n in a)}:{sum(n%2==0 for n in a)}",'consecutive_pairs':sum(b-a0==1 for a0,b in zip(a,a[1:])),
                    'committee_rank':int(rank[idx]),'committee_score':round(float(cs[idx]),6),'a1_score':round(float(s1[idx]),6),'core4_score':round(float(s2[idx]),6),
                    'prev_same':same,'prev_pm1':pm1,'prev2_same_pm1':p2})
union=sorted(set().union(*(set(r['numbers']) for r in records)))
out={'target_draw':target_draw,'history_through':int(rows[-1][0]),'method':'canonical A1 + 0.15*4core -> state 2/3/4 diversification -> anti-support 0.10','cap_used':cap,'anti_support':anti,
     'previous_draw':list(map(int,draws[-1])),'previous2_draw':list(map(int,draws[-2])),'tickets':records,'union_numbers':union,'union_size':len(union)}
path=ROOT/'data'/'miniloto-next-committee.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))