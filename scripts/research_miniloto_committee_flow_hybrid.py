import collections, itertools, json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if m: rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]))
draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16)
bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16)
T=len(draws)
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16); N=len(combos)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1

num_prefix=np.zeros((T+1,32),dtype=np.int32)
pair_prefix=np.zeros((T+1,32,32),dtype=np.int16)
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

def shape_arr(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
combo_shapes=np.array([shape_arr(c) for c in combos],dtype=object)
draw_shapes=[shape_arr(a) for a in draws]

actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),
                   'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),
                   'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
                   'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
       'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
       'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

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
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        idx=int(idx); nums=tuple(map(int,combos[idx]))
        trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
        if any(tc[x]>=triple_cap for x in trs): continue
        if any(pc[x]>=pair_cap for x in prs): continue
        if any(nc[x]>=num_cap for x in nums): continue
        sel.append(idx)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in nums: nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for idx in indices:
            idx=int(idx)
            if idx not in used:
                sel.append(idx); used.add(idx)
                if len(sel)>=K: break
    return sel

def canonical_final(score):
    top=top_indices(score,5000)
    base=[]; used_cap=4
    for cap in (2,3,4):
        trial=greedy(top[:1500],10,1,2,cap,False)
        if len(trial)>=10:
            base=trial[:10]; used_cap=cap; break
    if len(base)<10: base=greedy(top[:1500],10,1,2,4,True)
    support=inc[top[:500]].sum(0).astype(float)
    union=set(); [union.update(map(int,combos[i])) for i in base]
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=top[allowed[combos[top]].all(axis=1)]
    bset=set(base); pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
    if len(pool):
        supp=support[combos[pool]].mean(1); zz=(supp-supp.mean())/(supp.std()+1e-9)
        add=int(pool[np.argmax(score[pool]-0.10*zz)])
        best=None; bestkey=None
        for rem in base:
            keep=[x for x in base if x!=rem]+[add]
            nu=set(); pu=set()
            for idx in keep:
                nums=tuple(map(int,combos[idx])); nu.update(nums); pu.update(itertools.combinations(nums,2))
            key=(len(nu),len(pu),float(np.sum(score[keep])))
            if bestkey is None or key>bestkey: bestkey=key; best=keep
        base=best
    return base,used_cap

def rolling_layer_map(t,window=500):
    lo=max(0,t-window); cnt=collections.Counter(draw_shapes[lo:t]); den=max(1,t-lo)
    mp={}
    for sh in set(combo_shapes):
        f=100*cnt.get(sh,0)/den
        mp[sh]='A' if f>=5 else 'B' if f>=2 else 'C' if f>=0.5 else 'D'
    return mp

def flow_score_all(t, layer_map):
    prev=set(map(int,draws[t-1])); prev2=set(map(int,draws[t-2]))
    pm1=set()
    for n in prev:
        for x in (n-1,n+1):
            if 1<=x<=31 and x not in prev: pm1.add(x)
    p2=set()
    for n in prev2:
        for x in (n-1,n,n+1):
            if 1<=x<=31: p2.add(x)
    c_prev=inc[:,list(prev)].sum(1)
    c_pm1=inc[:,list(pm1)].sum(1)
    c_p2=inc[:,list(p2)].sum(1)
    s=np.zeros(N,np.float32)
    s += np.where(c_prev==1,1.0,np.where(c_prev==0,0.45,-0.8)).astype(np.float32)
    s += np.where(c_pm1==1,1.0,np.where((c_pm1==0)|(c_pm1==2),0.25,-0.7)).astype(np.float32)
    s += np.where((c_p2>=2)&(c_p2<=3),1.0,np.where((c_p2==1)|(c_p2==4),0.25,-0.7)).astype(np.float32)
    s += np.where((sums>=80)&(sums<=110),1.0,np.where(((sums>=70)&(sums<80))|((sums>110)&(sums<=120)),0.25,-0.6)).astype(np.float32)
    ls=np.array([{'A':1.0,'B':0.75,'C':0.15,'D':-0.5}[layer_map[x]] for x in combo_shapes],dtype=np.float32)
    s += ls
    return s

def hybrid_final(t, committee, canonical_ids, lam):
    layer_map=rolling_layer_map(t,500)
    union=set(); [union.update(map(int,combos[i])) for i in canonical_ids]
    ulist=sorted(union)
    mask=inc[:,ulist].sum(1)==5
    idx=np.where(mask)[0]
    fs=flow_score_all(t,layer_map)
    sc=committee + lam*z(fs)
    idx=idx[np.lexsort((idx,-sc[idx]))]
    # portfolio target: A 4-5, B 2-3, C 1-2, D <=1; soft sum 80-110 already in flow score.
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter(); lc=collections.Counter()
    def ok(i):
        nums=tuple(map(int,combos[i])); sh=combo_shapes[i]; L=layer_map[sh]
        if L=='D' and lc['D']>=1: return False
        if L=='C' and lc['C']>=2: return False
        if L=='B' and lc['B']>=3: return False
        if L=='A' and lc['A']>=5: return False
        trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
        if any(tc[x]>=1 for x in trs): return False
        if any(pc[x]>=2 for x in prs): return False
        if any(nc[x]>=4 for x in nums): return False
        return True
    # staged minima A4/B2/C1 where feasible
    for target,need in [('A',4),('B',2),('C',1)]:
        for i in idx:
            i=int(i)
            if lc[target]>=need: break
            if layer_map[combo_shapes[i]]!=target or i in sel or not ok(i): continue
            sel.append(i); lc[target]+=1
            nums=tuple(map(int,combos[i]));
            for x in itertools.combinations(nums,3): tc[x]+=1
            for x in itertools.combinations(nums,2): pc[x]+=1
            for x in nums: nc[x]+=1
    for i in idx:
        i=int(i)
        if len(sel)>=10: break
        if i in sel or not ok(i): continue
        sel.append(i); lc[layer_map[combo_shapes[i]]]+=1
        nums=tuple(map(int,combos[i]));
        for x in itertools.combinations(nums,3): tc[x]+=1
        for x in itertools.combinations(nums,2): pc[x]+=1
        for x in nums: nc[x]+=1
    # fallback without layer maxima but preserve D<=1 and diversity
    if len(sel)<10:
        for i in idx:
            i=int(i)
            if len(sel)>=10: break
            if i in sel: continue
            L=layer_map[combo_shapes[i]]
            if L=='D' and lc['D']>=1: continue
            nums=tuple(map(int,combos[i])); trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
            if any(tc[x]>=1 for x in trs) or any(pc[x]>=2 for x in prs) or any(nc[x]>=4 for x in nums): continue
            sel.append(i); lc[L]+=1
            for x in trs: tc[x]+=1
            for x in prs: pc[x]+=1
            for x in nums: nc[x]+=1
    if len(sel)<10:
        for i in idx:
            i=int(i)
            if len(sel)>=10: break
            if i not in sel: sel.append(i)
    return sel[:10],dict(lc),len(union)

def evaluate(period,lams):
    out={str(l):{'n':0,'d3':0,'d4':0,'d5':0,'sum_in_80_110':0,'layer_counts':collections.Counter(),'union_recall_sum':0,'union5':0} for l in lams}
    base={'n':0,'d3':0,'d4':0,'d5':0,'union_recall_sum':0,'union5':0}
    for rr in period:
        t=rr-1; s1=a1_score(t); s2=a2_score(t); cm=z(s1)+0.15*z(s2)
        canon,_=canonical_final(cm); win=set(map(int,draws[t])); base['n']+=1
        bm=max(len(win&set(map(int,combos[i]))) for i in canon); base['d3']+=bm>=3; base['d4']+=bm>=4; base['d5']+=bm>=5
        uni=set(); [uni.update(map(int,combos[i])) for i in canon]; ur=len(win&uni); base['union_recall_sum']+=ur; base['union5']+=ur==5
        for lam in lams:
            ids,lc,usize=hybrid_final(t,cm,canon,lam); d=out[str(lam)]; d['n']+=1
            bm=max(len(win&set(map(int,combos[i]))) for i in ids); d['d3']+=bm>=3; d['d4']+=bm>=4; d['d5']+=bm>=5
            uni=set(); [uni.update(map(int,combos[i])) for i in ids]; ur=len(win&uni); d['union_recall_sum']+=ur; d['union5']+=ur==5
            d['sum_in_80_110']+=sum(80<=int(sums[i])<=110 for i in ids)
            layer_map=rolling_layer_map(t,500)
            for i in ids: d['layer_counts'][layer_map[combo_shapes[i]]]+=1
        if rr%25==0: print('done',rr,flush=True)
    base['mean_union_recall']=base['union_recall_sum']/base['n']
    for d in out.values():
        d['mean_union_recall']=d['union_recall_sum']/d['n']; d['mean_sum_in_80_110']=d['sum_in_80_110']/d['n']; d['layer_counts']=dict(d['layer_counts'])
    return base,out

LAM=[0.10,0.20,0.30,0.40]
dev_base,dev=evaluate(range(1000,1200),LAM)
# choose on dev: d4, then d3, then union5, then closer to 8 of 10 sums in 80-110
best=max(LAM,key=lambda l:(dev[str(l)]['d4'],dev[str(l)]['d3'],dev[str(l)]['union5'],-abs(dev[str(l)]['mean_sum_in_80_110']-8)))
test_base,test=evaluate(range(1200,1400),[best])
# current 1403 pre-draw, using history through 1402
cur_t=T
s1=a1_score(cur_t); s2=a2_score(cur_t); cm=z(s1)+0.15*z(s2); canon,cap=canonical_final(cm); ids,lc,usize=hybrid_final(cur_t,cm,canon,best)
layer_map=rolling_layer_map(cur_t,500)
current=[]
for pos,i in enumerate(ids,1):
    nums=list(map(int,combos[i])); current.append({'position':pos,'numbers':nums,'sum':int(sum(nums)),'shape':combo_shapes[i],'layer':layer_map[combo_shapes[i]],'committee_rank':1+int(np.sum(cm>cm[i]))})

out={'protocol':{'development':[1000,1199],'fixed_test':[1200,1399],'current_target':1403,'note':'Hybrid reuses only the number-union of canonical Committee final10, then reassembles with pre-draw flow score and rolling-500 ABCD layers. Lambda selected only on development.'},
     'flow_definition':{'prev_same':'best=1, 0 mild, >=2 penalty','prev_pm1':'best=1, 0/2 mild, >=3 penalty','prev2_same_pm1':'best=2-3, 1/4 mild, 0/5 penalty','sum':'80-110 preferred, 70-79/111-120 mild, otherwise penalty','ABCD':'A>B>C>D soft score; portfolio aims A4-5,B2-3,C1-2,D<=1'},
     'development':{'canonical':dev_base,'hybrid_by_lambda':dev,'selected_lambda':best},
     'fixed_test':{'canonical':test_base,'hybrid':test[str(best)]},
     'current_1403':{'lambda':best,'canonical_cap':cap,'canonical_union_size':usize,'layer_counts':lc,'tickets':current}}
path=ROOT/'data'/'miniloto-committee-flow-hybrid-backtest.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',path)
print(json.dumps(out,ensure_ascii=False)[:5000])