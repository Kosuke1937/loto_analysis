from pathlib import Path
import collections, itertools, json, re, statistics

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'miniloto-crowd-portfolio-v1.json'

# -------------------- data --------------------
rows=[]
for i in range(1,9):
    p=ROOT/'data'/f'miniloto-chunk-{i}.js'
    t=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?\s*$',t,re.S)
    if not m: raise RuntimeError(p)
    rows.extend(json.loads(m.group(1)))
rows.sort(key=lambda r:int(r[0]))
draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16)
bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16)
counts=[None if len(r)<=8 or r[8] is None else int(r[8]) for r in rows]
T=len(draws)
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16)
N=len(combos)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1

# Prefixes for canonical A1/A2.
num_prefix=np.zeros((T+1,32),dtype=np.int32)
pair_prefix=np.zeros((T+1,32,32),dtype=np.int16)
for t,a in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,a]+=1
    pair_prefix[t+1]=pair_prefix[t]
    for x,y in itertools.combinations(map(int,a),2):
        pair_prefix[t+1,x,y]+=1; pair_prefix[t+1,y,x]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]
def rpair(t,w): return pair_prefix[t]-pair_prefix[max(0,t-w)]

# -------------------- canonical Committee --------------------
sums=combos.sum(1)
sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8)
consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1)
gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
static_prior={k:np.bincount(v,minlength=sizes[k]).astype(float) for k,v in static.items()}
actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]
    c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set()
    cur=set(map(int,draws[t]))
    actual.append({
        'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),
        'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),
        'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
        'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),
        'hot':sum(int(n) in hot for n in draws[t])
    })
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16)
    prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={
        'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
        'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
        'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8),
    }
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

def a1_score(t,hist=500,alpha=75):
    dyn=dyn_cats(t); score=np.zeros(N,np.float32); lo=max(0,t-hist); n=max(1,t-lo)
    for f in FEATURES:
        cats=static[f] if f in static else dyn[f]
        cc=static_prior[f] if f in static_prior else np.bincount(cats,minlength=sizes[f]).astype(float)
        p=cc/cc.sum()
        wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha)
        w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
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

def state_base(order):
    for cap in (2,3,4):
        trial=greedy(order[:1500],10,1,2,cap,False)
        if len(trial)>=10: return trial[:10],cap
    return greedy(order[:1500],10,1,2,4,True),4

def finalize(score_for_choice,order,committee_top500):
    base,cap=state_base(order)
    support=inc[committee_top500].sum(0).astype(float)
    union=set()
    for idx in base: union.update(map(int,combos[idx]))
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=order[allowed[combos[order]].all(axis=1)]
    bset=set(base); pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
    if len(pool)==0: return base,cap
    supp=support[combos[pool]].mean(1); zz=(supp-supp.mean())/(supp.std()+1e-9)
    add=int(pool[np.argmax(score_for_choice[pool]-0.10*zz)])
    best=None; bestkey=None
    for rem in base:
        keep=[x for x in base if x!=rem]+[add]
        nu=set(); pu=set()
        for idx in keep:
            a=tuple(map(int,combos[idx])); nu.update(a); pu.update(itertools.combinations(a,2))
        key=(len(nu),len(pu),float(np.sum(score_for_choice[keep])))
        if bestkey is None or key>bestkey: bestkey=key; best=keep
    return best,cap

# -------------------- fixed Crowd-v1 regression --------------------
CROWD_NAMES=[
    'sum','min','max','range','odd','low12','low19','b1_1_9','b2_10_19','b3_20_29','b4_30_31','occupied_bands',
    'consecutive_pairs','gap_std','max_gap','min_gap','same_last_digit_pairs','prev_overlap','prev_bonus_in','shape_prior_freq_pct',
    'layer_A','layer_B','layer_C','layer_D','prev20_count_median','prev20_count_mean'
]

def band_shape_tuple(nums):
    c=[0,0,0,0]
    for n in nums: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return tuple(c)

def layer_from_freq(pct):
    return 'A' if pct>=5 else 'B' if pct>=2 else 'C' if pct>=.5 else 'D'

def hist_crowd_dataset():
    shape_seen=collections.Counter(); cnt_seen=[]; prev_nums=None; prev_bonus=None
    X=[]; y=[]; draw_ids=[]
    for idx,r in enumerate(rows):
        nums=list(map(int,r[2:7])); cnt=counts[idx]; shape=band_shape_tuple(nums)
        prior_n=idx; sf=(100.0*shape_seen[shape]/prior_n) if prior_n>=100 else np.nan
        layer=layer_from_freq(sf) if not np.isnan(sf) else 'NA'
        gaps=[nums[i+1]-nums[i] for i in range(4)]; bands=list(shape)
        last20=[c for c in cnt_seen[-20:] if c is not None]
        vals={
            'sum':sum(nums),'min':min(nums),'max':max(nums),'range':max(nums)-min(nums),'odd':sum(n%2 for n in nums),
            'low12':sum(n<=12 for n in nums),'low19':sum(n<=19 for n in nums),'b1_1_9':bands[0],'b2_10_19':bands[1],
            'b3_20_29':bands[2],'b4_30_31':bands[3],'occupied_bands':sum(x>0 for x in bands),
            'consecutive_pairs':sum(g==1 for g in gaps),'gap_std':float(np.std(gaps)),'max_gap':max(gaps),'min_gap':min(gaps),
            'same_last_digit_pairs':sum(1 for a,b in itertools.combinations(nums,2) if a%10==b%10),
            'prev_overlap':sum(n in set(prev_nums or []) for n in nums),'prev_bonus_in':int(prev_bonus in nums) if prev_bonus is not None else 0,
            'shape_prior_freq_pct':sf,'layer_A':int(layer=='A'),'layer_B':int(layer=='B'),'layer_C':int(layer=='C'),'layer_D':int(layer=='D'),
            'prev20_count_median':float(statistics.median(last20)) if last20 else np.nan,
            'prev20_count_mean':float(statistics.mean(last20)) if last20 else np.nan,
        }
        if cnt is not None and prior_n>=100 and last20:
            X.append([vals[n] for n in CROWD_NAMES]); y.append(cnt); draw_ids.append(int(r[0]))
        shape_seen[shape]+=1; cnt_seen.append(cnt); prev_nums=nums; prev_bonus=int(r[7])
    return np.array(X,float),np.array(y,int),np.array(draw_ids,int)

Xh,yh,idh=hist_crowd_dataset(); train=np.where(idh<=999)[0]
med=np.nanmedian(Xh[train],axis=0); Xi=Xh.copy(); bad=np.where(np.isnan(Xi)); Xi[bad]=med[bad[1]]
sc=StandardScaler().fit(Xi[train]); Xtr=sc.transform(Xi[train])
reg=Ridge(alpha=100.0).fit(Xtr,np.log1p(yh[train]))
coef=reg.coef_.astype(float); intercept=float(reg.intercept_); mean=sc.mean_.astype(float); scale=sc.scale_.astype(float)

# Candidate static Crowd features and their standardized linear contribution.
gaps=np.diff(combos,axis=1)
shape_codes=[tuple(map(int,x)) for x in np.column_stack([b0,b10,b20,b30])]
static_vals={
    'sum':sums.astype(float),'min':combos[:,0].astype(float),'max':combos[:,-1].astype(float),'range':(combos[:,-1]-combos[:,0]).astype(float),
    'odd':odd.astype(float),'low12':(combos<=12).sum(1).astype(float),'low19':(combos<=19).sum(1).astype(float),
    'b1_1_9':b0.astype(float),'b2_10_19':b10.astype(float),'b3_20_29':b20.astype(float),'b4_30_31':b30.astype(float),
    'occupied_bands':np.column_stack([b0,b10,b20,b30]).astype(bool).sum(1).astype(float),'consecutive_pairs':consec.astype(float),
    'gap_std':np.std(gaps,axis=1).astype(float),'max_gap':gaps.max(1).astype(float),'min_gap':gaps.min(1).astype(float),
    'same_last_digit_pairs':np.array([sum(1 for a,b in itertools.combinations(map(int,c),2) if a%10==b%10) for c in combos],float),
}
name_to_j={n:i for i,n in enumerate(CROWD_NAMES)}
base_linear=np.full(N,intercept,dtype=float)
for n,v in static_vals.items():
    j=name_to_j[n]; base_linear += coef[j]*((v-mean[j])/scale[j])

shape_prefix=[]; cc=collections.Counter()
for a in draws:
    shape_prefix.append(cc.copy()); cc[band_shape_tuple(map(int,a))]+=1

def crowd_predict_all(t):
    lin=base_linear.copy()
    # Dynamic candidate features.
    prev=draws[t-1]; ov=inc[:,prev].sum(1).astype(float)
    pb=inc[:,int(bonuses[t-1])].astype(float)
    prior=shape_prefix[t]
    sf=np.array([100.0*prior[s]/t for s in shape_codes],dtype=float)
    la=(sf>=5).astype(float); lb=((sf>=2)&(sf<5)).astype(float); lc=((sf>=.5)&(sf<2)).astype(float); ld=(sf<.5).astype(float)
    last20=[c for c in counts[max(0,t-20):t] if c is not None]
    med20=float(statistics.median(last20)); mean20=float(statistics.mean(last20))
    dyn={'prev_overlap':ov,'prev_bonus_in':pb,'shape_prior_freq_pct':sf,'layer_A':la,'layer_B':lb,'layer_C':lc,'layer_D':ld,
         'prev20_count_median':np.full(N,med20),'prev20_count_mean':np.full(N,mean20)}
    for n,v in dyn.items():
        j=name_to_j[n]; lin += coef[j]*((v-mean[j])/scale[j])
    return np.clip(np.expm1(lin),0,None)

# -------------------- evaluation --------------------
def metrics_for(ids,t,crowd_pred):
    win=set(map(int,draws[t])); tickets=[set(map(int,combos[i])) for i in ids]
    best=max(len(win&s) for s in tickets)
    union=set().union(*tickets)
    pairs=set(); triples=set()
    for s in tickets:
        a=sorted(s); pairs.update(itertools.combinations(a,2)); triples.update(itertools.combinations(a,3))
    wp=set(itertools.combinations(sorted(win),2)); wt=set(itertools.combinations(sorted(win),3))
    vals=np.array([crowd_pred[i] for i in ids],float)
    return {'best':best,'union_recall':len(win&union),'pair_recall':len(wp&pairs),'triple_recall':len(wt&triples),
            'crowd_mean':float(vals.mean()),'crowd_median':float(np.median(vals))}

LAMBDAS=[0.0,0.02,0.05,0.10,0.15,0.20,0.30]
per_lambda={lam:[] for lam in LAMBDAS}

for rr in range(1000,1400):
    t=rr-1
    s1=a1_score(t); s2=a2_score(t); cs=z(s1)+0.15*z(s2)
    top5k=top_indices(cs,5000); top500=top5k[:500]
    cp=crowd_predict_all(t)
    logc=np.log1p(cp[top5k]); cz=(logc-logc.mean())/(logc.std()+1e-9)
    # lam=0 exactly reproduces canonical ordering/score.
    for lam in LAMBDAS:
        if lam==0.0:
            order=top5k; choice=cs
        else:
            adj_local=cs[top5k]-lam*cz
            order=top5k[np.lexsort((top5k,-adj_local))]
            choice=cs.copy(); choice[top5k]=adj_local
        ids,cap=finalize(choice,order,top500)
        m=metrics_for(ids,t,cp); m.update({'draw':rr,'cap':cap,'actual_first_prize_count':counts[t]})
        if rr==1395:
            winner_idx=combo_index[tuple(map(int,draws[t]))]; m['winner_selected']=winner_idx in ids
        per_lambda[lam].append(m)
    if rr%25==0: print('done',rr,flush=True)

def summarize(rows0,a,b):
    q=[r for r in rows0 if a<=r['draw']<=b]
    return {
        'n':len(q),'best3plus':sum(r['best']>=3 for r in q),'best4plus':sum(r['best']>=4 for r in q),'exact5':sum(r['best']==5 for r in q),
        'mean_union_recall':float(np.mean([r['union_recall'] for r in q])),'mean_pair_recall':float(np.mean([r['pair_recall'] for r in q])),
        'mean_triple_recall':float(np.mean([r['triple_recall'] for r in q])),'portfolio_crowd_mean':float(np.mean([r['crowd_mean'] for r in q])),
        'portfolio_crowd_median_mean':float(np.mean([r['crowd_median'] for r in q])),
    }

val={str(lam):summarize(per_lambda[lam],1000,1199) for lam in LAMBDAS}
basev=val['0.0']
# Validation selection: never sacrifice exact/4+; maximize 3+, then minimize predicted crowd.
elig=[]
for lam in LAMBDAS:
    s=val[str(lam)]
    if s['exact5']>=basev['exact5'] and s['best4plus']>=basev['best4plus']:
        elig.append((s['best3plus'],-s['portfolio_crowd_mean'],-lam,lam))
chosen=max(elig)[3] if elig else 0.0

test_base=summarize(per_lambda[0.0],1200,1399)
test_chosen=summarize(per_lambda[chosen],1200,1399)
# Paired gains/losses on fixed test.
bmap={r['draw']:r for r in per_lambda[0.0] if 1200<=r['draw']<=1399}
cmap={r['draw']:r for r in per_lambda[chosen] if 1200<=r['draw']<=1399}
gain3=[d for d in bmap if bmap[d]['best']<3 and cmap[d]['best']>=3]
loss3=[d for d in bmap if bmap[d]['best']>=3 and cmap[d]['best']<3]
gain4=[d for d in bmap if bmap[d]['best']<4 and cmap[d]['best']>=4]
loss4=[d for d in bmap if bmap[d]['best']>=4 and cmap[d]['best']<4]

r1395={str(lam):next(r for r in per_lambda[lam] if r['draw']==1395) for lam in LAMBDAS}
all_test={str(lam):summarize(per_lambda[lam],1200,1399) for lam in LAMBDAS}

out={
    'model':'MiniLoto Crowd-aware Portfolio v1',
    'design':{
        'committee':'canonical A1 + 0.15*4core; Committee Top5000 membership fixed',
        'crowd':'Crowd-v1 Ridge(log1p first-prize count), train <=999, alpha=100 fixed from prior validation study',
        'portfolio_adjustment':'within fixed Committee Top5000, reorder by CommitteeScore - lambda * Z(log1p predicted crowd); same state 2/3/4 diversification and anti-support 0.10',
        'lambda_selection':'draw 1000-1199 only: require exact5 and best4+ no worse than canonical, then maximize best3+, then minimize predicted crowd',
        'fixed_test':'draw 1200-1399; not used to choose lambda',
        'note':'Crowd score predicts sharing/popularity if the combination wins; it is not an increase in lottery draw probability.'
    },
    'validation_by_lambda':val,
    'chosen_lambda':chosen,
    'fixed_test':{
        'canonical':test_base,'crowd_aware':test_chosen,
        'crowd_mean_reduction_pct':100*(1-test_chosen['portfolio_crowd_mean']/test_base['portfolio_crowd_mean']),
        'gain3_draws':gain3,'loss3_draws':loss3,'gain4_draws':gain4,'loss4_draws':loss4,
    },
    'test_all_lambdas_diagnostic_not_for_selection':all_test,
    'draw1395':r1395,
}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chosen_lambda':chosen,'validation':val[str(chosen)],'fixed_test':out['fixed_test'],'draw1395':r1395[str(chosen)]},ensure_ascii=False,indent=2))
