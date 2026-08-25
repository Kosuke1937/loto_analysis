import itertools, json, math, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
# ---------- load data ----------
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

# ---------- prefixes ----------
num_prefix=np.zeros((T+1,32),dtype=np.int32)
pair_prefix=np.zeros((T+1,32,32),dtype=np.int16)
for t,row in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1
    pair_prefix[t+1]=pair_prefix[t]
    for a,b in itertools.combinations(map(int,row),2):
        pair_prefix[t+1,a,b]+=1; pair_prefix[t+1,b,a]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]
def rpair(t,w): return pair_prefix[t]-pair_prefix[max(0,t-w)]

# ---------- A1 static/dynamic ----------
sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
static_prior={k:np.bincount(v,minlength=sizes[k]).astype(float) for k,v in static.items()}

# fixed role layer shapes from prior research
A_SHAPES={'1-2-2-0','2-2-1-0','2-1-2-0','1-1-3-0','1-3-1-0','1-2-1-1','1-1-2-1'}
B_SHAPES={'2-1-1-1','3-1-1-0','0-2-3-0','0-2-2-1','2-0-3-0','0-3-2-0','2-2-0-1','2-3-0-0','3-2-0-0'}
def shape_arr(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
combo_shapes=np.array([shape_arr(c) for c in combos],dtype=object)
maskA=np.isin(combo_shapes,list(A_SHAPES)); maskB=np.isin(combo_shapes,list(B_SHAPES))
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
DYNAMIC=('prev','prev2','pbonus','hot')

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
       'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
       'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

def stat_score(t,hist=500,alpha=75,role=None,dynamic_only=False):
    dyn=dyn_cats(t); score=np.zeros(N,np.float32); lo=max(0,t-hist)
    feats=DYNAMIC if dynamic_only else FEATURES
    if role=='A': cmask=maskA; wanted=A_SHAPES
    elif role=='B': cmask=maskB; wanted=B_SHAPES
    else: cmask=np.ones(N,dtype=bool); wanted=None
    hist_idx=[u for u in range(lo,t) if wanted is None or draw_shapes[u] in wanted]
    n=max(1,len(hist_idx))
    for f in feats:
        cats=static[f] if f in static else dyn[f]
        cc=np.bincount(cats[cmask],minlength=sizes[f]).astype(float)
        p=cc/max(cc.sum(),1)
        wins=np.bincount([actual[u][f] for u in hist_idx],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha)
        w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        score+=w[cats].astype(np.float32)
    score[~cmask]=-1e9
    return score

def z(x,mask=None):
    if mask is None:
        return (x-x.mean())/(x.std()+1e-9)
    y=x[mask]; out=np.full_like(x,-1e9,dtype=np.float32); out[mask]=(y-y.mean())/(y.std()+1e-9); return out

# ---------- core variants ----------
def core_from_pairmetric(pm):
    total=np.zeros(N,np.float32); incident=np.zeros((N,5),np.float32)
    for i,j in itertools.combinations(range(5),2):
        v=pm[combos[:,i],combos[:,j]].astype(np.float32); total+=v; incident[:,i]+=v; incident[:,j]+=v
    return np.max(total[:,None]-incident,axis=1)

def pair_metric(t,name):
    if name.startswith('count'):
        w=name[5:]
        pc=pair_prefix[t] if w=='all' else rpair(t,int(w))
        return pc.astype(np.float32)
    # smoothed recent/long-run rate log-ratio
    if name.startswith('lift') or name.startswith('under'):
        w=int(re.findall(r'\d+',name)[0]); lo=max(0,t-w); eff=max(1,t-lo)
        recent=(pair_prefix[t]-pair_prefix[lo]).astype(np.float32)
        allc=pair_prefix[t].astype(np.float32); longn=max(1,t)
        # 0.5 pseudocount on per-draw pair occurrence rate
        lr=np.log((recent+0.5)/(eff+1.0))-np.log((allc+0.5)/(longn+1.0))
        lr=np.clip(lr,-2.0,2.0)
        return -lr if name.startswith('under') else lr
    raise ValueError(name)

TH=[500,1000,3000,5000,10000]
def rank_of(score,wi): return 1+int(np.sum(score>score[wi]))
def empty_metric(): return {str(k):0 for k in TH}
def add_rank(metric,r):
    for k in TH: metric[str(k)]+=int(r<=k)

def topk_idx(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def exact_budget_union(main_score, specialist_scores, allocations, budget):
    chosen=[]; used=set()
    for sc,k in zip(specialist_scores,allocations):
        for ii in topk_idx(sc,k):
            ii=int(ii)
            if ii not in used: chosen.append(ii); used.add(ii)
    if len(chosen)<budget:
        for ii in topk_idx(main_score,budget):
            ii=int(ii)
            if ii not in used: chosen.append(ii); used.add(ii)
            if len(chosen)>=budget: break
    return set(chosen[:budget])

DEV=range(1000,1200); TEST=range(1200,1400)
core_names=['count100','count200','count300','count500','countall','lift300','lift500','under300','under500']
lambdas=[0.0,0.05,0.10,0.15,0.20,0.30,0.50,1.0]
product_betas=[0.25,0.5,1.0]
role_specs={
 'Main5000':(5000,0,0,False),
 'M4000+A500+B500':(4000,500,500,False),
 'M3500+A750+B750':(3500,750,750,False),
 'M3000+A1000+B1000':(3000,1000,1000,False),
 'M4000+ADyn500+B500':(4000,500,500,True),
}

def run_period(period):
    out={'n':len(list(period)),'A1':empty_metric(),'core_alone':{n:empty_metric() for n in core_names},
         'add':{n:{str(l):empty_metric() for l in lambdas} for n in core_names},
         'product_shifted':{b:empty_metric() for b in product_betas},
         'interaction':{g:empty_metric() for g in [0.05,0.10,0.20]},
         'roles':{k:0 for k in role_specs},
         'winner_ranks_A1':[],'winner_ranks_current':[]}
    for rr in period:
        t=rr-1; wi=combo_index[tuple(map(int,draws[t]))]
        s1=stat_score(t); z1=z(s1); r1=rank_of(s1,wi); add_rank(out['A1'],r1); out['winner_ranks_A1'].append(r1)
        cores={}
        for name in core_names:
            c=core_from_pairmetric(pair_metric(t,name)); cores[name]=c; add_rank(out['core_alone'][name],rank_of(c,wi))
            zc=z(c)
            for lam in lambdas:
                sc=z1+lam*zc; add_rank(out['add'][name][str(lam)],rank_of(sc,wi))
        zc300=z(cores['count300'])
        current=z1+0.15*zc300; out['winner_ranks_current'].append(rank_of(current,wi))
        # Direct multiplicative diagnostic after positive shift. Shift-dependence is why this is diagnostic only.
        a=z1-z1.min()+1e-3; c=zc300-zc300.min()+1e-3
        for beta in product_betas:
            sc=a*np.power(c,beta); add_rank(out['product_shifted'][beta],rank_of(sc,wi))
        for gamma in [0.05,0.10,0.20]:
            sc=z1+0.15*zc300+gamma*(z1*zc300); add_rank(out['interaction'][gamma],rank_of(sc,wi))
        # Specialist prediction sets. A/B definitions are reconstructed from documented method.
        a_stat=stat_score(t,role='A'); b_stat=stat_score(t,role='B'); a_dyn=stat_score(t,role='A',dynamic_only=True)
        for key,(m,aK,bK,use_dyn) in role_specs.items():
            if key=='Main5000': sel=set(map(int,topk_idx(current,5000)))
            else:
                asp=a_dyn if use_dyn else a_stat
                # exact 5000 budget: specialist selections first, then fill from Main
                sel=exact_budget_union(current,[asp,b_stat],[aK,bK],5000)
            out['roles'][key]+=int(wi in sel)
        if rr%25==0: print('done',rr,flush=True)
    out['median_rank_A1']=float(np.median(out.pop('winner_ranks_A1')))
    out['median_rank_current']=float(np.median(out.pop('winner_ranks_current')))
    return out

out={'protocol':{'development':[1000,1199],'fixed_test':[1200,1399],'excluded':[1400,1401],
                 'note':'A/B specialist implementations are reconstructed from documented definitions; compare to archived A-layer summary before canonical use.'},
     'definitions':{'current':'Z(A1)+0.15*Z(count300 4core)',
                    'lift':'pair log[(recent+0.5)/(window+1)] - log[(allprior+0.5)/(t+1)], clipped ±2; 4core=max four-number six-pair sum',
                    'under':'negative of lift, testing mean-reversion-style under-representation; diagnostic only',
                    'product_shifted':'(Z(A1)-min+eps) * (Z(core300)-min+eps)^beta; shift-dependent diagnostic, not probabilistically principled'},
     'development':run_period(DEV),'fixed_test':run_period(TEST)}

# choose additive model on DEV by Top5000, then Top10000, then Top3000; report corresponding TEST only (no retuning)
best=None; best_tuple=None
for name in core_names:
    for lam in lambdas:
        m=out['development']['add'][name][str(lam)]
        tup=(m['5000'],m['10000'],m['3000'],m['1000'])
        if best_tuple is None or tup>best_tuple:
            best_tuple=tup; best=(name,lam)
out['selected_on_development']={'core':best[0],'lambda':best[1],
                                'development':out['development']['add'][best[0]][str(best[1])],
                                'fixed_test':out['fixed_test']['add'][best[0]][str(best[1])]}
# choose role allocation on DEV, fixed test report
br=max(out['development']['roles'],key=lambda k:(out['development']['roles'][k], int(k.startswith('Main'))))
out['role_selected_on_development']={'setting':br,'development_top5000':out['development']['roles'][br],
                                     'fixed_test_top5000':out['fixed_test']['roles'][br]}

path=ROOT/'data'/'miniloto-committee-v2-research.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',path)
print(json.dumps({'selected':out['selected_on_development'],'role':out['role_selected_on_development'],
                  'test_A1':out['fixed_test']['A1'],'test_current':out['fixed_test']['add']['count300']['0.15']},ensure_ascii=False,indent=2))
