# Generates verified Stat / 4core / Committee winner ranks for draws 1200-1399.
import ast, glob, itertools, json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if not m:
        raise RuntimeError(f'cannot parse {p}')
    rows.extend(ast.literal_eval(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]))
draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16)
bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16)
T=len(draws)
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16)
N=len(combos)
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
actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
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
def rank_of(score,t):
    wi=combo_index[tuple(map(int,draws[t]))]; return 1+int(np.sum(score>score[wi]))

def band_shape(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
shape_counts={}
for a in draws:
    s=band_shape(a); shape_counts[s]=shape_counts.get(s,0)+1

def layer(freq): return 'A' if freq>=5 else 'B' if freq>=2 else 'C' if freq>=.5 else 'D'

out=[]
for rr in range(1200,1400):
    t=rr-1; s1=a1_score(t); s2=a2_score(t); cm=z(s1)+0.15*z(s2); wi=combo_index[tuple(map(int,draws[t]))]
    sh=band_shape(draws[t]); freq=100*shape_counts[sh]/T
    out.append({'draw':rr,'nums':[int(x) for x in draws[t]],'statScore':round(float(s1[wi]),6),'coreScore':round(float(s2[wi]),6),'committeeScore':round(float(cm[wi]),6),'statRank':rank_of(s1,t),'coreRank':rank_of(s2,t),'committeeRank':rank_of(cm,t),'shape':sh,'freq':round(freq,4),'layer':layer(freq)})
    if rr%20==0: print('done',rr,flush=True)

r1395=next(x for x in out if x['draw']==1395)
assert r1395['statRank']==106, r1395
assert r1395['committeeRank']==127, r1395
checks={k:sum(x['committeeRank']<=k for x in out) for k in [500,1000,3000,5000,10000]}
assert checks=={500:2,1000:3,3000:8,5000:12,10000:20},checks
text='window.MINI_MODEL_RANKS='+json.dumps(out,ensure_ascii=False,separators=(',',':'))+';\n'
(ROOT/'data'/'miniloto-model-ranks-1200-1399.js').write_text(text,encoding='utf-8')
print('wrote',len(out),'rows',checks,r1395)
