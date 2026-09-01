import itertools, json, re
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
idxmap={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1

num_prefix=np.zeros((T+1,32),dtype=np.int32)
for t,row in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]

def shape(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
A_SHAPES={'1-2-2-0','2-2-1-0','2-1-2-0','1-1-3-0','1-3-1-0','1-2-1-1','1-1-2-1'}
combo_shapes=np.array([shape(c) for c in combos],dtype=object); maskA=np.isin(combo_shapes,list(A_SHAPES)); Aidx=np.where(maskA)[0]
draw_shapes=[shape(a) for a in draws]

sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band,'consec':consec,'gap':gap}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot'); DYN=('prev','prev2','pbonus','hot')

actual=[]
for t in range(T):
    wi=idxmap[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band[wi]),'consec':int(consec[wi]),'gap':int(gap[wi]),
                   'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),
                   'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
                   'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})

def dyn_cats(t):
    p=draws[t-1] if t>=1 else np.array([],dtype=np.int16); p2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,p].sum(1).astype(np.int8) if len(p) else np.zeros(N,np.int8),
       'prev2':inc[:,p2].sum(1).astype(np.int8) if len(p2) else np.zeros(N,np.int8),
       'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); h=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,h].sum(1).astype(np.int8) if len(h) else np.zeros(N,np.int8)
    return d

def score_A(t,dynamic_only=False,hist=500,alpha=75):
    dyn=dyn_cats(t); out=np.full(N,-1e9,np.float32); s=np.zeros(N,np.float32); lo=max(0,t-hist)
    hs=[u for u in range(lo,t) if draw_shapes[u] in A_SHAPES]; nh=max(1,len(hs))
    for f in (DYN if dynamic_only else FEATURES):
        cats=static[f] if f in static else dyn[f]
        cc=np.bincount(cats[maskA],minlength=sizes[f]).astype(float); p=cc/max(cc.sum(),1)
        winc=np.bincount([actual[u][f] for u in hs],minlength=sizes[f]).astype(float)
        q=(winc+alpha*p)/(nh+alpha); w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        s += w[cats].astype(np.float32)
    out[maskA]=s[maskA]; return out

TH=[10,50,100,500,1000,3000,5000]

def one_model(sc,winner):
    ov=inc[:,winner].sum(1)
    # score-band rank: 1 + number of A candidates strictly above the best near-hit score
    res={}
    for m in [4,3]:
        target=maskA & (ov==m)
        mx=float(sc[target].max())
        band_rank=1+int(np.sum(maskA & (sc>mx)))
        tie_size=int(np.sum(maskA & (sc==mx)))
        res[f'best{m}_band_rank']=band_rank; res[f'best{m}_bestscore_tie_size']=tie_size
    # deterministic rank list for operational top-K diagnostics
    order=Aidx[np.lexsort((Aidx,-sc[Aidx]))]
    ovs=ov[order]
    for k in TH:
        res[f'top{k}_has4']=bool(np.any(ovs[:k]==4)); res[f'top{k}_has3']=bool(np.any(ovs[:k]==3))
        res[f'top{k}_count4']=int(np.sum(ovs[:k]==4)); res[f'top{k}_count3']=int(np.sum(ovs[:k]==3))
    return res

def summarize(recs,key):
    n=len(recs); out={'n':n}
    for m in [4,3]:
        rr=[r[key][f'best{m}_band_rank'] for r in recs]
        out[f'best{m}_band_rank_median']=float(np.median(rr)); out[f'best{m}_band_rank_mean']=float(np.mean(rr))
        for k in TH: out[f'best{m}_band_rank_le_{k}']=sum(x<=k for x in rr)
    for k in TH:
        for m in [4,3]:
            out[f'top{k}_draws_has{m}']=sum(r[key][f'top{k}_has{m}'] for r in recs)
            out[f'top{k}_avg_count{m}']=float(np.mean([r[key][f'top{k}_count{m}'] for r in recs]))
    return out

blocks={'development_800_999':range(800,1000),'validation_1000_1199':range(1000,1200),'diagnostic_1200_1399':range(1200,1400)}
out={'protocol':{'A_layer_only':True,'definition':'Exact overlap=4 and overlap=3 near-hit candidates among A-layer candidate universe. best band rank ignores arbitrary ordering inside score ties. Operational TopK uses deterministic score/index ordering.','excluded':[1400,1401]},'A_candidate_count':int(maskA.sum()),'blocks':{}}
for bn,rrs in blocks.items():
    recs=[]
    for rr in rrs:
        t=rr-1
        if draw_shapes[t] not in A_SHAPES: continue
        w=draws[t]; sa=score_A(t,False); sd=score_A(t,True)
        recs.append({'draw':rr,'AStat':one_model(sa,w),'ADynamic':one_model(sd,w)})
    out['blocks'][bn]={'A_draws':len(recs),'AStat':summarize(recs,'AStat'),'ADynamic':summarize(recs,'ADynamic')}
    print(bn,len(recs),flush=True)

path=ROOT/'data'/'miniloto-a-nearhit-ranks-800-1399.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',path)
