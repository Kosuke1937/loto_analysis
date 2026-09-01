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
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1

# prefixes
num_prefix=np.zeros((T+1,32),dtype=np.int32)
for t,row in enumerate(draws):
    num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1

def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]

# A shapes
A_SHAPES={'1-2-2-0','2-2-1-0','2-1-2-0','1-1-3-0','1-3-1-0','1-2-1-1','1-1-2-1'}
def shape_arr(a):
    c=[0,0,0,0]
    for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
combo_shapes=np.array([shape_arr(c) for c in combos],dtype=object)
maskA=np.isin(combo_shapes,list(A_SHAPES)); Aidx=np.where(maskA)[0]
draw_shapes=[shape_arr(a) for a in draws]

# A-Stat features
sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8)
odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1)
band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}
sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}
FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')
actual=[]
for t in range(T):
    wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300)
    hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
    actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),
      'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),
      'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),
      'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})

def dyn_cats(t):
    prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
    d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),
       'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),
       'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
    c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int)
    d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8)
    return d

def astat_score(t,hist=500,alpha=75):
    dyn=dyn_cats(t); s=np.zeros(N,np.float32); lo=max(0,t-hist)
    hist_idx=[u for u in range(lo,t) if draw_shapes[u] in A_SHAPES]; n=max(1,len(hist_idx))
    for f in FEATURES:
        cats=static[f] if f in static else dyn[f]
        cc=np.bincount(cats[maskA],minlength=sizes[f]).astype(float); p=cc/max(cc.sum(),1)
        wins=np.bincount([actual[u][f] for u in hist_idx],minlength=sizes[f]).astype(float)
        q=(wins+alpha*p)/(n+alpha)
        w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5)
        s+=w[cats].astype(np.float32)
    s[~maskA]=-1e9
    return s

def topk(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def overlap(a,b): return len(set(map(int,a))&set(map(int,b)))

def rank_desc(vals):
    # rank 1 = highest; stable by key index later
    order=np.argsort(-vals,kind='mergesort'); ranks=np.empty(len(vals),int); ranks[order]=np.arange(1,len(vals)+1); return ranks

pair_list=list(itertools.combinations(range(1,32),2)); pair_to_i={p:i for i,p in enumerate(pair_list)}
tri_list=list(itertools.combinations(range(1,32),3)); tri_to_i={p:i for i,p in enumerate(tri_list)}

METHODS=['num','pair','triple','num_pair','num_pair_triple']
WEIGHTS={'num':(1,0,0),'pair':(0,1,0),'triple':(0,0,1),'num_pair':(1,1,0),'num_pair_triple':(1,1,1)}
POOL_MS=[8,10,12,15]

def evaluate_block(start,end):
    records=[]
    for rr in range(start,end+1):
        t=rr-1
        if draw_shapes[t] not in A_SHAPES: continue
        winner=set(map(int,draws[t])); score=astat_score(t); top100=topk(score,100); tc=combos[top100]
        num=np.zeros(32,float); pair=np.zeros(len(pair_list),float); tri=np.zeros(len(tri_list),float)
        # rank weighting: top ranks get more influence, but raw frequency retained separately
        num_raw=np.zeros(32,int); pair_raw=np.zeros(len(pair_list),int); tri_raw=np.zeros(len(tri_list),int)
        for r,c in enumerate(tc,1):
            w=1.0/np.sqrt(r)
            cc=list(map(int,c))
            for n in cc: num[n]+=w; num_raw[n]+=1
            for p in itertools.combinations(cc,2): pair[pair_to_i[p]]+=w; pair_raw[pair_to_i[p]]+=1
            for tr in itertools.combinations(cc,3): tri[tri_to_i[tr]]+=w; tri_raw[tri_to_i[tr]]+=1
        # raw frequency ranks for direct hypothesis
        nr=rank_desc(num_raw[1:]); pr=rank_desc(pair_raw); trr=rank_desc(tri_raw)
        win_num_ranks=[int(nr[n-1]) for n in winner]
        win_pairs=list(itertools.combinations(sorted(winner),2)); win_tris=list(itertools.combinations(sorted(winner),3))
        win_pair_ranks=[int(pr[pair_to_i[p]]) for p in win_pairs]
        win_tri_ranks=[int(trr[tri_to_i[x]]) for x in win_tris]
        rec={'draw':rr,
             'num_top5':sum(x<=5 for x in win_num_ranks),'num_top10':sum(x<=10 for x in win_num_ranks),'num_top15':sum(x<=15 for x in win_num_ranks),
             'num_mean_rank':float(np.mean(win_num_ranks)),
             'pair_top10':sum(x<=10 for x in win_pair_ranks),'pair_top30':sum(x<=30 for x in win_pair_ranks),'pair_top50':sum(x<=50 for x in win_pair_ranks),'pair_mean_rank':float(np.mean(win_pair_ranks)),
             'tri_top10':sum(x<=10 for x in win_tri_ranks),'tri_top30':sum(x<=30 for x in win_tri_ranks),'tri_top50':sum(x<=50 for x in win_tri_ranks),'tri_mean_rank':float(np.mean(win_tri_ranks)),
             'assemblies':{}}
        # standardized supports
        nz=num[1:]; zn=(num-np.mean(nz))/(np.std(nz)+1e-9)
        zp=(pair-np.mean(pair))/(np.std(pair)+1e-9); zt=(tri-np.mean(tri))/(np.std(tri)+1e-9)
        # top number pools then all 5-combos in pool, A-layer only
        num_order=np.argsort(-num[1:],kind='mergesort')+1
        for m in POOL_MS:
            pool=set(map(int,num_order[:m])); cand=[i for i in Aidx if set(map(int,combos[i])).issubset(pool)]
            pool_has=int(winner.issubset(pool))
            for meth in METHODS:
                key=f'm{m}_{meth}'; a,b,cw=WEIGHTS[meth]
                if not cand:
                    rec['assemblies'][key]={'pool_has_winner':pool_has,'best_match':0,'exact':0}
                    continue
                vals=[]
                for i in cand:
                    cc=list(map(int,combos[i])); v=a*sum(zn[n] for n in cc)
                    if b: v+=b*sum(zp[pair_to_i[p]] for p in itertools.combinations(cc,2))
                    if cw: v+=cw*sum(zt[tri_to_i[x]] for x in itertools.combinations(cc,3))
                    vals.append(v)
                vals=np.asarray(vals); order=np.argsort(-vals,kind='mergesort')[:10]; sel=[cand[j] for j in order]
                best=max(overlap(combos[i],winner) for i in sel); exact=any(set(map(int,combos[i]))==winner for i in sel)
                rec['assemblies'][key]={'pool_has_winner':pool_has,'best_match':int(best),'exact':int(exact)}
        records.append(rec)
        if rr%25==0: print('done',rr,flush=True)
    def avg(k): return float(np.mean([r[k] for r in records]))
    out={'A_draws':len(records),
         'support':{
           'winner_num_top5_avg':avg('num_top5'),'winner_num_top10_avg':avg('num_top10'),'winner_num_top15_avg':avg('num_top15'),'winner_num_mean_rank':avg('num_mean_rank'),
           'winner_pair_top10_avg':avg('pair_top10'),'winner_pair_top30_avg':avg('pair_top30'),'winner_pair_top50_avg':avg('pair_top50'),'winner_pair_mean_rank':avg('pair_mean_rank'),
           'winner_tri_top10_avg':avg('tri_top10'),'winner_tri_top30_avg':avg('tri_top30'),'winner_tri_top50_avg':avg('tri_top50'),'winner_tri_mean_rank':avg('tri_mean_rank')},
         'assembly':{}}
    for m in POOL_MS:
        for meth in METHODS:
            key=f'm{m}_{meth}'; aa=[r['assemblies'][key] for r in records]
            out['assembly'][key]={'pool_has_winner':sum(x['pool_has_winner'] for x in aa),'exact5':sum(x['exact'] for x in aa),
               'best4plus':sum(x['best_match']>=4 for x in aa),'best3plus':sum(x['best_match']>=3 for x in aa),'avg_best':float(np.mean([x['best_match'] for x in aa]))}
    return out

blocks={'development_800_999':evaluate_block(800,999),'validation_1000_1199':evaluate_block(1000,1199),'diagnostic_1200_1399':evaluate_block(1200,1399)}
# choose method using development only: exact5,4+,3+,pool_has, then smaller pool preferred
best=None; bt=None
for key,m in blocks['development_800_999']['assembly'].items():
    pool=int(key.split('_')[0][1:]); tup=(m['exact5'],m['best4plus'],m['best3plus'],m['pool_has_winner'],-pool)
    if bt is None or tup>bt: bt=tup; best=key
out={'protocol':{'A_layer_only':True,'top100':'A-Stat Top100','development':[800,999],'validation':[1000,1199],'diagnostic':[1200,1399],'excluded':[1400,1401],
 'support':'raw Top100 occurrence frequency ranks; assembly scores use rank-weighted 1/sqrt(rank) number/pair/triple support'},
 'random_reference':{'winner_num_top5':5*5/31,'winner_num_top10':5*10/31,'winner_num_top15':5*15/31,
 'winner_pair_top10':10*10/465,'winner_pair_top30':10*30/465,'winner_pair_top50':10*50/465,
 'winner_tri_top10':10*10/4495,'winner_tri_top30':10*30/4495,'winner_tri_top50':10*50/4495},
 'selected_on_development':best,'blocks':blocks}
path=ROOT/'data'/'miniloto-astat-top100-reverse-800-1399.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'selected':best,'random_reference':out['random_reference'],'development':blocks['development_800_999'],'validation':blocks['validation_1000_1199'],'diagnostic':blocks['diagnostic_1200_1399']},ensure_ascii=False,indent=2))
print('WROTE',path)
