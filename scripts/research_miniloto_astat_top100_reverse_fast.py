import itertools, json, re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8'); m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if m: rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0])); draws=np.array([[int(x) for x in r[2:7]] for r in rows],dtype=np.int16); bonuses=np.array([int(r[7]) for r in rows],dtype=np.int16); T=len(draws)
combos=np.array(list(itertools.combinations(range(1,32),5)),dtype=np.int16); N=len(combos); combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
inc=np.zeros((N,32),dtype=np.uint8); inc[np.arange(N)[:,None],combos]=1
num_prefix=np.zeros((T+1,32),dtype=np.int32)
for t,row in enumerate(draws): num_prefix[t+1]=num_prefix[t]; num_prefix[t+1,row]+=1
def rnum(t,w): return num_prefix[t]-num_prefix[max(0,t-w)]
A_SHAPES={'1-2-2-0','2-2-1-0','2-1-2-0','1-1-3-0','1-3-1-0','1-2-1-1','1-1-2-1'}
def shape_arr(a):
 c=[0,0,0,0]
 for n in a: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
 return '-'.join(map(str,c))
combo_shapes=np.array([shape_arr(c) for c in combos],dtype=object); maskA=np.isin(combo_shapes,list(A_SHAPES)); draw_shapes=[shape_arr(a) for a in draws]
sums=combos.sum(1); sum_bin=np.clip((sums-15)//5,0,25).astype(np.int8); odd=(combos%2).sum(1).astype(np.int8); consec=(np.diff(combos,axis=1)==1).sum(1).astype(np.int8)
b0=(combos<=9).sum(1); b10=((combos>=10)&(combos<=19)).sum(1); b20=((combos>=20)&(combos<=29)).sum(1); b30=(combos>=30).sum(1); band_code=(b0*216+b10*36+b20*6+b30).astype(np.int16)
gapstd=np.std(np.diff(combos,axis=1),axis=1); gap_bin=np.digitize(gapstd,np.quantile(gapstd,[.2,.4,.6,.8])).astype(np.int8)
static={'sum':sum_bin,'odd':odd,'band':band_code,'consec':consec,'gap':gap_bin}; sizes={'sum':int(sum_bin.max()+1),'odd':6,'band':int(band_code.max()+1),'consec':5,'gap':5,'prev':6,'prev2':6,'pbonus':2,'hot':6}; FEATURES=('sum','odd','band','consec','gap','prev','prev2','pbonus','hot')
actual=[]
for t in range(T):
 wi=combo_index[tuple(map(int,draws[t]))]; c300=rnum(t,300); hot=set((np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15]) if t else set(); cur=set(map(int,draws[t]))
 actual.append({'sum':int(sum_bin[wi]),'odd':int(odd[wi]),'band':int(band_code[wi]),'consec':int(consec[wi]),'gap':int(gap_bin[wi]),'prev':len(cur&(set(map(int,draws[t-1])) if t>=1 else set())),'prev2':len(cur&(set(map(int,draws[t-2])) if t>=2 else set())),'pbonus':int(t>=1 and int(bonuses[t-1]) in cur),'hot':sum(int(n) in hot for n in draws[t])})
def dyn_cats(t):
 prev=draws[t-1] if t>=1 else np.array([],dtype=np.int16); prev2=draws[t-2] if t>=2 else np.array([],dtype=np.int16)
 d={'prev':inc[:,prev].sum(1).astype(np.int8) if len(prev) else np.zeros(N,np.int8),'prev2':inc[:,prev2].sum(1).astype(np.int8) if len(prev2) else np.zeros(N,np.int8),'pbonus':inc[:,int(bonuses[t-1])].astype(np.int8) if t>=1 else np.zeros(N,np.int8)}
 c300=rnum(t,300); hot=(np.lexsort((np.arange(1,32),-c300[1:]))+1)[:15] if t else np.array([],int); d['hot']=inc[:,hot].sum(1).astype(np.int8) if len(hot) else np.zeros(N,np.int8); return d
def astat_score(t,hist=500,alpha=75):
 dyn=dyn_cats(t); s=np.zeros(N,np.float32); lo=max(0,t-hist); hi=[u for u in range(lo,t) if draw_shapes[u] in A_SHAPES]; n=max(1,len(hi))
 for f in FEATURES:
  cats=static[f] if f in static else dyn[f]; cc=np.bincount(cats[maskA],minlength=sizes[f]).astype(float); p=cc/max(cc.sum(),1); wins=np.bincount([actual[u][f] for u in hi],minlength=sizes[f]).astype(float); q=(wins+alpha*p)/(n+alpha); w=np.zeros_like(p); nz=p>0; w[nz]=np.log(q[nz]/p[nz]); w=np.clip(w,-1.5,1.5); s+=w[cats].astype(np.float32)
 s[~maskA]=-1e9; return s
def topk(sc,k):
 idx=np.argpartition(-sc,k-1)[:k]; return idx[np.lexsort((idx,-sc[idx]))]
def ov(c,w): return len(set(c)&w)
def ranks(vals):
 order=np.argsort(-vals,kind='mergesort'); r=np.empty(len(vals),int); r[order]=np.arange(1,len(vals)+1); return r
pair_list=list(itertools.combinations(range(1,32),2)); pi={p:i for i,p in enumerate(pair_list)}; tri_list=list(itertools.combinations(range(1,32),3)); ti={p:i for i,p in enumerate(tri_list)}
POOLS=[8,10,12,15]; METHODS={'num':(1,0,0),'pair':(0,1,0),'triple':(0,0,1),'num_pair':(1,1,0),'num_pair_triple':(1,1,1)}
def block(a,b):
 recs=[]
 for rr in range(a,b+1):
  t=rr-1
  if draw_shapes[t] not in A_SHAPES: continue
  W=set(map(int,draws[t])); sc=astat_score(t); tc=combos[topk(sc,100)]; nrw=np.zeros(32,float); prw=np.zeros(465,float); trw=np.zeros(4495,float); nraw=np.zeros(32,int); praw=np.zeros(465,int); traw=np.zeros(4495,int)
  for rank,c in enumerate(tc,1):
   ww=1/np.sqrt(rank); L=list(map(int,c))
   for n in L: nrw[n]+=ww; nraw[n]+=1
   for p in itertools.combinations(L,2): prw[pi[p]]+=ww; praw[pi[p]]+=1
   for q in itertools.combinations(L,3): trw[ti[q]]+=ww; traw[ti[q]]+=1
  rn=ranks(nraw[1:]); rp=ranks(praw); rt=ranks(traw); wp=list(itertools.combinations(sorted(W),2)); wt=list(itertools.combinations(sorted(W),3)); rec={'num5':sum(rn[n-1]<=5 for n in W),'num10':sum(rn[n-1]<=10 for n in W),'num15':sum(rn[n-1]<=15 for n in W),'numrank':float(np.mean([rn[n-1] for n in W])),'pair10':sum(rp[pi[x]]<=10 for x in wp),'pair30':sum(rp[pi[x]]<=30 for x in wp),'pair50':sum(rp[pi[x]]<=50 for x in wp),'pairrank':float(np.mean([rp[pi[x]] for x in wp])),'tri10':sum(rt[ti[x]]<=10 for x in wt),'tri30':sum(rt[ti[x]]<=30 for x in wt),'tri50':sum(rt[ti[x]]<=50 for x in wt),'trirank':float(np.mean([rt[ti[x]] for x in wt])),'asm':{}}
  zn=(nrw-np.mean(nrw[1:]))/(np.std(nrw[1:])+1e-9); zp=(prw-prw.mean())/(prw.std()+1e-9); zt=(trw-trw.mean())/(trw.std()+1e-9); no=np.argsort(-nrw[1:],kind='mergesort')+1
  for m in POOLS:
   pool=list(map(int,no[:m])); cand=[tuple(c) for c in itertools.combinations(pool,5) if shape_arr(c) in A_SHAPES]; ph=int(W.issubset(set(pool)))
   for name,(x,y,z) in METHODS.items():
    vals=[]
    for c in cand:
     v=x*sum(zn[n] for n in c)
     if y: v+=y*sum(zp[pi[p]] for p in itertools.combinations(sorted(c),2))
     if z: v+=z*sum(zt[ti[q]] for q in itertools.combinations(sorted(c),3))
     vals.append(v)
    order=np.argsort(-np.asarray(vals),kind='mergesort')[:10] if vals else [] ; sel=[cand[i] for i in order]; best=max([ov(c,W) for c in sel],default=0); ex=int(any(set(c)==W for c in sel)); rec['asm'][f'm{m}_{name}']=(ph,ex,best)
  recs.append(rec)
 def av(k): return float(np.mean([r[k] for r in recs]))
 out={'A_draws':len(recs),'support':{'winner_num_top5_avg':av('num5'),'winner_num_top10_avg':av('num10'),'winner_num_top15_avg':av('num15'),'winner_num_mean_rank':av('numrank'),'winner_pair_top10_avg':av('pair10'),'winner_pair_top30_avg':av('pair30'),'winner_pair_top50_avg':av('pair50'),'winner_pair_mean_rank':av('pairrank'),'winner_tri_top10_avg':av('tri10'),'winner_tri_top30_avg':av('tri30'),'winner_tri_top50_avg':av('tri50'),'winner_tri_mean_rank':av('trirank')},'assembly':{}}
 for m in POOLS:
  for name in METHODS:
   key=f'm{m}_{name}'; aa=[r['asm'][key] for r in recs]; out['assembly'][key]={'pool_has_winner':sum(x[0] for x in aa),'exact5':sum(x[1] for x in aa),'best4plus':sum(x[2]>=4 for x in aa),'best3plus':sum(x[2]>=3 for x in aa),'avg_best':float(np.mean([x[2] for x in aa]))}
 return out
blocks={'development_800_999':block(800,999),'validation_1000_1199':block(1000,1199),'diagnostic_1200_1399':block(1200,1399)}
best=None; bt=None
for k,v in blocks['development_800_999']['assembly'].items():
 m=int(k.split('_')[0][1:]); tup=(v['exact5'],v['best4plus'],v['best3plus'],v['pool_has_winner'],-m)
 if bt is None or tup>bt: bt=tup; best=k
out={'protocol':{'A_layer_only':True,'top100':'A-Stat Top100','development':[800,999],'validation':[1000,1199],'diagnostic':[1200,1399],'excluded':[1400,1401]},'random_reference':{'num_top5':25/31,'num_top10':50/31,'num_top15':75/31,'pair_top10':100/465,'pair_top30':300/465,'pair_top50':500/465,'tri_top10':100/4495,'tri_top30':300/4495,'tri_top50':500/4495},'selected_on_development':best,'blocks':blocks}
p=ROOT/'data'/'miniloto-astat-top100-reverse-fast.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2)); print('WROTE',p)
