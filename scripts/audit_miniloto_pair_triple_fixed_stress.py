import itertools,json,re,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8');m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if m: rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]));draws=[tuple(int(x) for x in r[2:7]) for r in rows]
PAIR=list(itertools.combinations(range(1,32),2));TRI=list(itertools.combinations(range(1,32),3));pidx={p:i for i,p in enumerate(PAIR)};tidx={q:i for i,q in enumerate(TRI)}
T=len(draws);P=len(PAIR);Q=len(TRI)
pp=np.zeros((T+1,P),np.int16);tp=np.zeros((T+1,Q),np.int16)
for i,d in enumerate(draws):
 pp[i+1]=pp[i];tp[i+1]=tp[i]
 for p in itertools.combinations(d,2):pp[i+1,pidx[tuple(sorted(p))]]+=1
 for q in itertools.combinations(d,3):tp[i+1,tidx[tuple(sorted(q))]]+=1

def score(t,pair=True):
 if pair:return (pp[t]-pp[max(0,t-300)]).astype(float)
 return (tp[t]-tp[max(0,t-100)]).astype(float)
def hyper_var(U,K,n=10):
 p=K/U;return n*p*(1-p)*(U-n)/(U-1)
def eval_block(a,b):
 ph=[];th=[];rows=[]
 for rr in range(a,b+1):
  t=rr-1;w=draws[t];wp={pidx[tuple(sorted(x))] for x in itertools.combinations(w,2)};wt={tidx[tuple(sorted(x))] for x in itertools.combinations(w,3)}
  po=np.argsort(-score(t,True),kind='stable')[:50];to=np.argsort(-score(t,False),kind='stable')[:300]
  p=len(wp&set(map(int,po)));q=len(wt&set(map(int,to)));ph.append(p);th.append(q);rows.append({'draw':rr,'pair_hits':p,'triple_hits':q})
 def s(x,U,K):
  x=np.asarray(x,float);rand=10*K/U;tot=float(x.sum());er=rand*len(x);sd=math.sqrt(hyper_var(U,K)*len(x));z=(tot-er)/sd if sd else 0
  return {'n':len(x),'mean_hits':float(x.mean()),'random_mean':rand,'enrichment':float(x.mean()/rand),'total_hits':int(tot),'expected_total':er,'z_vs_random_approx':z,'draws_ge1':int((x>=1).sum()),'draws_ge2':int((x>=2).sum()),'draws_ge3':int((x>=3).sum()),'max_hits':int(x.max())}
 return {'range':[a,b],'pair_raw300_top50':s(ph,P,50),'triple_raw100_top300':s(th,Q,300),'rows':rows}
blocks=[]
for a,b in [(600,799),(800,999),(1000,1199),(1200,1399)]:blocks.append(eval_block(a,b))
if T>=1402:blocks.append(eval_block(1400,1402))
out={'definition':'Fixed stress audit chosen after the 1000-1399 exploratory audit: pair_raw300 Top50 and triple_raw100 Top300. No winner information is used in scoring. 600-999 was not used in choosing these signals and is the main historical stress check.','signals':{'pair':'past 300 draw raw pair count, Top50 of 465','triple':'past 100 draw raw triple count, Top300 of 4495'},'blocks':blocks,'warning':'Because the signal definitions were noticed after examining 1000-1399, even 600-999 is a post-selection stress test rather than a pristine prospective validation. Approximate z scores are descriptive and do not correct for model search/multiple comparisons.'}
(ROOT/'data'/'miniloto-pair-triple-fixed-stress.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
