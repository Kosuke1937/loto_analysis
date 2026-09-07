#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
CAL_START=1628; EVAL_START=1828; END=2134

def local_rows():
 out=[]
 for k in range(1,13):
  s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8');payload=s.split('push(',1)[1].rsplit(');',1)[0]
  for r in json.loads(payload):out.append((int(r[0]),tuple(sorted(int(x) for x in r[2:8])),int(r[8])))
 out.sort(key=lambda x:x[0]);return out

def band(a):
 a=np.asarray(a);return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))
def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def layer(f):return 'A' if f>=.02 else ('B' if f>=.01 else ('C' if f>=.005 else 'D'))

def direct_static(X,q):
 X=np.asarray(X,np.int16);s=X.sum(1);sb=np.clip((s-21)//5,0,50).astype(np.int16);odd=(X%2).sum(1).astype(np.int8);con=(np.diff(X,axis=1)==1).sum(1).astype(np.int8);bc=np.array([bcode(band(r)) for r in X],np.int32);gap=np.digitize(np.std(np.diff(X,axis=1),axis=1),q).astype(np.int8);inc=np.zeros((len(X),44),np.uint8);rr=np.arange(len(X))
 for j in range(6):inc[rr,X[:,j]]=1
 return {'sum':sb,'odd':odd,'band':bc,'consec':con,'gap':gap},inc

def wscore(t,win,W,q,draws,bonus,npref,pairc,rs,r5,r4):
 X=np.asarray([win],np.int16);st,inc=direct_static(X,q);s=p.stat_score(t,500,W,st,inc,draws,bonus,npref);c5,c4=p.cores(X,pairc)
 return float((s[0]-rs.mean())/(rs.std()+1e-9)+.20*(c5[0]-r5.mean())/(r5.std()+1e-9)+.15*(c4[0]-r4.mean())/(r4.std()+1e-9))

def select(order,C,layers,shapes,n=10):
 quota={'A':3,'B':3,'C':2,'D':2};lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter();sel=[]
 for strict in (True,False):
  for i in order:
   i=int(i)
   if i in sel:continue
   row=tuple(map(int,C[i]));ly=layers[i];sh=shapes[i]
   if lc[ly]>=quota[ly] or sc[sh]>=2:continue
   pa=list(itertools.combinations(row,2));tr=list(itertools.combinations(row,3))
   if strict and (any(nc[x]>=4 for x in row) or any(pc[x]>=2 for x in pa) or any(tc[x]>=1 for x in tr)):continue
   sel.append(i);lc[ly]+=1;sc[sh]+=1
   for x in row:nc[x]+=1
   for x in pa:pc[x]+=1
   for x in tr:tc[x]+=1
   if len(sel)>=n:return sel
 return sel

def metric(win,sel,C):
 W=set(map(int,win));best=max([len(W&set(map(int,C[i]))) for i in sel] or [0]);U=set().union(*(set(map(int,C[i])) for i in sel)) if sel else set()
 return {'best':best,'union':len(W&U),'d3':int(best>=3),'d4':int(best>=4),'d5':int(best>=5),'d6':int(best>=6)}

def summ(rr):
 return {'n':len(rr),'mean_best':float(np.mean([r['best'] for r in rr])) if rr else None,'mean_union':float(np.mean([r['union'] for r in rr])) if rr else None,'d3':sum(r['d3'] for r in rr),'d4':sum(r['d4'] for r in rr),'d5':sum(r['d5'] for r in rr),'d6':sum(r['d6'] for r in rr)}

def main():
 rows=local_rows();rows=[r for r in rows if r[0]<=END];di={d:i for i,(d,_,_) in enumerate(rows)};C=p.fixed_sample();st,inc,q=p.build_static(C);sums=C.sum(1);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
 past=[];past_layer=defaultdict(list);R={'top':[],'match':[],'layer_match':[]};details=[]
 for draw in range(CAL_START,END+1):
  t=di[draw];W=p.weights(t,500,actual,sizes,priors);rs=p.stat_score(t,500,W,st,inc,draws,bonus,npref);pairc=ppref[t]-ppref[max(0,t-300)];r5,r4=p.cores(C,pairc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(rs)+.20*z(r5)+.15*z(r4)
  win=tuple(map(int,draws[t]));wsc=wscore(t,win,W,q,draws,bonus,npref,pairc,rs,r5,r4);wsh=band(win);prev=set(map(int,draws[t-1]));prevsum=int(draws[t-1].sum());c20=Counter(band(r) for r in draws[t-20:t]);c2150=Counter(band(r) for r in draws[t-50:t-20]);c50=Counter(band(r) for r in draws[t-50:t]);hall=Counter(band(r) for r in draws[:t]);wly=layer(hall[wsh]/t);wqual=(c20[wsh]==0 and (c2150[wsh]==1 or c50[wsh]==0) and len(set(win)&prev)<=1 and int(bonus[t-1]) not in win and sum(win)>prevsum)
  # candidate masks using the same structural idea; ranking layer only, no Core/Satellite compression here
  allowed=[];layermap=np.empty(16807,dtype='<U1')
  for b0 in range(7):
   for b1 in range(7-b0):
    for b2 in range(7-b0-b1):
     for b3 in range(7-b0-b1-b2):
      b4=6-b0-b1-b2-b3;sh=(b0,b1,b2,b3,b4);code=bcode(sh);freq=hall[sh]/t;layermap[code]=layer(freq)
      if c20[sh]==0 and (c2150[sh]==1 or c50[sh]==0):allowed.append(code)
  prev_idx=list(prev);po=inc[:,prev_idx].sum(1);valid=np.isin(st['band'],np.asarray(allowed,np.int32))&(po<=1)&(inc[:,int(bonus[t-1])]==0)&(sums>prevsum)
  ids=np.where(valid)[0];layers=layermap[st['band']];shapes=st['band']
  if draw>=EVAL_START and len(ids)>=10 and len(past)>=20:
   toporder=ids[np.argsort(comm[ids])[::-1]]
   med=float(np.median(past));matchorder=ids[np.argsort(np.abs(comm[ids]-med))]
   # layer-specific historical target with fallback to overall median
   targets={ly:(float(np.median(past_layer[ly])) if len(past_layer[ly])>=8 else med) for ly in 'ABCD'}
   dist=np.array([abs(comm[i]-targets[layers[i]]) for i in ids]);layerorder=ids[np.argsort(dist)]
   sels={'top':select(toporder,C,layers,shapes),'match':select(matchorder,C,layers,shapes),'layer_match':select(layerorder,C,layers,shapes)}
   rec={'draw':draw,'winner_qualifies':bool(wqual),'winner_score':wsc,'rolling_target_median':med,'valid_sample_count':len(ids)}
   for k,sel in sels.items():
    m=metric(win,sel,C);R[k].append(m);rec[k]=m
   details.append(rec)
  if wqual:
   past.append(wsc);past_layer[wly].append(wsc)
  if draw%25==0:print(draw,'qual',wqual,'past',len(past),flush=True)
 out={'definition':'Ranking-layer backtest on deterministic 60k sample. Candidate structural filter: temporal shape rule, previous overlap<=1, previous bonus absent, candidate sum > previous draw sum. Both methods use identical A3/B3/C2/D2, shape<=2 and diversity caps. top maximizes Committee; match minimizes distance to rolling median of prior qualifying winner scores; layer_match uses prior layer-specific median when >=8 samples. No target-draw leakage. Core/Satellite compression is intentionally excluded to isolate the Committee ranking question.','calibration_start':CAL_START,'evaluation_start':EVAL_START,'end':END,'summaries':{k:summ(v) for k,v in R.items()},'details':details}
 (OUT/'loto6_committee_top_vs_scorematch_backtest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'definition':out['definition'],'summaries':out['summaries']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
