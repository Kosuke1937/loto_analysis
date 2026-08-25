#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast fixed Phase2 reverse test. No hyperparameter sweep.
Frozen rule: top 3 observable consensus 3cores, replace 4 of 10 tickets.
Dev/test split is reported only as stability check; parameters are not fitted.
"""
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ph1',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
DEV_END=1877; TOPK=3; REPLACE=4; SUPPORTN=500

def row(C,i):return tuple(map(int,C[int(i)]))
def rw(r):return 1/np.log2(r+2.0)
def support(agents,C):
 v3=Counter();w3=Counter();v4=Counter();w4=Counter()
 for name,idx in agents.items():
  a3={};a4={}
  for rank,i in enumerate(idx[:SUPPORTN],1):
   t=row(C,i);z=rw(rank)
   for c in itertools.combinations(t,3):a3[c]=max(a3.get(c,0),z)
   for c in itertools.combinations(t,4):a4[c]=max(a4.get(c,0),z)
  for c,z in a3.items():v3[c]+=1;w3[c]+=z
  for c,z in a4.items():v4[c]+=1;w4[c]+=z
 return v3,w3,v4,w4

def score_ticket(t,S):
 v3,w3,v4,w4=S
 a4=sorted(((v4[c],w4[c]) for c in itertools.combinations(t,4)),reverse=True)[:2]
 a3=sorted(((v3[c],w3[c]) for c in itertools.combinations(t,3)),reverse=True)[:2]
 return 4*a4[0][0]+1.2*a4[1][0]+1.8*a3[0][0]+.6*a3[1][0]+.5*a4[0][1]+.15*a4[1][1]+.2*a3[0][1]+.08*a3[1][1]

def generated(U,S,maxn=500):
 v3,w3,_,_=S; U=tuple(sorted(U)); cs=list(itertools.combinations(U,3));cs.sort(key=lambda c:(v3[c],w3[c]),reverse=True)
 pool={}
 for core in cs[:TOPK]:
  rest=[n for n in U if n not in core]
  for add in itertools.combinations(rest,3):
   t=tuple(sorted(core+add));pool[t]=score_ticket(t,S)
 return sorted(pool,key=pool.get,reverse=True)[:maxn],cs

def metr(win,T):
 W=set(map(int,win));U=set().union(*(set(t) for t in T));u=len(W&U);b=max(len(W&set(t)) for t in T)
 w3=set(itertools.combinations(tuple(sorted(W)),3));w4=set(itertools.combinations(tuple(sorted(W)),4))
 s3=set(c for t in T for c in itertools.combinations(t,3));s4=set(c for t in T for c in itertools.combinations(t,4))
 return {'union':u,'best':b,'assembly':b/u if u else 0,'core3':int(bool(w3&s3)),'core4':int(bool(w4&s4))}
def summ(R):
 return {'draws':len(R),'mean_union':float(np.mean([x['union'] for x in R])),'union5plus':sum(x['union']>=5 for x in R),'union6':sum(x['union']==6 for x in R),'mean_best':float(np.mean([x['best'] for x in R])),'mean_assembly':float(np.mean([x['assembly'] for x in R])),'core3_capture':sum(x['core3'] for x in R),'core4_capture':sum(x['core4'] for x in R),'d3':sum(x['best']>=3 for x in R),'d4':sum(x['best']>=4 for x in R),'d5':sum(x['best']>=5 for x in R),'d6':sum(x['best']>=6 for x in R)}
def agents_for(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc):
 ss={}
 for h in (200,500,800):
  W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
 pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
 A={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
 con=p.consensus_portfolio(A,{'committee':comm},C)
 return A,con

def main():
 rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)};C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
 all0=[];all1=[];dev0=[];dev1=[];test0=[];test1=[];rev=[]
 for draw in range(p.START_DRAW,p.END_DRAW+1):
  t=di[draw];A,con=agents_for(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc);orig=[row(C,i) for i in con];win=tuple(map(int,draws[t]));m0=metr(win,orig);S=support(A,C);G,all3=generated(set().union(*(set(x) for x in orig)),S)
  sel=orig[:10-REPLACE]
  for x in G:
   if x in sel:continue
   if any(len(set(x)&set(y))>=5 for y in sel):continue
   sel.append(x)
   if len(sel)==10:break
  for x in orig:
   if len(sel)>=10:break
   if x not in sel:sel.append(x)
  m1=metr(win,sel);all0.append(m0);all1.append(m1);(dev0 if draw<=DEV_END else test0).append(m0);(dev1 if draw<=DEV_END else test1).append(m1)
  if m0['union']==6:
   U=sorted(set().union(*(set(x) for x in orig)));v3,w3,v4,w4=S;u3=sorted(itertools.combinations(U,3),key=lambda c:(v3[c],w3[c]),reverse=True);u4=sorted(itertools.combinations(U,4),key=lambda c:(v4[c],w4[c]),reverse=True);W3=set(itertools.combinations(tuple(sorted(win)),3));W4=set(itertools.combinations(tuple(sorted(win)),4));r3=min([i+1 for i,c in enumerate(u3) if c in W3] or [999999]);r4=min([i+1 for i,c in enumerate(u4) if c in W4] or [999999]);ex=(G.index(tuple(sorted(win)))+1) if tuple(sorted(win)) in G else 999999;rev.append({'draw':draw,'r3':r3,'r4':r4,'exact_generated_rank':ex,'before_best':m0['best'],'after_best':m1['best']})
 out={'method':'fixed Phase2 fast; TOPK=3, replace=4; no tuning','range':f'{p.START_DRAW}-{p.END_DRAW}','development':{'range':f'{p.START_DRAW}-{DEV_END}','phase1':summ(dev0),'phase2':summ(dev1)},'holdout':{'range':f'{DEV_END+1}-{p.END_DRAW}','phase1':summ(test0),'phase2':summ(test1)},'all500':{'phase1':summ(all0),'phase2':summ(all1)},'reverse_union6':{'count':len(rev),'r3_le1':sum(x['r3']<=1 for x in rev),'r3_le3':sum(x['r3']<=3 for x in rev),'r3_le5':sum(x['r3']<=5 for x in rev),'r3_le10':sum(x['r3']<=10 for x in rev),'r4_le1':sum(x['r4']<=1 for x in rev),'r4_le3':sum(x['r4']<=3 for x in rev),'r4_le5':sum(x['r4']<=5 for x in rev),'r4_le10':sum(x['r4']<=10 for x in rev),'exact_top10':sum(x['exact_generated_rank']<=10 for x in rev),'exact_top50':sum(x['exact_generated_rank']<=50 for x in rev),'exact_top100':sum(x['exact_generated_rank']<=100 for x in rev),'exact_top500':sum(x['exact_generated_rank']<=500 for x in rev),'best_improved':sum(x['after_best']>x['before_best'] for x in rev),'best_worsened':sum(x['after_best']<x['before_best'] for x in rev)},'reverse_detail':rev,'caveat':'Fixed 60k candidate-sample diagnostic, not exact all-6,096,454 ranking. Reverse metrics are exploratory; holdout stability is the key check.'}
 (OUT/'loto6_consensus_phase2_fast_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='reverse_detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
