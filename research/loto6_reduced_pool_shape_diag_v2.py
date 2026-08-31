#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wider reduced-pool + shape-condition diagnostic.
Tests candidate-pool sizes and the user's structural hypothesis:
A-layer rolling 500 frequency >=2.5%, sum>=120, and band shape unseen in prior20.
Winner is used only for retrospective evaluation.
"""
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
POOL_KS=(20,24,28,32,36)
A_THR=.025

def band(a):
 a=np.asarray(a);return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))
def shape_freq(draws,t,w=500):
 lo=max(0,t-w);c=Counter(band(draws[i]) for i in range(lo,t));n=max(1,t-lo);return {k:v/n for k,v in c.items()}
def num_support(agents,C,topn=500):
 s=np.zeros(44,float)
 for idx in agents.values():
  for rank,i in enumerate(idx[:topn],1):
   w=1/np.log2(rank+2.0)
   for x in map(int,C[int(i)]):s[x]+=w
 return s
def pool(s,k):
 z=list(range(1,44));z.sort(key=lambda x:(s[x],-x),reverse=True);return set(z[:k])
def summarize(rs):
 return {'n':len(rs),'r6':sum(x==6 for x in rs),'r5plus':sum(x>=5 for x in rs),'mean':float(np.mean(rs))}

def main():
 rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)};C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
 rec={k:{'dev':[],'hold':[]} for k in POOL_KS};conds={'dev':Counter(),'hold':Counter()};joint_pool={k:{'dev':[],'hold':[]} for k in POOL_KS}
 for draw in range(START,END+1):
  t=di[draw];ss={}
  for h in (200,500,800):
   W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
  pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
  agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)};sup=num_support(agents,C)
  win=set(map(int,draws[t]));sh=band(draws[t]);sf=shape_freq(draws,t,500).get(sh,0.0);recent={band(draws[i]) for i in range(max(0,t-20),t)}
  A=sf>=A_THR;S=int(np.sum(draws[t]))>=120;N=sh not in recent;split='dev' if draw<=DEV_END else 'hold'
  conds[split]['A']+=A;conds[split]['sum120']+=S;conds[split]['novel20']+=N;conds[split]['A_sum120']+=(A and S);conds[split]['A_novel20']+=(A and N);conds[split]['joint']+=(A and S and N)
  for k in POOL_KS:
   P=pool(sup,k);r=len(win&P);rec[k][split].append(r)
   if A and S and N:joint_pool[k][split].append(r)
 out={'method':'wider reduced-pool recall + retrospective structural hypothesis diagnostic','range':f'{START}-{END}','A_threshold':A_THR,'conditions':{s:dict(c) for s,c in conds.items()},'pool':{},'joint_condition_pool':{},'caveat':'Pool uses pre-draw multi-agent support. A/sum/novel20 statistics are retrospective checks of the actual winner shape; they are not assumed predictive until validated.'}
 for k in POOL_KS:
  out['pool'][str(k)]={'dev':summarize(rec[k]['dev']),'hold':summarize(rec[k]['hold'])};out['joint_condition_pool'][str(k)]={'dev':summarize(joint_pool[k]['dev']) if joint_pool[k]['dev'] else {},'hold':summarize(joint_pool[k]['hold']) if joint_pool[k]['hold'] else {}}
 (OUT/'loto6_reduced_pool_shape_diag_v2_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
