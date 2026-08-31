#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5+1 rescue diagnostic for reduced-pool Loto6.

Question: when a primary reduced pool contains exactly 5 of the 6 winning
numbers, can a small satellite band of next-ranked numbers recover the omitted
sixth without simply reverting to a large full pool?

Primary pool is built only from pre-draw multi-agent number support
(Stat200/500/800/Committee). Satellite numbers are the next R numbers in that
same pre-draw ranking. Winner is used only for retrospective evaluation.

This evaluates candidate-family coverage rather than ticket ranking:
  - all 6 numbers from Primary K, OR
  - exactly 5 from Primary K + 1 from Satellite R.
This keeps the combination family far smaller than using K+R as an unrestricted
pool.
"""
from __future__ import annotations
import importlib.util,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v2',ROOT/'research'/'loto6_reduced_pool_shape_diag_v2.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
p=v2.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
KS=(20,24,28,32)
RS=(2,4,6,8)
A_THR=.025

def ranking_from_support(s):
    nums=list(range(1,44)); nums.sort(key=lambda x:(s[x],-x),reverse=True); return nums

def fam_size(k,r): return math.comb(k,6)+math.comb(k,5)*r

def full_size(k,r): return math.comb(k+r,6)

def eval_one(win,rank,k,r):
    P=set(rank[:k]); S=set(rank[k:k+r]); W=set(map(int,win)); pin=len(W&P); sin=len(W&S)
    recovered=int(pin==6 or (pin==5 and sin>=1))
    return pin,recovered,int(pin==5 and sin>=1)

def summarize(rows):
    n=len(rows)
    return {'n':n,'primary_r6':sum(x[0]==6 for x in rows),'primary_r5_exact':sum(x[0]==5 for x in rows),'rescued_exact5':sum(x[2] for x in rows),'family_r6':sum(x[1] for x in rows),'family_r6_rate':sum(x[1] for x in rows)/n if n else 0.0}

def main():
    rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    rec={(k,r):{'dev':[],'hold':[],'joint_dev':[],'joint_hold':[]} for k in KS for r in RS}
    for draw in range(START,END+1):
        t=di[draw]
        ss={}
        for h in (200,500,800):
            W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
        pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
        agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
        sup=v2.num_support(agents,C); rank=ranking_from_support(sup); win=draws[t]
        sf=v2.rolling_shape_freq(draws,t,500); shape=v2.band_tuple(win); A=sf.get(shape,0)>=A_THR; sum120=int(np.sum(win))>=120
        recent={v2.band_tuple(draws[u]) for u in range(max(0,t-20),t)}; novel=shape not in recent; joint=A and sum120 and novel
        part='dev' if draw<=DEV_END else 'hold'; jpart='joint_dev' if draw<=DEV_END else 'joint_hold'
        for k in KS:
            for r in RS:
                e=eval_one(win,rank,k,r);rec[(k,r)][part].append(e)
                if joint:rec[(k,r)][jpart].append(e)
    out={'method':'primary reduced pool + next-rank satellite 5+1 rescue','range':f'{START}-{END}','Ks':KS,'Rs':RS,'results':{},'joint_results':{},'family_sizes':{}}
    for k in KS:
        for r in RS:
            key=f'K{k}_R{r}'; out['results'][key]={'dev':summarize(rec[(k,r)]['dev']),'hold':summarize(rec[(k,r)]['hold'])};out['joint_results'][key]={'dev':summarize(rec[(k,r)]['joint_dev']),'hold':summarize(rec[(k,r)]['joint_hold'])};out['family_sizes'][key]={'restricted_5plus1':fam_size(k,r),'unrestricted_KplusR':full_size(k,r),'fraction_of_all_6096454':fam_size(k,r)/6096454}
    out['caveat']='Satellite is only the next R numbers in the same pre-draw support ranking. This is a clean threshold-rescue test, not yet a learned sixth-number completion model.'
    path=OUT/'loto6_pool_5plus1_rescue_v3_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
