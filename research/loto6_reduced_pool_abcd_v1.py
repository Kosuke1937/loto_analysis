#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reduced-pool experiment inspired by Mandel-style candidate reduction.

Goal: first reduce 43 numbers to a compact candidate pool using only pre-draw
multi-agent support, then assemble 10 tickets under a rolling ABCD-shape rule.

ABCD here is based on rolling 500-draw empirical frequency of 5-band composition
(1-9/10-19/20-29/30-39/40-43). User-requested A threshold for this experiment:
frequency >= 2.5%.

No winner information is used for pool construction or ticket ranking.
"""
from __future__ import annotations
import importlib.util,itertools,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
POOL_KS=(12,15,18,20)
A_THR=0.025

def band_tuple(a):
    a=np.asarray(a)
    return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))

def rolling_shape_freq(draws,t,w=500):
    lo=max(0,t-w); c=Counter(band_tuple(draws[i]) for i in range(lo,t)); n=max(1,t-lo)
    return {k:v/n for k,v in c.items()}

def num_support(agents,C,topn=500):
    s=np.zeros(44,float)
    for name,idx in agents.items():
        for rank,i in enumerate(idx[:topn],1):
            w=1.0/np.log2(rank+2.0)
            for x in map(int,C[int(i)]): s[x]+=w
    return s

def pool_from_support(s,k):
    nums=list(range(1,44)); nums.sort(key=lambda x:(s[x],-x),reverse=True); return tuple(sorted(nums[:k]))

def candidates_in_pool(C,pool,comm,shape_freq,hard_a=True):
    P=set(pool); idx=[]
    for i,row in enumerate(C):
        t=tuple(map(int,row))
        if not set(t)<=P: continue
        f=shape_freq.get(band_tuple(t),0.0)
        if hard_a and f < A_THR: continue
        idx.append((i,float(comm[i]),f))
    idx.sort(key=lambda z:(z[1],z[2]),reverse=True)
    return idx

def assemble10(items,C):
    sel=[]; tc=Counter(); pc=Counter(); nc=Counter()
    for i,score,f in items:
        row=tuple(map(int,C[i])); trs=list(itertools.combinations(row,3)); pas=list(itertools.combinations(row,2))
        if any(tc[x]>=2 for x in trs): continue
        if any(pc[x]>=3 for x in pas): continue
        if any(nc[x]>=5 for x in row): continue
        sel.append(i)
        for x in trs: tc[x]+=1
        for x in pas: pc[x]+=1
        for x in row: nc[x]+=1
        if len(sel)==10: break
    if len(sel)<10:
        used=set(sel)
        for i,_,_ in items:
            if i not in used: sel.append(i);used.add(i)
            if len(sel)==10: break
    return sel[:10]

def eval_draw(win,pool,sel,C):
    W=set(map(int,win)); pr=len(W & set(pool))
    if sel:
        best=max(len(W & set(map(int,C[i]))) for i in sel)
        U=set().union(*(set(map(int,C[i])) for i in sel)); ur=len(W&U)
    else: best=0;ur=0
    return {'pool_recall':pr,'best':best,'union':ur,'d3':int(best>=3),'d4':int(best>=4),'d5':int(best>=5),'d6':int(best>=6)}

def summary(rows):
    if not rows:return {}
    return {'n':len(rows),'pool_recall6':sum(r['pool_recall']==6 for r in rows),'pool_recall5plus':sum(r['pool_recall']>=5 for r in rows),'mean_pool_recall':float(np.mean([r['pool_recall'] for r in rows])),'mean_union':float(np.mean([r['union'] for r in rows])),'d3':sum(r['d3'] for r in rows),'d4':sum(r['d4'] for r in rows),'d5':sum(r['d5'] for r in rows),'d6':sum(r['d6'] for r in rows),'mean_best':float(np.mean([r['best'] for r in rows]))}

def main():
    rows=p.fetch_history(); di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample(); st,inc,q=p.build_static(C); draws,bonus,npref,ppref,actual=p.hist_actual(rows,q); sizes,priors=p.prepare_priors(st)
    out={'method':'reduced-pool first, then ABCD A>=2.5% hard-shape assembly','range':f'{START}-{END}','pool_ks':POOL_KS,'A_threshold':A_THR,'development':{},'holdout':{},'detail':{}}
    rec={k:{'dev':[],'hold':[]} for k in POOL_KS}
    for draw in range(START,END+1):
        t=di[draw]
        ss={}
        for h in (200,500,800):
            W=p.weights(t,h,actual,sizes,priors); ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
        pc=ppref[t]-ppref[max(0,t-300)]; c5,c4=p.cores(C,pc); z=lambda x:(x-x.mean())/(x.std()+1e-9); comm=z(ss[500])+.20*z(c5)+.15*z(c4)
        agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
        sup=num_support(agents,C); sf=rolling_shape_freq(draws,t,500); win=draws[t]
        for k in POOL_KS:
            pool=pool_from_support(sup,k); items=candidates_in_pool(C,pool,comm,sf,True); sel=assemble10(items,C); r=eval_draw(win,pool,sel,C); r.update({'draw':draw,'pool':pool,'selected_n':len(sel)})
            rec[k]['dev' if draw<=DEV_END else 'hold'].append(r)
    for k in POOL_KS:
        out['development'][str(k)]=summary(rec[k]['dev']); out['holdout'][str(k)]=summary(rec[k]['hold'])
    # choose K on dev lexicographically: d4,d3,pool recall6,mean_best
    def obj(k):
        s=out['development'][str(k)]; return (s['d4'],s['d3'],s['pool_recall6'],s['mean_best'])
    best=max(POOL_KS,key=obj); out['selected_k_on_dev']=best; out['selected_holdout']=out['holdout'][str(best)]
    out['caveat']='Approximate fixed 60k candidate universe; tests whether number reduction before assembly helps. A>=2.5% is user-specified and treated as a hard scenario filter only in this experiment.'
    (OUT/'loto6_reduced_pool_abcd_v1_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
