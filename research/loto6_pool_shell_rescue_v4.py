#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restricted shell rescue coverage study for Loto6.

Extends 5+1 rescue by allowing up to M satellite numbers while preserving a
small primary pool. Evaluates coverage only; winner is never used in ranking.

Families:
  M=0: 6+0
  M=1: 6+0 or 5+1
  M=2: add 4+2
  M=3: add 3+3

Also evaluates user scenario A>=2.5%, sum>=120, shape absent previous20.
"""
from __future__ import annotations
import importlib.util,itertools,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v3',ROOT/'research'/'loto6_pool_5plus1_rescue_v3.py')
v3=importlib.util.module_from_spec(spec);spec.loader.exec_module(v3)
p=v3.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
KS=(24,28,30,32)
RS=(6,8,10,11)
MS=(0,1,2,3)
A_THR=.025

def band_tuple(a):
    a=np.asarray(a)
    return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))

def shape_freq(draws,t,w=500):
    lo=max(0,t-w); cnt={}
    for u in range(lo,t):
        sh=band_tuple(draws[u]);cnt[sh]=cnt.get(sh,0)+1
    n=max(1,t-lo);return {k:v/n for k,v in cnt.items()}

def shell_size(k,r,m):
    return sum(math.comb(k,6-j)*math.comb(r,j) for j in range(0,min(m,6,r)+1) if 6-j<=k)

def eval_cover(win,rank,k,r,m):
    P=set(rank[:k]);S=set(rank[k:k+r]);W=set(map(int,win));pcount=len(W&P);scount=len(W&S)
    outside=6-pcount-scount
    ok=(outside==0 and scount<=m and pcount+scount==6)
    return pcount,scount,int(ok)

def summ(rows):
    n=len(rows);return {'n':n,'coverage':sum(x[2] for x in rows),'rate':sum(x[2] for x in rows)/n if n else 0.0,'mean_primary':float(np.mean([x[0] for x in rows])) if n else 0.0,'p6':sum(x[0]==6 for x in rows),'p5':sum(x[0]==5 for x in rows),'p4':sum(x[0]==4 for x in rows)}

def main():
    rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    rec={(k,r,m):{'dev':[],'hold':[],'jdev':[],'jhold':[]} for k in KS for r in RS if k+r<=43 for m in MS}
    for draw in range(START,END+1):
        t=di[draw];ss={}
        for h in (200,500,800):
            W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
        pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
        agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
        sup=v3.v2.num_support(agents,C);rank=v3.ranking_from_support(sup);win=draws[t]
        sf=shape_freq(draws,t,500);sh=band_tuple(win);A=sf.get(sh,0)>=A_THR;sum120=int(np.sum(win))>=120;recent={band_tuple(draws[u]) for u in range(max(0,t-20),t)};joint=A and sum120 and sh not in recent
        part='dev' if draw<=DEV_END else 'hold';jpart='jdev' if draw<=DEV_END else 'jhold'
        for k in KS:
            for r in RS:
                if k+r>43:continue
                for m in MS:
                    e=eval_cover(win,rank,k,r,m);rec[(k,r,m)][part].append(e)
                    if joint:rec[(k,r,m)][jpart].append(e)
    out={'method':'restricted primary/satellite shell rescue','range':f'{START}-{END}','results':{},'joint':{},'sizes':{}}
    for key,d in rec.items():
        k,r,m=key;nm=f'K{k}_R{r}_M{m}';out['results'][nm]={'dev':summ(d['dev']),'hold':summ(d['hold'])};out['joint'][nm]={'dev':summ(d['jdev']),'hold':summ(d['jhold'])};sz=shell_size(k,r,m);out['sizes'][nm]={'candidates':sz,'fraction_all':sz/6096454}
    path=OUT/'loto6_pool_shell_rescue_v4_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
