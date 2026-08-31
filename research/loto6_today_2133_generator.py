#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v4',ROOT/'research'/'loto6_pool_shell_rescue_v4.py')
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
p=v4.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
K,R=32,10
A_THR=.025

def z(x):
    x=np.asarray(x,float);s=x.std();return (x-x.mean())/(s+1e-9)

def shape_freq(draws,t,w=500):
    c=Counter(v4.band_tuple(draws[u]) for u in range(max(0,t-w),t));n=max(1,min(w,t));return {k:v/n for k,v in c.items()}

def num_support(agents,C,topn=500):
    s=np.zeros(44,float)
    for idx in agents.values():
        for rank,i in enumerate(idx[:topn],1):
            w=1/np.log2(rank+2)
            for x in C[int(i)]:s[int(x)]+=w
    return s

def rank_nums(s):return sorted(range(1,44),key=lambda x:(s[x],-x),reverse=True)

def score_ticket(t,ns,pc,sf):
    a=np.array(t); nscore=sum(ns[x] for x in t)
    pscore=sum(pc[x,y] for x,y in itertools.combinations(t,2))
    sh=v4.band_tuple(a);f=sf.get(sh,0.0)
    return nscore+0.20*pscore+8.0*f

def main():
    rows=p.fetch_history();draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16);t=len(draws)
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws2,bonus2,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws2,bonus2,npref)
    pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    ns=num_support(agents,C);rank=rank_nums(ns);P=set(rank[:K]);S=set(rank[K:K+R]);sf=shape_freq(draws2,t,500);recent={v4.band_tuple(draws2[u]) for u in range(max(0,t-20),t)}
    # robust main family: 6+0 or 5+1. deeper rescue tickets added separately.
    cand=[]
    universe=sorted(P|S)
    for comb in itertools.combinations(universe,6):
        sc=sum(x in S for x in comb)
        if sc>3:continue
        sh=v4.band_tuple(np.array(comb)); f=sf.get(sh,0.0)
        if f<A_THR or sum(comb)<120 or sh in recent:continue
        base=score_ticket(comb,ns,pc,sf)
        # prefer M1; allow M2/M3 as rescue with small penalty
        base-=0.18*max(0,sc-1)
        cand.append((base,comb,sc,sh,f))
    cand.sort(reverse=True)
    sel=[];tc=Counter();pcap=Counter();nc=Counter();rescue=0
    # first 7 from satellite<=1, then up to 3 deeper rescue
    for phase in (1,3):
        for score,comb,sc,sh,f in cand:
            if comb in [x[1] for x in sel]:continue
            if phase==1 and sc>1:continue
            if phase==3 and sc<=1:continue
            trs=list(itertools.combinations(comb,3));pas=list(itertools.combinations(comb,2))
            if any(tc[x]>=2 for x in trs) or any(pcap[x]>=3 for x in pas) or any(nc[x]>=5 for x in comb):continue
            sel.append((score,comb,sc,sh,f))
            for x in trs:tc[x]+=1
            for x in pas:pcap[x]+=1
            for x in comb:nc[x]+=1
            if sc>1:rescue+=1
            if (phase==1 and len(sel)>=7) or len(sel)>=10:break
        if len(sel)>=10:break
    out={'target_draw':rows[-1][0]+1,'latest_draw':rows[-1][0],'latest_nums':list(rows[-1][1]),'primary32':sorted(P),'satellite10':sorted(S),'rules':{'A_freq_min':A_THR,'sum_min':120,'shape_absent_prev20':True,'main_shell':'6+0/5+1','deep_rescue':'4+2/3+3 up to 3 tickets'},'tickets':[{'nums':list(c),'sum':sum(c),'satellite_count':sc,'shape':list(sh),'shape_freq500':f,'score':score} for score,c,sc,sh,f in sel]}
    path=OUT/'loto6_today_2133_candidates.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
