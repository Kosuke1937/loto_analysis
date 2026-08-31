#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
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
SUM_BUCKETS=[(120,129,2),(130,139,3),(140,149,3),(150,159,1),(160,999,1)]

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

def score_ticket(t,ns,pairc,sf):
    nscore=sum(ns[x] for x in t)
    pscore=sum(pairc[x,y] for x,y in itertools.combinations(t,2))
    f=sf.get(v4.band_tuple(np.array(t)),0.0)
    # mild preference toward centre of requested 120-149 main region, not a hard optimum
    center_pen=0.015*abs(sum(t)-137)
    return nscore+0.20*pscore+8.0*f-center_pen

def bucket_id(total):
    for i,(lo,hi,_) in enumerate(SUM_BUCKETS):
        if lo<=total<=hi:return i
    return None

def main():
    rows=p.fetch_history();draws=np.asarray([r[1] for r in rows],np.int16);t=len(draws)
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws2,bonus2,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws2,bonus2,npref)
    pairc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pairc);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    ns=num_support(agents,C);rank=rank_nums(ns);P=set(rank[:K]);S=set(rank[K:K+R]);sf=shape_freq(draws2,t,500);recent={v4.band_tuple(draws2[u]) for u in range(max(0,t-20),t)}
    cand=[]
    for comb in itertools.combinations(sorted(P|S),6):
        sc=sum(x in S for x in comb)
        if sc>3:continue
        sh=v4.band_tuple(np.array(comb));f=sf.get(sh,0.0);sm=sum(comb);bid=bucket_id(sm)
        if bid is None or f<A_THR or sh in recent:continue
        base=score_ticket(comb,ns,pairc,sf)-0.18*max(0,sc-1)
        cand.append((base,comb,sc,sh,f,bid))
    cand.sort(reverse=True)

    sel=[];tc=Counter();pcap=Counter();nc=Counter();bc=Counter();shc=Counter()
    def allowed(comb,sh,bid):
        quota=SUM_BUCKETS[bid][2]
        if bc[bid]>=quota:return False
        # prevent all 10 collapsing into one shape; max 4 per exact band shape
        if shc[sh]>=4:return False
        trs=list(itertools.combinations(comb,3));pas=list(itertools.combinations(comb,2))
        if any(tc[x]>=2 for x in trs) or any(pcap[x]>=3 for x in pas) or any(nc[x]>=5 for x in comb):return False
        return True
    def add(rec):
        score,comb,sc,sh,f,bid=rec;sel.append(rec);bc[bid]+=1;shc[sh]+=1
        for x in itertools.combinations(comb,3):tc[x]+=1
        for x in itertools.combinations(comb,2):pcap[x]+=1
        for x in comb:nc[x]+=1

    # fill each sum bucket in order, preferring main shell <=1 satellite
    for bid,(_,_,quota) in enumerate(SUM_BUCKETS):
        for phase in (1,3):
            for rec in cand:
                score,comb,sc,sh,f,b=rec
                if b!=bid or any(comb==x[1] for x in sel):continue
                if phase==1 and sc>1:continue
                if phase==3 and sc<=1:continue
                if not allowed(comb,sh,bid):continue
                add(rec)
                if bc[bid]>=quota:break
            if bc[bid]>=quota:break
    # fallback if constraints make fewer than 10
    if len(sel)<10:
        for rec in cand:
            score,comb,sc,sh,f,bid=rec
            if any(comb==x[1] for x in sel):continue
            if shc[sh]>=4:continue
            # only respect core overlap here; relax bucket quota
            trs=list(itertools.combinations(comb,3));pas=list(itertools.combinations(comb,2))
            if any(tc[x]>=2 for x in trs) or any(pcap[x]>=3 for x in pas) or any(nc[x]>=5 for x in comb):continue
            add(rec)
            if len(sel)>=10:break

    out={'target_draw':rows[-1][0]+1,'latest_draw':rows[-1][0],'latest_nums':list(rows[-1][1]),'primary32':sorted(P),'satellite10':sorted(S),'rules':{'A_freq_min':A_THR,'shape_absent_prev20':True,'sum_buckets':[list(x) for x in SUM_BUCKETS],'main_shell':'6+0/5+1','deep_rescue':'4+2/3+3','shape_cap_per_exact_band':4},'tickets':[{'nums':list(c),'sum':sum(c),'satellite_count':sc,'shape':list(sh),'shape_freq500':f,'score':score} for score,c,sc,sh,f,bid in sel]}
    path=OUT/'loto6_today_2133_candidates.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
