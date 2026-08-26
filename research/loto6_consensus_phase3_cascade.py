#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase3: winner-preserving 3core -> 4th -> 5th -> 6th cascade diagnostic.

No target winner is used to rank candidates. Winner is used only after each stage
for recall evaluation. Focus is Phase1 Consensus rounds whose 10-ticket union
contains all 6 winning numbers, because the purpose is to isolate the assembly
bottleneck after number recall succeeds.

Fixed rules (no tuning):
  - observable support from Stat200/500/800/Committee top500
  - keep top min(2000, all 3cores in the Phase1 union)
  - expand to unique 4sets inside the Phase1 union; keep top5000
  - expand to unique 5sets; keep top5000
  - expand to unique 6sets; keep top5000
  - scoring rewards direct 4core support plus supported constituent subcores
Split is reported for stability only: dev 1628-1877, holdout 1878-2127.
"""
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ph2',ROOT/'research'/'loto6_consensus_phase2_fast.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
p=q.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
DEV_END=1877; K3=2000; K4=5000; K5=5000; K6=5000

def rank_weight(rank): return 1.0/np.log2(rank+2.0)

def support_scores(A,C):
    v3,w3,v4,w4=q.support(A,C)
    def s3(c): return 2.0*v3[c]+0.6*w3[c]
    def s4(c):
        subs=list(itertools.combinations(c,3))
        a=sorted((s3(x) for x in subs),reverse=True)
        return 3.0*v4[c]+0.8*w4[c]+0.8*a[0]+0.25*a[1]
    return v3,w3,v4,w4,s3,s4

def topn_dict(d,k):
    return sorted(d,key=d.get,reverse=True)[:min(k,len(d))]

def cascade(U,A,C):
    U=tuple(sorted(U));v3,w3,v4,w4,s3,s4=support_scores(A,C)
    all3={c:s3(c) for c in itertools.combinations(U,3)}
    t3=topn_dict(all3,K3); set3=set(t3)

    d4={}
    for c3 in t3:
        for x in U:
            if x in c3: continue
            c4=tuple(sorted(c3+(x,)))
            d4[c4]=max(d4.get(c4,-1e99),s4(c4))
    t4=topn_dict(d4,K4)

    d5={}
    for c4 in t4:
        for x in U:
            if x in c4: continue
            c5=tuple(sorted(c4+(x,)))
            subs4=list(itertools.combinations(c5,4)); subs3=list(itertools.combinations(c5,3))
            sc=max(s4(z) for z in subs4)+0.35*np.mean([s4(z) for z in subs4])+0.15*max(s3(z) for z in subs3)
            d5[c5]=max(d5.get(c5,-1e99),float(sc))
    t5=topn_dict(d5,K5)

    d6={}
    for c5 in t5:
        for x in U:
            if x in c5: continue
            c6=tuple(sorted(c5+(x,)))
            subs4=list(itertools.combinations(c6,4)); subs3=list(itertools.combinations(c6,3))
            sc=max(d5.get(z,-1e99) for z in itertools.combinations(c6,5))
            if sc < -1e90: sc=0.0
            sc += 0.25*max(s4(z) for z in subs4)+0.10*np.mean([s4(z) for z in subs4])+0.05*max(s3(z) for z in subs3)
            d6[c6]=max(d6.get(c6,-1e99),float(sc))
    t6=topn_dict(d6,K6)
    return t3,t4,t5,t6

def best_rank(stage,winner,k):
    W=set(winner)
    if k==3: targets=set(itertools.combinations(tuple(sorted(W)),3))
    elif k==4: targets=set(itertools.combinations(tuple(sorted(W)),4))
    elif k==5: targets=set(itertools.combinations(tuple(sorted(W)),5))
    else: targets={tuple(sorted(W))}
    for i,c in enumerate(stage,1):
        if c in targets:return i
    return 999999

def recall(rows,key,cuts): return {f'le{c}':sum(r[key]<=c for r in rows) for c in cuts}

def summarize(rows):
    if not rows:return {}
    return {
      'n':len(rows),
      'r3':recall(rows,'r3',[50,100,250,500,1000,1500,2000]),
      'r4':recall(rows,'r4',[100,500,1000,3000,5000]),
      'r5':recall(rows,'r5',[100,500,1000,3000,5000]),
      'r6':recall(rows,'r6',[10,50,100,500,1000,3000,5000]),
      'median_r3':float(np.median([r['r3'] for r in rows])),
      'median_r4':float(np.median([r['r4'] for r in rows if r['r4']<999999])) if any(r['r4']<999999 for r in rows) else None,
      'median_r5':float(np.median([r['r5'] for r in rows if r['r5']<999999])) if any(r['r5']<999999 for r in rows) else None,
      'median_r6':float(np.median([r['r6'] for r in rows if r['r6']<999999])) if any(r['r6']<999999 for r in rows) else None,
    }

def main():
    rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,gq=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,gq);sizes,priors=p.prepare_priors(st)
    dev=[];test=[];detail=[]
    for draw in range(p.START_DRAW,p.END_DRAW+1):
        t=di.get(draw)
        if t is None:continue
        A,con=q.agents_for(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc)
        orig=[q.row(C,i) for i in con];win=tuple(map(int,draws[t]));m=q.metr(win,orig)
        if m['union']!=6:continue
        U=set().union(*(set(x) for x in orig));t3,t4,t5,t6=cascade(U,A,C)
        rec={'draw':draw,'union_size':len(U),'r3':best_rank(t3,win,3),'r4':best_rank(t4,win,4),'r5':best_rank(t5,win,5),'r6':best_rank(t6,win,6),'phase1_best':m['best']}
        detail.append(rec);(dev if draw<=DEV_END else test).append(rec)
    out={'method':'fixed Phase3 winner-preserving cascade on Phase1 union6 rounds','range':f'{p.START_DRAW}-{p.END_DRAW}','split':{'dev':f'{p.START_DRAW}-{DEV_END}','holdout':f'{DEV_END+1}-{p.END_DRAW}'},'fixed_caps':{'K3':K3,'K4':K4,'K5':K5,'K6':K6},'development':summarize(dev),'holdout':summarize(test),'all':summarize(detail),'detail':detail,'caveat':'Oracle-facing diagnostic conditioned on Phase1 union6. It measures where correct structure is lost after successful number recall; it is not a prospective exact-6 hit-rate estimate.'}
    path=OUT/'loto6_consensus_phase3_cascade_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
