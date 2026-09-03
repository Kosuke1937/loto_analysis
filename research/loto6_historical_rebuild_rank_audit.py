#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('a',ROOT/'research'/'loto6_rebuild_2134_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
p=a.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)

def support_pool21(t,Cref,stref,inc,draws,bonus,npref,ppref,actual,sizes,priors,recent20):
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors); ss[h]=p.stat_score(t,h,W,stref,inc,draws,bonus,npref)
    pc=ppref[t]-ppref[max(0,t-300)]; c5,c4=p.cores(Cref,pc)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    weighted=np.zeros(44,float); raw=np.zeros(44,int)
    for idx in agents.values():
        for rank,i in enumerate(idx[:500],1):
            w=1/np.log2(rank+2.0)
            for x in map(int,Cref[int(i)]): weighted[x]+=w; raw[x]+=1
    elig=[x for x in range(1,44) if x in recent20]
    elig.sort(key=lambda x:(weighted[x],raw[x],-x),reverse=True)
    return tuple(sorted(elig[:21])),ss[500],c5,c4,pc

def main():
    rows=p.fetch_history(); rows=[r for r in rows if r[0]<=2133];di={d:i for i,(d,_,_) in enumerate(rows)}
    draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16)
    Cref=p.fixed_sample();stref,inc,qcuts=p.build_static(Cref);draws2,bonus2,npref,ppref,actual=p.hist_actual(rows,qcuts);sizes,priors=p.prepare_priors(stref)
    cases=[]
    for draw in range(1628,2134):
        t=di.get(draw)
        if t is None or t<50: continue
        win=tuple(map(int,draws[t])); sh=a.band(win)
        c20=Counter(a.band(r) for r in draws[t-20:t]); c2150=Counter(a.band(r) for r in draws[t-50:t-20]); c50=Counter(a.band(r) for r in draws[t-50:t])
        temporal=(c20[sh]==0 and (c2150[sh]==1 or c50[sh]==0))
        recent20=set(map(int,draws[t-20:t].ravel())); all_recent=all(x in recent20 for x in win)
        if not (temporal and all_recent): continue
        pool21,sref,c5ref,c4ref,pairc=support_pool21(t,Cref,stref,inc,draws,bonus,npref,ppref,actual,sizes,priors,recent20)
        inpool=all(x in pool21 for x in win)
        prev=set(map(int,draws[t-1]));
        valid=[]
        for c in itertools.combinations(pool21,6):
            s=a.band(c)
            if c20[s]>0: continue
            if not (c2150[s]==1 or c50[s]==0): continue
            if len(set(c)&prev)>1: continue
            valid.append(c)
        rec={'draw':draw,'winner':list(win),'sum':sum(win),'pool21':list(pool21),'winner_in_pool21':inpool,'valid_count':len(valid),'winner_prev_overlap':len(set(win)&prev)}
        if inpool and len(set(win)&prev)<=1:
            W=p.weights(t,500,actual,sizes,priors)
            X,fx=a.direct_static(valid,qcuts); _,_,_,vcomm=a.direct_score(t,X,fx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref)
            WX,wfx=a.direct_static([win],qcuts); _,_,_,wc=a.direct_score(t,WX,wfx,W,draws,bonus,npref,pairc,sref,c5ref,c4ref)
            rank=int(np.count_nonzero(vcomm>wc[0])+1); rec['winner_committee']=float(wc[0]);rec['rank']=rank;rec['rank_pct']=rank/len(valid)*100 if valid else None;rec['equiv_rank_40882']=int(round(rank/len(valid)*40882)) if valid else None
        else:
            rec['winner_committee']=None;rec['rank']=None;rec['rank_pct']=None;rec['equiv_rank_40882']=None
        cases.append(rec)
        print(draw, 'pool',inpool,'n',len(valid),'rank',rec['rank'],flush=True)
    ranked=[r for r in cases if r['rank'] is not None]
    def summ(vals):
        x=np.asarray(vals,float)
        return {'n':len(x),'min':float(x.min()) if len(x) else None,'q25':float(np.quantile(x,.25)) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'mean':float(x.mean()) if len(x) else None,'q75':float(np.quantile(x,.75)) if len(x) else None,'max':float(x.max()) if len(x) else None}
    out={'definition':'For each historical strict structural case: build pre-draw multi-agent Top500 support pool of 21 numbers among numbers seen in prior20; generate combinations whose shape is absent prior20 and appears exactly once in prior21-50 or is absent prior50, with previous-draw overlap <=1; rank by Committee. Current 2134-specific manual exclusions are not applied historically.','n_strict_cases':len(cases),'winner_in_pool21':sum(r['winner_in_pool21'] for r in cases),'winner_in_pool21_rate':sum(r['winner_in_pool21'] for r in cases)/len(cases) if cases else None,'ranked_cases':len(ranked),'valid_count_summary':summ([r['valid_count'] for r in cases]),'rank_percent_summary':summ([r['rank_pct'] for r in ranked]),'equiv_rank_40882_summary':summ([r['equiv_rank_40882'] for r in ranked]),'last15':cases[-15:]}
    (OUT/'loto6_historical_rebuild_rank_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
