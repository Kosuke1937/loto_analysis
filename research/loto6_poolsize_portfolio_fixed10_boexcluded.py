#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('f',ROOT/'research'/'loto6_pool_coverage_scorematch_compare_fixed10.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
m=f.m
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
CAL_START=1628;EVAL_START=1828;END=2135;POOL_SIZES=(22,26,28,30,32)

def pool_excl(sup,k,ex):
    nums=np.array([x for x in range(1,44) if x!=ex],dtype=int)
    order=np.lexsort((nums,-sup[nums]));return tuple(map(int,nums[order[:k]]))

def main():
    rows=m.local_rows();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=m.p.fixed_sample();st,inc,q=m.p.build_static(C);sums=C.sum(1);draws,bonus,npref,ppref,actual=m.p.hist_actual(rows,q);sizes,priors=m.p.prepare_priors(st)
    poolR={k:[] for k in POOL_SIZES};R={k:{'scorematch':[],'coverage':[]} for k in (26,28,30,32)};details=[];past=[]
    for draw in range(CAL_START,END+1):
        if draw not in di:continue
        t=di[draw];ss={}
        for h in (200,500,800):
            Wh=m.p.weights(t,h,actual,sizes,priors);ss[h]=m.p.stat_score(t,h,Wh,st,inc,draws,bonus,npref)
        W=m.p.weights(t,500,actual,sizes,priors);pairc=ppref[t]-ppref[max(0,t-300)];r5,r4=m.p.cores(C,pairc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(r5)+.15*z(r4)
        agents={'stat200':m.p.topidx(ss[200],500),'stat500':m.p.topidx(ss[500],500),'stat800':m.p.topidx(ss[800],500),'committee':m.p.topidx(comm,500)};sup=m.number_support(agents,C);prevbo=int(bonus[t-1]);pools={k:pool_excl(sup,k,prevbo) for k in POOL_SIZES}
        win=tuple(map(int,draws[t]));Ws=set(win)
        if draw>=EVAL_START:
            for k,pool in pools.items():poolR[k].append(len(Ws&set(pool)))
        wsc=m.winner_score(t,win,W,q,draws,bonus,npref,pairc,ss[500],r5,r4);ash,c20,c2150,c50=m.allowed_shapes(draws,t);hall=Counter(m.band(r) for r in draws[:t]);wsh=m.band(win);prev=set(map(int,draws[t-1]));prevsum=int(draws[t-1].sum());wqual=(c20[wsh]==0 and (c2150[wsh]==1 or c50[wsh]==0) and len(Ws&prev)<=1 and prevbo not in Ws and sum(win)>prevsum)
        if draw>=EVAL_START and len(past)>=20:
            lm=m.build_layer_map(hall,t);layers=lm[st['band']];shapes=st['band'];po=inc[:,list(prev)].sum(1);base=np.isin(st['band'],ash)&(po<=1)&(inc[:,prevbo]==0)&(sums>prevsum);target=float(np.median(past));rec={'draw':draw,'winner':list(win),'winner_score':wsc,'target':target,'previous_bonus':prevbo}
            for k in (26,28,30,32):
                pool=pools[k];ids=np.where(base&(inc[:,list(pool)].sum(1)==6))[0]
                if len(ids)<10:continue
                d=np.abs(comm[ids]-target);order=ids[np.argsort(d)];s1=f.fixed_select_order(order,C,layers,shapes);s2=f.fixed_coverage(ids,d,C,layers,shapes,pool,sup);a=m.metrics(win,s1,C);b=m.metrics(win,s2,C);R[k]['scorematch'].append(a);R[k]['coverage'].append(b)
                rec[str(k)]={'pool':list(pool),'pool_recall':len(Ws&set(pool)),'valid_count':len(ids),'scorematch':a,'coverage':b}
                if draw==2135:
                    rec[str(k)]['scorematch_tickets']=[list(map(int,C[i])) for i in s1];rec[str(k)]['coverage_tickets']=[list(map(int,C[i])) for i in s2]
            details.append(rec)
        if wqual:past.append(wsc)
        if draw%25==0:print(draw,flush=True)
    pool_summary={}
    for k,v in poolR.items():pool_summary[str(k)]={'n':len(v),'mean_recall':float(np.mean(v)),'recall4plus':sum(x>=4 for x in v),'recall5plus':sum(x>=5 for x in v),'recall6':sum(x==6 for x in v),'distribution':{str(i):sum(x==i for x in v) for i in range(7)}}
    out={'definition':'Current-operation-aligned comparison: pre-draw previous bonus is excluded before selecting the supported number pool. Fixed deterministic 60k model sample; structural candidate rules and fixed10 fallback are identical across methods.','evaluation_start':EVAL_START,'end':END,'pool_summary':pool_summary,'portfolio':{},'draw2135':next((x for x in details if x['draw']==2135),None)}
    for k in (26,28,30,32):
        a=R[k]['scorematch'];b=R[k]['coverage'];out['portfolio'][str(k)]={'scorematch':m.summarize(a),'coverage':m.summarize(b),'coverage_vs_scorematch':{key:m.paired(a,b,key) for key in ('best','union','pair_recall','triple_recall','d3','d4')}}
    (OUT/'loto6_poolsize_portfolio_fixed10_boexcluded.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
