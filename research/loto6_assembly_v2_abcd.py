#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loto6 Assembly-v2: concentrate tickets without collapsing winner-number recall.

Design motivation (fixed before holdout evaluation):
- Phase1 raised union recall but weakened AssemblyRatio.
- Phase3 showed winner 3core preservation is broad, while 4th/5th completion is weak.
- Therefore do NOT bet only the top few cores. Instead use soft concentration:
  model-consensus ticket score + moderate 2/3-number overlap + union-size cap.
- Empirical ABCD band-composition layers are calculated using PRIOR draws only:
    A >= 5%, B >= 2%, C >= 0.5%, D < 0.5% in prior 500 draws.
  ABCD is a portfolio target, never a hard candidate exclusion; fallback can exceed targets.

Approximate fixed 60k candidate sample, same as Phase1. No winner injection.
Dev 1628-1877 is used only to select among a small predeclared config grid.
Holdout 1878-2127 is then evaluated once with the selected config.
"""
from __future__ import annotations
import importlib.util, itertools, json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
DEV_END=1877
TARGET={'A':3,'B':3,'C':2,'D':2}
CONFIGS=[
 {'union_cap':26,'overlap_w':0.5}, {'union_cap':26,'overlap_w':1.0},
 {'union_cap':28,'overlap_w':0.5}, {'union_cap':28,'overlap_w':1.0},
 {'union_cap':30,'overlap_w':0.5}, {'union_cap':30,'overlap_w':1.0},
]

def z(x): return (x-x.mean())/(x.std()+1e-9)

def band_tuple(row):
    a=np.asarray(row)
    return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))

def abcd_layer(t, draw_band, shape, lookback=500):
    lo=max(0,t-lookback); n=max(1,t-lo)
    f=sum(draw_band[u]==shape for u in range(lo,t))/n
    if f>=.05:return 'A'
    if f>=.02:return 'B'
    if f>=.005:return 'C'
    return 'D'

def agent_state(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc):
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors); ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
    pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc)
    comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    A={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    return A,comm

def ranked_pool(A,comm,C):
    v3,rw3,v4,rw4=p.support(A,C)
    pool=np.unique(np.concatenate([x[:1500] for x in A.values()]))
    cr={}
    for name,idx in A.items():
        for rank,i in enumerate(idx[:1500],1): cr[(name,int(i))]=1/np.log2(rank+2)
    def base(i):
        row=tuple(map(int,C[i])); b3=sorted(((v3[c],rw3[c]) for c in itertools.combinations(row,3)),reverse=True)[:3]
        b4=sorted(((v4[c],rw4[c]) for c in itertools.combinations(row,4)),reverse=True)[:2]
        model=sum(cr.get((n,int(i)),0.0) for n in A)
        return 2.2*b4[0][0]+1.0*b3[0][0]+.55*b4[0][1]+.25*b3[0][1]+.30*model+.08*float(comm[i])
    scores={int(i):base(int(i)) for i in pool}
    return sorted(scores,key=scores.get,reverse=True),scores

def select_v2(order,scores,C,t,draw_band,cfg):
    sel=[]; union=set(); tc=Counter(); lc=Counter()
    # layer-aware soft rounds: aim at targets, but never exclude a layer permanently.
    pending=dict(TARGET)
    def candidate_gain(i):
        row=tuple(map(int,C[i])); s=set(row)
        ovs=[len(s&set(map(int,C[j]))) for j in sel]
        ov=max(ovs or [0])
        # reward 2/3 overlap (assembly), neutral 1/4, penalize 0/5+
        shape_bonus={0:-1.0,1:0.0,2:1.0,3:1.35,4:0.15,5:-2.0,6:-3.0}.get(ov,-3.0)
        newn=len(s-union)
        return scores[i]+cfg['overlap_w']*shape_bonus-.10*newn
    def allowed(i, respect_layer=True):
        row=tuple(map(int,C[i])); s=set(row); newU=union|s
        if len(newU)>cfg['union_cap']: return False
        if any(len(s&set(map(int,C[j])))>=5 for j in sel): return False
        # no 3core more than twice: permits branching, avoids total collapse
        for c in itertools.combinations(row,3):
            if tc[c]>=2:return False
        if respect_layer:
            L=abcd_layer(t,draw_band,band_tuple(row))
            if pending[L]<=0:return False
        return True
    # dynamic greedy, recalculating overlap score each pick
    while len(sel)<10:
        cand=[i for i in order[:6000] if i not in sel and allowed(i,True)]
        if not cand: break
        i=max(cand,key=candidate_gain); row=tuple(map(int,C[i]));L=abcd_layer(t,draw_band,band_tuple(row))
        sel.append(i); union.update(row);pending[L]-=1;lc[L]+=1
        for c in itertools.combinations(row,3):tc[c]+=1
    # fallback relaxes layer target, then union cap if needed; ABCD is not hard exclusion
    for relax_union in (0,2,5,43):
        if len(sel)>=10:break
        oldcap=cfg['union_cap']; cfgcap=min(43,oldcap+relax_union)
        cand=[]
        for i in order:
            if i in sel:continue
            row=tuple(map(int,C[i]));s=set(row)
            if len(union|s)>cfgcap:continue
            if any(len(s&set(map(int,C[j])))>=5 for j in sel):continue
            if any(tc[c]>=2 for c in itertools.combinations(row,3)):continue
            cand.append(i)
            if len(cand)>=2000:break
        while cand and len(sel)<10:
            i=max(cand,key=candidate_gain); cand.remove(i);row=tuple(map(int,C[i]));L=abcd_layer(t,draw_band,band_tuple(row))
            sel.append(i);union.update(row);lc[L]+=1
            for c in itertools.combinations(row,3):tc[c]+=1
            cand=[j for j in cand if len(union|set(map(int,C[j])))<=cfgcap and not any(tc[c]>=2 for c in itertools.combinations(tuple(map(int,C[j])),3))]
    return sel[:10],dict(lc),len(union)

def metric(win,sel,C):
    if not sel:return {'union':0,'best':0,'assembly':0,'core3':0,'core4':0,'d3':0,'d4':0,'d5':0,'d6':0}
    m=p.metrics(win,sel,C);b=m['best'];m.update({'d3':int(b>=3),'d4':int(b>=4),'d5':int(b>=5),'d6':int(b>=6)});return m

def summ(R):
    return {'draws':len(R),'mean_union':float(np.mean([r['union'] for r in R])),'union5plus':sum(r['union']>=5 for r in R),'union6':sum(r['union']==6 for r in R),'core3_capture':sum(r['core3'] for r in R),'core4_capture':sum(r['core4'] for r in R),'mean_best':float(np.mean([r['best'] for r in R])),'mean_assembly':float(np.mean([r['assembly'] for r in R])),'d3':sum(r['d3'] for r in R),'d4':sum(r['d4'] for r in R),'d5':sum(r['d5'] for r in R),'d6':sum(r['d6'] for r in R)}
def obj(s): return (s['d5'],s['d4'],s['d3'],s['mean_best'],s['core3_capture'],s['mean_assembly'])

def main():
    rows=p.fetch_history(); di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,gq=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,gq);sizes,priors=p.prepare_priors(st)
    draw_band=[band_tuple(r) for r in draws]
    dev_base=[];hold_base=[];by_dev={str(c):[] for c in CONFIGS};by_hold={str(c):[] for c in CONFIGS};layer_logs={str(c):Counter() for c in CONFIGS};union_logs={str(c):[] for c in CONFIGS}
    for draw in range(p.START_DRAW,p.END_DRAW+1):
        t=di[draw];A,comm=agent_state(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc)
        base=p.consensus_portfolio(A,{'committee':comm},C); bm=metric(draws[t],base,C);(dev_base if draw<=DEV_END else hold_base).append(bm)
        order,scores=ranked_pool(A,comm,C)
        for cfg0 in CONFIGS:
            cfg=dict(cfg0);sel,lc,usz=select_v2(order,scores,C,t,draw_band,cfg);m=metric(draws[t],sel,C)
            key=str(cfg0);(by_dev if draw<=DEV_END else by_hold)[key].append(m);layer_logs[key].update(lc);union_logs[key].append(usz)
    dev_summary={k:summ(v) for k,v in by_dev.items()}; bestkey=max(dev_summary,key=lambda k:obj(dev_summary[k])); bestcfg=eval(bestkey)
    out={'method':'Assembly-v2 soft concentration + dynamic empirical ABCD portfolio','candidate_sample':{'size':p.NSAMPLE,'seed':p.SEED,'winner_injection':False},'ABCD':{'definition':'prior-500 empirical five-band composition frequency','bands':'1-9/10-19/20-29/30-39/40-43','A':'>=5%','B':'2-5%','C':'0.5-2%','D':'<0.5%','target':TARGET,'hard_exclusion':False},'development':{'range':f'{p.START_DRAW}-{DEV_END}','baseline':summ(dev_base),'configs':dev_summary,'selected':bestcfg,'selected_summary':dev_summary[bestkey]},'holdout':{'range':f'{DEV_END+1}-{p.END_DRAW}','baseline':summ(hold_base),'v2':summ(by_hold[bestkey])},'selected_portfolio_diagnostics':{'layer_ticket_totals':dict(layer_logs[bestkey]),'mean_union_size':float(np.mean(union_logs[bestkey]))},'caveat':'Approximate 60k candidate-sample backtest. Small config selection uses development only; holdout 1878-2127 is the relevant stability check. ABCD is computed causally from prior draws and used only as soft portfolio allocation.'}
    (OUT/'loto6_assembly_v2_abcd_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
