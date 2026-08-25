#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loto6 Consensus-Assembly Phase 2: reverse diagnosis + leakage-safe split validation.

Uses the frozen Phase1 feature/agent implementation by importing
research/loto6_consensus_phase1_sample.py.

Exploratory reverse diagnosis:
  - Focus on Phase1 Consensus rounds with winner union coverage 6/6.
  - Rank winner 3cores/4cores by observable multi-agent support.
  - Build model-only recombination candidates from the selected union and measure
    whether the actual winner appears in top 10/50/100/500 assembly candidates.

Prospective validation:
  - Development: draws 1628-1877.
  - Holdout:     draws 1878-2127.
  - Hyperparameters are selected ONLY on development, then frozen on holdout.
  - Compare Phase1 Consensus10 vs Selective Core Expansion mixed10.

No target winner is used to generate/rank tickets. Winner is only used after
portfolio creation for evaluation.
"""
from __future__ import annotations
import importlib.util, itertools, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PH1=ROOT/'research'/'loto6_consensus_phase1_sample.py'
spec=importlib.util.spec_from_file_location('ph1',PH1)
p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
DEV_END=1877


def rw(rank): return 1.0/np.log2(rank+2.0)

def tuple_row(C,i): return tuple(map(int,C[int(i)]))

def core_support(agents,C,topn=500):
    v3=Counter(); w3=Counter(); v4=Counter(); w4=Counter()
    for name,idx in agents.items():
        a3={};a4={}
        for rank,i in enumerate(idx[:topn],1):
            r=tuple_row(C,i); z=rw(rank)
            for c in itertools.combinations(r,3): a3[c]=max(a3.get(c,0.0),z)
            for c in itertools.combinations(r,4): a4[c]=max(a4.get(c,0.0),z)
        for c,z in a3.items(): v3[c]+=1; w3[c]+=z
        for c,z in a4.items(): v4[c]+=1; w4[c]+=z
    return v3,w3,v4,w4

def score_core3(c,v3,w3): return (v3[c],w3[c])
def score_core4(c,v4,w4): return (v4[c],w4[c])

def full_ticket_score(t,v3,w3,v4,w4):
    s4=sorted(((v4[c],w4[c]) for c in itertools.combinations(t,4)),reverse=True)
    s3=sorted(((v3[c],w3[c]) for c in itertools.combinations(t,3)),reverse=True)
    # robustly reward several supported subcores, not only one dominant core
    return (4.0*s4[0][0] + 1.2*s4[1][0] + 1.8*s3[0][0] + .6*s3[1][0]
            + .50*s4[0][1] + .15*s4[1][1] + .20*s3[0][1] + .08*s3[1][1])

def assemble_candidates(union_nums,agents,C,top_core_k=3,max_candidates=800):
    U=tuple(sorted(union_nums)); v3,w3,v4,w4=core_support(agents,C)
    all3=list(itertools.combinations(U,3))
    all3.sort(key=lambda c:(v3[c],w3[c]),reverse=True)
    seeds=all3[:top_core_k]
    pool={}
    for core in seeds:
        rest=[n for n in U if n not in core]
        for add in itertools.combinations(rest,3):
            t=tuple(sorted(core+add))
            pool[t]=max(pool.get(t,-1e99),full_ticket_score(t,v3,w3,v4,w4))
    ordered=sorted(pool,key=lambda t:pool[t],reverse=True)
    return ordered[:max_candidates], (v3,w3,v4,w4)

def phase1_portfolios(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc):
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors); ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
    pc=ppref[t]-ppref[max(0,t-300)]; c5,c4=p.cores(C,pc)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={
        'stat200':p.topidx(ss[200],1500),
        'stat500':p.topidx(ss[500],1500),
        'stat800':p.topidx(ss[800],1500),
        'committee':p.topidx(comm,1500),
    }
    base=p.diverse(agents['committee'],C,n=10,triple_cap=1,pair_cap=2,num_cap=4)
    con=p.consensus_portfolio(agents,{'committee':comm},C)
    return agents,base,con

def metrics_ticket_tuples(win,tickets):
    W=set(map(int,win)); U=set().union(*(set(t) for t in tickets)) if tickets else set()
    uc=len(W&U); best=max([len(W&set(t)) for t in tickets] or [0])
    w3=set(itertools.combinations(tuple(sorted(W)),3)); w4=set(itertools.combinations(tuple(sorted(W)),4))
    s3=set(c for t in tickets for c in itertools.combinations(t,3)); s4=set(c for t in tickets for c in itertools.combinations(t,4))
    return {'union':uc,'core3':int(bool(w3&s3)),'core4':int(bool(w4&s4)),'best':best,'assembly':best/uc if uc else 0.0}

def mixed_portfolio(con_idx,C,agents,top_core_k,n_replace):
    original=[tuple_row(C,i) for i in con_idx]
    U=set().union(*(set(t) for t in original))
    generated,_=assemble_candidates(U,agents,C,top_core_k=top_core_k,max_candidates=500)
    keep_n=10-n_replace
    selected=original[:keep_n]
    def maxov(t): return max([len(set(t)&set(s)) for s in selected] or [0])
    # favor high-ranked generated tickets while blocking 5-number duplicates
    for t in generated:
        if t in selected: continue
        if maxov(t)>=5: continue
        selected.append(t)
        if len(selected)>=10: break
    # deterministic fallback
    for t in original:
        if t not in selected: selected.append(t)
        if len(selected)>=10: break
    return selected[:10],generated

def summary(ms):
    if not ms:return {}
    return {
      'draws':len(ms),'mean_union':float(np.mean([x['union'] for x in ms])),
      'union5plus':sum(x['union']>=5 for x in ms),'union6':sum(x['union']==6 for x in ms),
      'core3_capture':sum(x['core3'] for x in ms),'core4_capture':sum(x['core4'] for x in ms),
      'mean_assembly':float(np.mean([x['assembly'] for x in ms])),'mean_best':float(np.mean([x['best'] for x in ms])),
      'd3':sum(x['best']>=3 for x in ms),'d4':sum(x['best']>=4 for x in ms),
      'd5':sum(x['best']>=5 for x in ms),'d6':sum(x['best']>=6 for x in ms)
    }

def objective(s):
    # Assembly objective: prioritize upper-tail matches, then d3 and mean best.
    return (s['d5'],s['d4'],s['d3'],s['mean_best'],s['mean_assembly'])

def main():
    rows=p.fetch_history(); draw_to_idx={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample(); st,inc,q=p.build_static(C)
    draws,bonus,npref,ppref,actual=p.hist_actual(rows,q); sizes,priors=p.prepare_priors(st)

    cache={}; reverse=[]
    configs=[(k,r) for k in (1,2,3,5) for r in (2,4,6)]
    dev_by={cfg:[] for cfg in configs}; test_by={cfg:[] for cfg in configs}
    dev_base=[];test_base=[]

    for draw in range(p.START_DRAW,p.END_DRAW+1):
        t=draw_to_idx.get(draw)
        if t is None: continue
        agents,base,con=phase1_portfolios(t,C,draws,bonus,npref,ppref,actual,sizes,priors,st,inc)
        win=tuple(map(int,draws[t]))
        con_t=[tuple_row(C,i) for i in con]
        mb=metrics_ticket_tuples(win,con_t)
        target = dev_base if draw<=DEV_END else test_base; target.append(mb)

        # Reverse diagnosis only uses winner after observable rankings are created.
        U=set().union(*(set(x) for x in con_t))
        generated,supp=assemble_candidates(U,agents,C,top_core_k=5,max_candidates=500)
        v3,w3,v4,w4=supp
        if mb['union']==6:
            all3=sorted(itertools.combinations(sorted(U),3),key=lambda c:(v3[c],w3[c]),reverse=True)
            all4=sorted(itertools.combinations(sorted(U),4),key=lambda c:(v4[c],w4[c]),reverse=True)
            w3s=set(itertools.combinations(tuple(sorted(win)),3)); w4s=set(itertools.combinations(tuple(sorted(win)),4))
            r3=min([i+1 for i,c in enumerate(all3) if c in w3s] or [999999])
            r4=min([i+1 for i,c in enumerate(all4) if c in w4s] or [999999])
            try: exrank=generated.index(tuple(sorted(win)))+1
            except ValueError: exrank=999999
            reverse.append({'draw':draw,'best_winner3core_rank':r3,'best_winner4core_rank':r4,'generated_exact_rank':exrank,'phase1_best':mb['best']})

        for cfg in configs:
            k,r=cfg; mix,_=mixed_portfolio(con,C,agents,k,r)
            mm=metrics_ticket_tuples(win,mix)
            (dev_by if draw<=DEV_END else test_by)[cfg].append(mm)

    dev_scores={str(cfg):summary(v) for cfg,v in dev_by.items()}
    bestcfg=max(configs,key=lambda cfg:objective(summary(dev_by[cfg])))
    out={
      'method':'Phase2 reverse diagnosis + dev/holdout Selective Core Expansion',
      'candidate_sample':{'seed':p.SEED,'size':p.NSAMPLE,'target_winner_injection':False},
      'split':{'development':f'{p.START_DRAW}-{DEV_END}','holdout':f'{DEV_END+1}-{p.END_DRAW}'},
      'reverse_union6':{
        'count':len(reverse),
        'winner3core_rank_le1':sum(x['best_winner3core_rank']<=1 for x in reverse),
        'winner3core_rank_le3':sum(x['best_winner3core_rank']<=3 for x in reverse),
        'winner3core_rank_le5':sum(x['best_winner3core_rank']<=5 for x in reverse),
        'winner3core_rank_le10':sum(x['best_winner3core_rank']<=10 for x in reverse),
        'winner4core_rank_le1':sum(x['best_winner4core_rank']<=1 for x in reverse),
        'winner4core_rank_le3':sum(x['best_winner4core_rank']<=3 for x in reverse),
        'winner4core_rank_le5':sum(x['best_winner4core_rank']<=5 for x in reverse),
        'winner4core_rank_le10':sum(x['best_winner4core_rank']<=10 for x in reverse),
        'exact_in_generated_top10':sum(x['generated_exact_rank']<=10 for x in reverse),
        'exact_in_generated_top50':sum(x['generated_exact_rank']<=50 for x in reverse),
        'exact_in_generated_top100':sum(x['generated_exact_rank']<=100 for x in reverse),
        'exact_in_generated_top500':sum(x['generated_exact_rank']<=500 for x in reverse),
      },
      'development':{'phase1':summary(dev_base),'configs':dev_scores,'selected_config':{'top_core_k':bestcfg[0],'n_replace':bestcfg[1],'summary':summary(dev_by[bestcfg])}},
      'holdout':{'phase1':summary(test_base),'selected_config':{'top_core_k':bestcfg[0],'n_replace':bestcfg[1],'summary':summary(test_by[bestcfg])}},
      'reverse_detail':reverse,
      'caveat':'Reverse union6 diagnostics are exploratory/oracle-facing. Only the frozen development-selected configuration evaluated on the 1878-2127 holdout is evidence for prospective assembly improvement.'
    }
    path=OUT/'loto6_consensus_phase2_reverse_summary.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='reverse_detail'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
