#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

from loto6_2137_cluster_flow_committee import load_rows, band, shape_meta, sum_state

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
PRED=OUT/'loto6_2138_multihorizon_band_predict.json'


def smoothed_next_probs(seq, context_len, alpha=2.0):
    labels=['A','B','C','D']
    base=Counter(seq[:-1]); base_n=sum(base.values())
    prior={x:(base[x]/base_n if base_n else .25) for x in labels}
    ctx=tuple(seq[-context_len:])
    cnt=Counter(); n=0
    for i in range(context_len, len(seq)):
        if tuple(seq[i-context_len:i])==ctx:
            cnt[seq[i]]+=1; n+=1
    probs={x:(cnt[x]+alpha*prior[x])/(n+alpha) for x in labels}
    return ctx,n,probs,cnt


def transition_from_A(seq, start=0):
    labels=['A','B','C','D']; cnt=Counter(); n=0
    for i in range(max(1,start),len(seq)):
        if seq[i-1]=='A': cnt[seq[i]]+=1; n+=1
    return n,{x:(cnt[x]/n if n else 0.0) for x in labels},cnt


def main():
    rows=load_rows(); assert rows[-1][0]==2137
    hist=[r[1] for r in rows]
    shapes=[band(r) for r in hist]
    sm=shape_meta(hist)
    layers=[sm[s]['layer'] for s in shapes]
    states=[sum_state(sum(r)) for r in hist]
    pred=json.loads(PRED.read_text(encoding='utf-8'))
    candidates=pred['consensus_top_temporal_no4plus'][:20]

    # Current layer flow context
    ng={}
    for L in (1,2,3):
        ctx,n,p,c=smoothed_next_probs(layers,L,alpha=2.0)
        ng[str(L)]={'context':list(ctx),'n_exact':n,'probs':p,'raw_counts':dict(c)}
    nA_all,pA_all,cA_all=transition_from_A(layers,0)
    nA_500,pA_500,cA_500=transition_from_A(layers,max(0,len(layers)-500))

    # Sum-state next transition from current S1
    labelsS=['S1','S2','S3','S4']
    cntS=Counter(); nS=0
    for i in range(1,len(states)):
        if states[i-1]==states[-1]: cntS[states[i]]+=1; nS+=1
    pS={x:(cntS[x]/nS if nS else 0.0) for x in labelsS}

    # Historical state distribution conditional on each band shape
    shape_state=defaultdict(Counter); shape_n=Counter()
    for sh,st in zip(shapes,states):
        shape_state[sh][st]+=1; shape_n[sh]+=1

    reviewed=[]
    for x in candidates:
        sh=tuple(map(int,x['shape'].split('-')))
        layer=sm[sh]['layer']
        n=shape_n[sh]
        sdist={st:(shape_state[sh][st]/n if n else 0.0) for st in labelsS}
        # Compatibility of this shape's historical sum-state mix with current prev-state transition.
        sum_compat=sum(sdist[st]*pS[st] for st in labelsS)
        # Layer-flow consensus: current A transition (recent+all) plus exact recent context ngrams.
        layer_flow=(0.25*pA_all[layer]+0.35*pA_500[layer]+0.20*ng['1']['probs'][layer]+0.12*ng['2']['probs'][layer]+0.08*ng['3']['probs'][layer])
        reviewed.append({
            **x,
            'layer':layer,
            'long_freq':sm[sh]['freq'],
            'layer_flow':layer_flow,
            'shape_occurrences':n,
            'shape_sum_state_dist':sdist,
            'sum_flow_compat':sum_compat,
            's34_share':sdist['S3']+sdist['S4'],
            'b30':sh[3], 'b40':sh[4]
        })

    # Ranks, no opaque composite score: rank separately on three evidence streams.
    def rank_map(key, reverse=True):
        vals=sorted(range(len(reviewed)), key=lambda i: reviewed[i][key], reverse=reverse)
        return {i:r+1 for r,i in enumerate(vals)}
    r_shape=rank_map('consensus_score'); r_layer=rank_map('layer_flow'); r_sum=rank_map('sum_flow_compat')
    for i,r in enumerate(reviewed):
        r['rank_shape_flow']=r_shape[i]; r['rank_layer_flow']=r_layer[i]; r['rank_sum_flow']=r_sum[i]
        r['rank_mean']=(r_shape[i]+r_layer[i]+r_sum[i])/3.0
    reviewed.sort(key=lambda r:(r['rank_mean'], r['rank_shape_flow']))

    out={
      'draw':2138,
      'latest':{'draw':rows[-1][0],'shape':'-'.join(map(str,shapes[-1])),'layer':layers[-1],'state':states[-1]},
      'last12':[{'draw':rows[i][0],'shape':'-'.join(map(str,shapes[i])),'layer':layers[i],'state':states[i]} for i in range(len(rows)-12,len(rows))],
      'layer_transition_from_A_all':{'n':nA_all,'probs':pA_all,'counts':dict(cA_all)},
      'layer_transition_from_A_last500':{'n':nA_500,'probs':pA_500,'counts':dict(cA_500)},
      'layer_ngram':ng,
      'sum_transition_from_latest_state':{'from':states[-1],'n':nS,'probs':pS,'counts':dict(cntS)},
      'reviewed_candidates':reviewed
    }
    p=OUT/'loto6_2138_band_flow_review.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
