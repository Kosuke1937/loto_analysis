#!/usr/bin/env python3
from __future__ import annotations
import json, math
from collections import Counter, defaultdict
from pathlib import Path

from loto6_2137_cluster_flow_committee import load_rows, band, shape_meta

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
HORIZONS=(3,5,8,12)
K=180


def dist(seq, i, L):
    # Compare current last-L band-count path with historical path ending at i-1.
    # Recent lags weigh slightly more, but every draw in the window contributes.
    cur=seq[-L:]
    past=seq[i-L:i]
    s=0.0; sw=0.0
    for j,(a,b) in enumerate(zip(past,cur)):
        # j=L-1 is most recent
        w=0.82**(L-1-j)
        s += w*sum(abs(x-y) for x,y in zip(a,b))
        sw += w
    return s/sw


def predict_horizon(seq, rows, L, k=K):
    cand=[]
    for i in range(L, len(seq)):
        # seq[i] is the historical next draw after context seq[i-L:i]
        d=dist(seq,i,L)
        cand.append((d,i))
    cand.sort(key=lambda z:z[0])
    near=cand[:min(k,len(cand))]
    cnt=Counter(); wt=Counter(); examples=[]
    for rank,(d,i) in enumerate(near):
        # smooth inverse-distance × mild rank decay
        w=(1.0/(1.0+d))*math.exp(-rank/max(1,len(near)))
        sh=tuple(seq[i])
        cnt[sh]+=1; wt[sh]+=w
        if len(examples)<12:
            examples.append({'distance':d,'context_end_draw':rows[i-1][0],'next_draw':rows[i][0],'next_shape':'-'.join(map(str,sh))})
    total=sum(wt.values())
    probs={sh:(v/total if total else 0.0) for sh,v in wt.items()}
    marg=[]
    for b in range(5):
        c=Counter(); tw=0.0
        for d,i in near:
            w=(1.0/(1.0+d)) # marginals don't need rank decay twice
            c[seq[i][b]]+=w; tw+=w
        rates={str(n):c[n]/tw for n in sorted(c)} if tw else {}
        exp=sum(n*c[n] for n in c)/tw if tw else 0.0
        marg.append({'expected':exp,'rates':rates})
    return probs,marg,examples


def main():
    rows=load_rows(); assert rows[-1][0]==2137
    seq=[band(r[1]) for r in rows]
    sm=shape_meta([r[1] for r in rows])
    byH={}; all_shapes=set()
    for L in HORIZONS:
        probs,marg,examples=predict_horizon(seq,rows,L,K)
        byH[str(L)]={'marginal':marg,'nearest_examples':examples}
        byH[str(L)]['top']=[{'shape':'-'.join(map(str,sh)),'score':p} for sh,p in sorted(probs.items(), key=lambda kv:kv[1], reverse=True)[:30]]
        byH[str(L)]['_probs']=probs
        all_shapes.update(probs)

    consensus=[]
    for sh in all_shapes:
        scores=[byH[str(L)]['_probs'].get(sh,0.0) for L in HORIZONS]
        mean=sum(scores)/len(scores)
        if max(sh)>=4: continue
        meta=sm[sh]
        temporal=(meta['c20']==0 and meta['c50']<=1)
        consensus.append({'shape':'-'.join(map(str,sh)),'consensus_score':mean,'horizon_scores':{str(L):scores[j] for j,L in enumerate(HORIZONS)},'c20':meta['c20'],'c50':meta['c50'],'passes_temporal':temporal})
    consensus.sort(key=lambda x:x['consensus_score'],reverse=True)

    # Consensus marginal is simple mean across horizons, to avoid one chosen lookback dominating.
    cm=[]
    for b in range(5):
        values=defaultdict(float)
        exp=0.0
        for L in HORIZONS:
            m=byH[str(L)]['marginal'][b]
            exp+=m['expected']/len(HORIZONS)
            for n,p in m['rates'].items(): values[n]+=p/len(HORIZONS)
        cm.append({'expected':exp,'rates':dict(sorted(values.items(), key=lambda kv:int(kv[0])))})

    out={
      'draw':2138,
      'labels':['1-9','10-19','20-29','30-39','40-43'],
      'horizons':list(HORIZONS),
      'current_context_last12':[{'draw':rows[i][0],'shape':'-'.join(map(str,seq[i]))} for i in range(len(rows)-12,len(rows))],
      'consensus_marginal':cm,
      'consensus_top_no4plus':consensus[:30],
      'consensus_top_temporal_no4plus':[x for x in consensus if x['passes_temporal']][:20],
      'by_horizon':{k:{kk:vv for kk,vv in v.items() if kk!='_probs'} for k,v in byH.items()}
    }
    p=OUT/'loto6_2138_multihorizon_band_predict.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
