#!/usr/bin/env python3
from __future__ import annotations
import json, math
from collections import Counter, defaultdict
from pathlib import Path
from loto6_2137_cluster_flow_committee import load_rows, band

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)


def shape(row): return tuple(band(row))

def dist3(ctx, cur):
    # ctx/cur are [t-3,t-2,t-1]; latest draw gets the largest weight.
    ws=(1.0,1.7,2.6)
    return sum(w*sum(abs(a-b) for a,b in zip(x,y)) for w,x,y in zip(ws,ctx,cur))

def main():
    rows=load_rows(); assert rows[-1][0]==2137
    seq=[{'draw':r[0],'shape':shape(r[1])} for r in rows]
    cur=[seq[-3]['shape'],seq[-2]['shape'],seq[-1]['shape']]
    records=[]
    for t in range(3,len(seq)):
        ctx=[seq[t-3]['shape'],seq[t-2]['shape'],seq[t-1]['shape']]
        d=dist3(ctx,cur)
        records.append((d,seq[t]['draw'],seq[t]['shape'],ctx))
    records.sort(key=lambda x:(x[0],-x[1]))
    # KNN with recency-neutral distance weighting; compare K=100/200/400 for stability.
    results={}
    for K in (100,200,400):
        neigh=records[:K]
        scores=defaultdict(float); marg=[Counter() for _ in range(5)]
        for d,dr,sh,ctx in neigh:
            w=math.exp(-0.32*d)
            scores[sh]+=w
            for j,v in enumerate(sh): marg[j][v]+=w
        total=sum(scores.values())
        ranked=[]
        for sh,s in sorted(scores.items(), key=lambda kv:kv[1], reverse=True):
            if max(sh)>=4: continue
            ranked.append({'shape':'-'.join(map(str,sh)),'prob_knn':s/total,'score':s})
        m=[]
        for j,c in enumerate(marg):
            z=sum(c.values())
            m.append({'expected':sum(k*v for k,v in c.items())/z,'rates':{str(k):c[k]/z for k in sorted(c)}})
        results[str(K)]={'top_no4plus':ranked[:15],'marginal':m}
    # Last20/50 occurrence count for candidate patterns.
    last20=Counter(x['shape'] for x in seq[-20:]); last50=Counter(x['shape'] for x in seq[-50:])
    # Consensus score across K, then keep no4+.
    cons=defaultdict(float)
    for K,wk in [(100,0.45),(200,0.35),(400,0.20)]:
        for x in results[str(K)]['top_no4plus']:
            sh=tuple(map(int,x['shape'].split('-'))); cons[sh]+=wk*x['prob_knn']
    top=[]
    for sh,s in sorted(cons.items(), key=lambda kv:kv[1], reverse=True):
        top.append({'shape':'-'.join(map(str,sh)),'consensus_score':s,'c20':last20[sh],'c50':last50[sh],
                    'passes_existing_temporal_rule':last20[sh]==0 and last50[sh]<=1})
    out={'draw':2138,'current_context':[{'draw':seq[-3+i]['draw'],'shape':'-'.join(map(str,cur[i]))} for i in range(3)],
         'labels':['1-9','10-19','20-29','30-39','40-43'],'results':results,'consensus_top':top[:20],
         'nearest_examples':[{'distance':d,'target_draw':dr,'target_shape':'-'.join(map(str,sh)),
                              'context':['-'.join(map(str,z)) for z in ctx]} for d,dr,sh,ctx in records[:20]]}
    p=OUT/'loto6_2138_band3_predict.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
