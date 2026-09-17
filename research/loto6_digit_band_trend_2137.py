#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from loto6_2137_cluster_flow_committee import load_rows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)

BANDS=[(1,9),(10,19),(20,29),(30,39),(40,43)]
LABELS=['1-9','10-19','20-29','30-39','40-43']

def band_counts(nums):
    return [sum(lo<=n<=hi for n in nums) for lo,hi in BANDS]

def main():
    rows=load_rows(); assert rows[-1][0]==2137, rows[-1][0]
    series=[]
    for draw,nums,bo in rows:
        c=band_counts(nums)
        series.append({'draw':draw,'counts':c,'shape':'-'.join(map(str,c)),'sum':sum(nums)})

    recent50=series[-50:]
    recent30=series[-30:]
    # Per-band count distributions over recent windows.
    dist={}
    for w in [20,30,50,100,500]:
        z=series[-w:]
        dist[str(w)]={LABELS[i]:dict(sorted(Counter(r['counts'][i] for r in z).items())) for i in range(5)}

    # First-order transition for each band count, conditioned on the latest count.
    transitions={}
    for i,label in enumerate(LABELS):
        cur=series[-1]['counts'][i]
        c=Counter()
        for a,b in zip(series[-501:-1],series[-500:]):
            if a['counts'][i]==cur:
                c[b['counts'][i]]+=1
        n=sum(c.values())
        transitions[label]={'current':cur,'n':n,'counts':dict(sorted(c.items())),'rates':{str(k):(v/n if n else 0) for k,v in sorted(c.items())}}

    # Next-shape empirical frequencies after the latest full 5-band shape over last 1000 transitions.
    cur_shape=tuple(series[-1]['counts'])
    sc=Counter()
    for a,b in zip(series[-1001:-1],series[-1000:]):
        if tuple(a['counts'])==cur_shape:
            sc[tuple(b['counts'])]+=1
    sn=sum(sc.values())
    next_shapes=[{'shape':'-'.join(map(str,k)),'count':v,'rate':v/sn if sn else 0} for k,v in sc.most_common(20)]

    # rolling 5-draw average for visualization
    roll5=[]
    for idx in range(len(series)-29,len(series)):
        start=max(0,idx-4); z=series[start:idx+1]
        roll5.append({'draw':series[idx]['draw'],'avg':[sum(r['counts'][i] for r in z)/len(z) for i in range(5)]})

    out={'latest_draw':series[-1],'labels':LABELS,'recent30':recent30,'recent50':recent50,'distribution':dist,'transition_from_latest_count':transitions,'next_shape_after_latest_shape':{'current_shape':'-'.join(map(str,cur_shape)),'n':sn,'top':next_shapes},'rolling5_recent30':roll5}
    p=OUT/'loto6_digit_band_trend_2137.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
