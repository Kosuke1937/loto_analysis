#!/usr/bin/env python3
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

from loto6_2137_cluster_flow_committee import load_rows

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)

BANDS=[('1-9',1,9),('10-19',10,19),('20-29',20,29),('30-39',30,39),('40-43',40,43)]

def counts(nums):
    return [sum(lo<=n<=hi for n in nums) for _,lo,hi in BANDS]

def avg(rows, w):
    z=rows[-w:]
    return [sum(r['counts'][i] for r in z)/len(z) for i in range(5)]

def transition_for_band(rows, idx, window=500):
    z=rows[-window:] if len(rows)>window else rows
    d={str(k):Counter() for k in range(7)}
    for a,b in zip(z[:-1],z[1:]):
        d[str(a['counts'][idx])][b['counts'][idx]]+=1
    out={}
    for k,c in d.items():
        n=sum(c.values())
        out[k]={'n':n,'counts':{str(j):c[j] for j in range(7)},'rates':{str(j):(c[j]/n if n else 0) for j in range(7)}}
    return out

def main():
    rows=load_rows(); assert rows[-1][0]==2137, rows[-1][0]
    seq=[]
    for draw,nums,bo in rows:
        seq.append({'draw':draw,'numbers':list(nums),'bonus':bo,'counts':counts(nums)})
    latest=seq[-1]
    out={
        'latest_draw':latest['draw'],
        'bands':[b[0] for b in BANDS],
        'latest_counts':latest['counts'],
        'recent50':seq[-50:],
        'recent30':seq[-30:],
        'averages':{str(w):avg(seq,w) for w in [10,20,50,100,500]},
        'transition_last500':{},
    }
    for i,name in enumerate(out['bands']):
        tr=transition_for_band(seq,i,500)
        out['transition_last500'][name]={
            'current_count':latest['counts'][i],
            'next_given_current':tr[str(latest['counts'][i])]
        }
    p=OUT/'loto6_2137_band_count_trend.json'
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
