#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)

def local_rows():
    out=[]
    for k in range(1,13):
        s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8')
        payload=s.split('push(',1)[1].rsplit(');',1)[0]
        for r in json.loads(payload):
            out.append((int(r[0]), tuple(sorted(int(x) for x in r[2:8])), int(r[8])))
    out.sort(key=lambda x:x[0])
    return out

def summarize_rate(xs, ys):
    xs=np.asarray(xs,float); ys=np.asarray(ys,int)
    vals=[]
    for v in sorted(set(map(int,xs))):
        m=xs==v; vals.append({'count':v,'n':int(m.sum()),'next_hit_rate':float(ys[m].mean())})
    return vals

def rank_groups(counts):
    # cross-sectional low/mid/high thirds, stable ties by number
    order=np.lexsort((np.arange(1,44), counts))
    grp=np.empty(43,np.int8)
    grp[order[:14]]=0; grp[order[14:29]]=1; grp[order[29:]]=2
    return grp

def main():
    rows=local_rows(); rows=[r for r in rows if r[0]<=2134]
    assert rows[-1][0]==2134
    draws=[set(r[1]) for r in rows]; T=len(draws)
    A=np.zeros((T,44),np.int8)
    for i,s in enumerate(draws):
        for n in s:A[i,n]=1
    pref=A.cumsum(axis=0)
    def window_counts(t,w):
        hi=pref[t-1] if t>0 else np.zeros(44,int)
        lo=pref[t-w-1] if t-w-1>=0 else np.zeros(44,int)
        return (hi-lo)[1:44]
    windows=[20,50,100]
    result={'history_last':2134,'target_draw':2135,'individual':{},'one_digit':{},'current':{}}
    for w in windows:
        obs_x=[];obs_y=[];obs_g=[]
        per_draw_low=[];per_draw_mid=[];per_draw_high=[]
        for t in range(w,T):
            c=window_counts(t,w); y=A[t,1:44]
            g=rank_groups(c)
            obs_x.extend(c.tolist()); obs_y.extend(y.tolist()); obs_g.extend(g.tolist())
            per_draw_low.append(y[g==0].mean());per_draw_mid.append(y[g==1].mean());per_draw_high.append(y[g==2].mean())
        x=np.asarray(obs_x); y=np.asarray(obs_y); g=np.asarray(obs_g)
        rates=[float(y[g==j].mean()) for j in range(3)]
        # covariance/correlation between frequency and next-hit binary outcome
        corr=float(np.corrcoef(x,y)[0,1])
        current=window_counts(T,w)
        expected=6*w/43
        result['individual'][str(w)]={
            'expected_count':expected,
            'overall_next_hit_rate':float(y.mean()),
            'count_next_hit_table':summarize_rate(x,y),
            'cross_section_group_rates':{'low_third':rates[0],'mid_third':rates[1],'high_third':rates[2]},
            'low_vs_high_relative_rate':float(rates[0]/rates[2]) if rates[2] else None,
            'frequency_next_hit_correlation':corr,
            'current_counts':{str(n):int(current[n-1]) for n in range(1,44)},
            'current_low14':[int(x+1) for x in np.lexsort((np.arange(1,44),current))[:14]],
            'current_high14':[int(x+1) for x in np.lexsort((np.arange(1,44),current))[-14:][::-1]],
        }
        # 1-9 aggregate
        agg=[]; nxt=[]; nxt2=[]
        for t in range(w,T):
            c=window_counts(t,w); total=int(c[:9].sum()); k=int(A[t,1:10].sum())
            agg.append(total); nxt.append(k); nxt2.append(int(k>=2))
        agg=np.asarray(agg); nxt=np.asarray(nxt); nxt2=np.asarray(nxt2)
        q25=float(np.quantile(agg,.25));q75=float(np.quantile(agg,.75))
        low=agg<=q25; high=agg>=q75
        cur_total=int(current[:9].sum())
        result['one_digit'][str(w)]={
            'expected_total':9*6*w/43,
            'current_total':cur_total,
            'q25':q25,'q75':q75,
            'low_history_n':int(low.sum()),'high_history_n':int(high.sum()),
            'next_draw_mean_1digit_count_when_low':float(nxt[low].mean()),
            'next_draw_mean_1digit_count_when_high':float(nxt[high].mean()),
            'next_draw_p_ge2_when_low':float(nxt2[low].mean()),
            'next_draw_p_ge2_when_high':float(nxt2[high].mean()),
            'correlation_history_total_to_next_count':float(np.corrcoef(agg,nxt)[0,1]),
        }
    # current compact table
    for n in range(1,44):
        result['current'][str(n)]={str(w):result['individual'][str(w)]['current_counts'][str(n)] for w in windows}
    (OUT/'loto6_frequency_mean_reversion_2135.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
