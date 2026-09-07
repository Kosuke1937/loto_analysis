#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter
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
        m=xs==v
        vals.append({'count':v,'n':int(m.sum()),'next_hit_rate':float(ys[m].mean())})
    return vals

def rank_groups(counts):
    order=np.lexsort((np.arange(1,44), counts))
    grp=np.empty(43,np.int8)
    grp[order[:14]]=0; grp[order[14:29]]=1; grp[order[29:]]=2
    return grp

def main():
    rows=local_rows(); rows=[r for r in rows if r[0]<=2134]
    assert rows[-1][0]==2134
    draws=[tuple(r[1]) for r in rows]; T=len(draws)
    A=np.zeros((T,44),np.int8)
    D=np.zeros((T,10),np.int8)
    for i,row in enumerate(draws):
        for n in row:
            A[i,n]=1
            D[i,n%10]+=1
    pref=A.cumsum(axis=0)
    dpref=D.cumsum(axis=0)
    def window_counts(t,w):
        hi=pref[t-1] if t>0 else np.zeros(44,int)
        lo=pref[t-w-1] if t-w-1>=0 else np.zeros(44,int)
        return (hi-lo)[1:44]
    def digit_counts(t,w):
        hi=dpref[t-1] if t>0 else np.zeros(10,int)
        lo=dpref[t-w-1] if t-w-1>=0 else np.zeros(10,int)
        return hi-lo
    windows=[20,50,100]
    digit_card={d:len([n for n in range(1,44) if n%10==d]) for d in range(10)}
    result={'history_last':2134,'target_draw':2135,'definitions':{'individual':'exact number 1-43','last_digit':'units digit n%10; e.g. 35->5, 42->2'},'digit_cardinality':digit_card,'individual':{},'last_digit':{},'current':{}}
    for w in windows:
        obs_x=[];obs_y=[];obs_g=[]
        for t in range(w,T):
            c=window_counts(t,w); y=A[t,1:44]; g=rank_groups(c)
            obs_x.extend(c.tolist()); obs_y.extend(y.tolist()); obs_g.extend(g.tolist())
        x=np.asarray(obs_x); y=np.asarray(obs_y); g=np.asarray(obs_g)
        rates=[float(y[g==j].mean()) for j in range(3)]
        corr=float(np.corrcoef(x,y)[0,1])
        current=window_counts(T,w)
        result['individual'][str(w)]={
            'expected_count_per_number':6*w/43,
            'overall_next_hit_rate':float(y.mean()),
            'count_next_hit_table':summarize_rate(x,y),
            'cross_section_group_rates':{'low_third':rates[0],'mid_third':rates[1],'high_third':rates[2]},
            'low_vs_high_relative_rate':float(rates[0]/rates[2]) if rates[2] else None,
            'frequency_next_hit_correlation':corr,
            'current_counts':{str(n):int(current[n-1]) for n in range(1,44)},
        }
        # Last-digit analysis, preserving each digit's unequal candidate cardinality.
        curd=digit_counts(T,w)
        per_digit={}
        all_z=[]; all_next=[]
        for d in range(10):
            hist=[]; nxt=[]
            for t in range(w,T):
                hist.append(int(digit_counts(t,w)[d]))
                nxt.append(int(D[t,d]))
            hist=np.asarray(hist,float); nxt=np.asarray(nxt,float)
            q33=float(np.quantile(hist,1/3)); q67=float(np.quantile(hist,2/3))
            low=hist<=q33; high=hist>=q67
            expected=w*6*digit_card[d]/43
            sd=float(hist.std()) or 1.0
            z=(hist-hist.mean())/sd
            all_z.extend(z.tolist()); all_next.extend(nxt.tolist())
            per_digit[str(d)]={
                'candidate_numbers':[n for n in range(1,44) if n%10==d],
                'expected_count':expected,
                'current_count':int(curd[d]),
                'current_vs_expected':float(curd[d]-expected),
                'current_ratio_to_expected':float(curd[d]/expected),
                'historical_mean_count':float(hist.mean()),
                'q33':q33,'q67':q67,
                'next_draw_mean_digit_count_when_recent_low':float(nxt[low].mean()),
                'next_draw_mean_digit_count_when_recent_high':float(nxt[high].mean()),
                'low_vs_high_relative_next_count':float(nxt[low].mean()/nxt[high].mean()) if nxt[high].mean()>0 else None,
                'correlation_recent_count_to_next_count':float(np.corrcoef(hist,nxt)[0,1]),
                'next_draw_p_any_when_recent_low':float((nxt[low]>0).mean()),
                'next_draw_p_any_when_recent_high':float((nxt[high]>0).mean()),
            }
        result['last_digit'][str(w)]={
            'current_counts':{str(d):int(curd[d]) for d in range(10)},
            'per_digit':per_digit,
            'pooled_standardized_frequency_to_next_count_correlation':float(np.corrcoef(np.asarray(all_z),np.asarray(all_next))[0,1]),
        }
    for n in range(1,44):
        result['current'][str(n)]={str(w):result['individual'][str(w)]['current_counts'][str(n)] for w in windows}
    (OUT/'loto6_frequency_mean_reversion_2135.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
