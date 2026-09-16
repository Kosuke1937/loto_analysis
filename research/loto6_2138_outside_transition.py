#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from loto6_2137_cluster_flow_committee import (
    load_rows, flow_sets, shape_meta, band, sum_state, has_three_consecutive,
    cluster_audit, committee_scores
)
from loto6_2138_overlap_flow import multiplicity_sets, calibrate, sum_transition

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)


def outside_series(rows):
    seq=[]
    for t in range(5,len(rows)):
        hist=[r[1] for r in rows[:t]]
        f1,f2,f3,f4,outside=flow_sets(hist)
        win=set(rows[t][1])
        seq.append({'draw':rows[t][0],'outside_count':len(win&outside)})
    return seq


def trans_stats(seq, window=None):
    z=seq[-window:] if window else seq
    mat={str(i):Counter() for i in range(7)}
    for a,b in zip(z[:-1],z[1:]):
        mat[str(a['outside_count'])][b['outside_count']]+=1
    out={}
    for k,c in mat.items():
        n=sum(c.values())
        out[k]={'n':n,'counts':{str(j):c[j] for j in range(7)},'rates':{str(j):(c[j]/n if n else 0) for j in range(7)}}
    return out


def second_order(seq, window=1000):
    z=seq[-window:] if window else seq
    d=defaultdict(Counter)
    for a,b,c in zip(z[:-2],z[1:-1],z[2:]):
        d[(a['outside_count'],b['outside_count'])][c['outside_count']]+=1
    out={}
    for k,c in d.items():
        n=sum(c.values())
        out[f'{k[0]}->{k[1]}']={'n':n,'counts':{str(j):c[j] for j in range(7)},'rates':{str(j):(c[j]/n if n else 0) for j in range(7)}}
    return out


def allocate10(rates):
    raw={k:10*v for k,v in rates.items()}
    alloc={k:int(math.floor(v)) for k,v in raw.items()}
    left=10-sum(alloc.values())
    for k in sorted(raw,key=lambda x:(raw[x]-alloc[x],rates[x]),reverse=True)[:left]: alloc[k]+=1
    return {int(k):v for k,v in alloc.items() if v>0}


def expand_plan(alloc):
    # Interleave counts rather than bunching the same outside-count scenario together.
    items=[]
    for k,v in sorted(alloc.items(), key=lambda kv:(-kv[1],kv[0])):
        items += [k]*v
    # deterministic round-robin by frequency groups
    buckets={k:[k]*v for k,v in alloc.items()}
    plan=[]
    while len(plan)<sum(alloc.values()):
        for k in sorted(buckets,key=lambda x:(-alloc[x],x)):
            if buckets[k]: plan.append(buckets[k].pop())
    return plan


def main():
    rows=load_rows(); assert rows[-1][0]==2137, rows[-1][0]
    hist=[r[1] for r in rows]; bonus=np.asarray([r[2] for r in rows],np.int16)
    prev=hist[-1]; prevbo=rows[-1][2]
    (f1,f2,f3,f4,outside),mult=multiplicity_sets(hist)
    sets={'prev_same':f1,'prev_pm1':f2,'prev2_same_pm1':f3,'draws3to5_same':f4}
    sm=shape_meta(hist)

    seq=outside_series(rows)
    current=seq[-1]['outside_count']
    prev2=seq[-2]['outside_count']
    tr_all=trans_stats(seq,None)
    tr500=trans_stats(seq,500)
    tr200=trans_stats(seq,200)
    so=second_order(seq,1000)

    # Primary allocation uses the latest 500 first-order transition from the current outside-count.
    rates=tr500[str(current)]['rates']
    alloc=allocate10(rates)
    outside_plan=expand_plan(alloc)

    # Keep previous agreed allocations for sum-state and ABCD layer.
    state_plan=['S4','S4','S4','S3','S3','S3','S2','S2','S1','S1']
    layer_plan=['C','B','A','C','B','C','B','C','D','A']  # C4/B3/A2/D1
    bo_plan=[0,0,0,0,0,0,0,0,0,1]
    comp_plan=[
        (2,3,1,0),(2,2,2,0),(3,2,1,0),(1,3,2,0),(2,3,1,0),
        (2,2,2,0),(2,2,1,1),(1,3,1,1),(3,2,1,0),(2,2,2,0)
    ]

    candidates=[]
    for row in itertools.combinations(range(1,44),6):
        if 23 in row or 24 in row or has_three_consecutive(row): continue
        R=set(row); c1=len(R&f1); c2=len(R&f2); c3=len(R&f3); c4=len(R&f4); co=len(R&outside)
        if not (0<=c1<=2 and 1<=c2<=2 and 1<=c3<=4 and 1<=c4<=2): continue
        if sum(x<=31 for x in row)<2 or sum(x>=32 for x in row)<1: continue
        sh=band(row); meta=sm[sh]
        if not (meta['c20']==0 and meta['c50']<=1): continue
        viol,detail=cluster_audit(row,sets)
        mc=Counter(mult[n] for n in row)
        comp=(mc[0],mc[1],mc[2],mc[3]+mc[4])
        candidates.append({
            'row':row,'state':sum_state(sum(row)),'layer':meta['layer'],'shape':sh,'sum':sum(row),
            'c20':meta['c20'],'c50':meta['c50'],'c1':c1,'c2':c2,'c3':c3,'c4':c4,'outside_count':co,
            'bo':int(prevbo in R),'cluster_viol':viol,'cluster_detail':detail,'overlap_comp':comp,
            'overlap_values':[mult[n] for n in row],'high_overlap_numbers':[n for n in row if mult[n]>=3]
        })

    selected=[]; numuse=Counter(); pairuse=Counter(); shapeuse=Counter(); highlines=0
    state_targets={'S1':103,'S2':119,'S3':139,'S4':156}
    for pos,(st,lay,out_target) in enumerate(zip(state_plan,layer_plan,outside_plan)):
        target=comp_plan[pos]
        best=bestk=None
        for relax in range(4):
            for x in candidates:
                if x['state']!=st or x['bo']!=bo_plan[pos]: continue
                if x['row'] in [y['row'] for y in selected]: continue
                if relax==0 and (x['layer']!=lay or x['outside_count']!=out_target or x['cluster_viol']!=0): continue
                if relax==1 and (x['layer']!=lay or abs(x['outside_count']-out_target)>0 or x['cluster_viol']>0): continue
                if relax==2 and (x['layer']!=lay or abs(x['outside_count']-out_target)>1 or x['cluster_viol']>1): continue
                if relax==3 and abs(x['outside_count']-out_target)>1: continue
                if len(x['high_overlap_numbers'])>1: continue
                if x['high_overlap_numbers'] and highlines>=2: continue
                compdist=sum(abs(x['overlap_comp'][i]-target[i]) for i in range(4))
                pairs=list(itertools.combinations(x['row'],2))
                k=(
                    abs(x['outside_count']-out_target),
                    0 if x['layer']==lay else 1,
                    compdist,
                    x['cluster_viol'],
                    shapeuse[x['shape']],
                    sum(pairuse[p] for p in pairs),
                    max((numuse[n] for n in x['row']),default=0),
                    sum(numuse[n] for n in x['row']),
                    abs(x['sum']-state_targets[st]),
                    x['row']
                )
                if bestk is None or k<bestk: bestk=k;best=x
            if best is not None: break
        assert best is not None,(pos,st,lay,out_target)
        selected.append(best); shapeuse[best['shape']]+=1
        if best['high_overlap_numbers']: highlines+=1
        for n in best['row']: numuse[n]+=1
        for p in itertools.combinations(best['row'],2): pairuse[p]+=1

    scores=committee_scores(np.asarray(hist,np.int16),bonus,[x['row'] for x in selected])
    out={
        'draw':2138,
        'previous':{'draw':2137,'numbers':list(prev),'bonus':prevbo},
        'outside_series_recent30':seq[-30:],
        'current_outside_count':current,
        'previous_outside_count':prev2,
        'transition_all':tr_all[str(current)],
        'transition_last500':tr500[str(current)],
        'transition_last200':tr200[str(current)],
        'second_order_current':so.get(f'{prev2}->{current}',{}),
        'recommended_outside_allocation_10':alloc,
        'outside_plan':outside_plan,
        'requested_layer_allocation':{'C':4,'B':3,'A':2,'D':1},
        'requested_state_allocation':{'S4':3,'S3':3,'S2':2,'S1':2},
        'band_shape_rule':{'last20_max':0,'last50_max':1},
        'portfolio':[]
    }
    for i,(x,s) in enumerate(zip(selected,scores),1):
        z=dict(x);z['row']=list(z['row']);z['shape']=list(z['shape']);z['overlap_comp']=list(z['overlap_comp']);z.update(s);z['slot']=i;z['outside_target']=outside_plan[i-1]
        out['portfolio'].append(z)
    out['actual_outside_allocation']=dict(Counter(x['outside_count'] for x in selected))
    out['actual_layer_allocation']=dict(Counter(x['layer'] for x in selected))
    out['actual_state_allocation']=dict(Counter(x['state'] for x in selected))
    p=OUT/'loto6_2138_outside_transition.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
