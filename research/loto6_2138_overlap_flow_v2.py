#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math
from collections import Counter
from pathlib import Path
import numpy as np

from loto6_2137_cluster_flow_committee import (
    load_rows, flow_sets, shape_meta, band, sum_state, has_three_consecutive,
    cluster_audit, committee_scores
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)


def multiplicity_sets(hist):
    f1,f2,f3,f4,outside=flow_sets(hist)
    sets=[f1,f2,f3,f4]
    m={n:sum(n in s for s in sets) for n in range(1,44)}
    return (f1,f2,f3,f4,outside),m


def calibrate(rows, window=500):
    start=max(5,len(rows)-window)
    exposure=Counter(); hits=Counter(); draw_comps=Counter(); outside_counts=Counter()
    for t in range(start,len(rows)):
        hist=[r[1] for r in rows[:t]]
        (_,m)=multiplicity_sets(hist)
        cls=Counter(m.values())
        for k,v in cls.items(): exposure[k]+=v
        win=rows[t][1]
        wc=Counter(m[n] for n in win)
        for k,v in wc.items(): hits[k]+=v
        comp=(wc[0],wc[1],wc[2],wc[3]+wc[4])
        draw_comps[comp]+=1
        outside_counts[wc[0]]+=1
    rate={}; base=6/43
    for k in range(5):
        r=hits[k]/exposure[k] if exposure[k] else 0
        rate[k]={'exposure':exposure[k],'hits':hits[k],'hit_rate_per_number':r,'relative_to_base':r/base if base else 0}
    return {
        'window':len(range(start,len(rows))),
        'rate_by_overlap_count':rate,
        'winner_overlap_composition_top':[{'composition_0_1_2_3plus':list(k),'draws':v} for k,v in draw_comps.most_common(12)],
        'winner_zero_condition_count':dict(sorted(outside_counts.items()))
    }


def sum_transition(rows, prev_state='S1', window=500):
    st=[sum_state(sum(r[1])) for r in rows]
    start=max(1,len(rows)-window); c=Counter()
    for i in range(start,len(rows)):
        if st[i-1]==prev_state:c[st[i]]+=1
    total=sum(c.values())
    return {'previous_state':prev_state,'n':total,'counts':dict(c),'rates':{k:(c[k]/total if total else 0) for k in ['S1','S2','S3','S4']}}


def choose_plan(trans):
    rates=trans['rates']; raw={s:rates[s]*10 for s in rates}
    alloc={s:max(1,int(math.floor(raw[s]))) for s in rates}
    while sum(alloc.values())<10:
        s=max(rates,key=lambda x:(raw[x]-alloc[x],rates[x])); alloc[s]+=1
    while sum(alloc.values())>10:
        cand=[s for s in rates if alloc[s]>1]
        s=min(cand,key=lambda x:(raw[x]-alloc[x],rates[x])); alloc[s]-=1
    order=sorted(alloc,key=lambda s:(-rates[s],s)); plan=[]
    for s in order: plan += [s]*alloc[s]
    return alloc,plan


def main():
    rows=load_rows(); assert rows[-1][0]==2137,rows[-1][0]
    hist=[r[1] for r in rows]; bonus=np.asarray([r[2] for r in rows],np.int16)
    prev=hist[-1]; prevbo=rows[-1][2]
    (f1,f2,f3,f4,outside),mult=multiplicity_sets(hist)
    sets={'prev_same':f1,'prev_pm1':f2,'prev2_same_pm1':f3,'draws3to5_same':f4}
    sm=shape_meta(hist)
    cal=calibrate(rows,500); trans=sum_transition(rows,'S1',500); alloc,state_plan=choose_plan(trans)

    # 0/1/2-overlap centered portfolio, with only two 3+ overlap rescue lines.
    comp_plan=[
        (2,3,1,0),(2,2,2,0),(3,2,1,0),(1,3,2,0),(2,3,1,0),
        (2,2,2,0),(2,2,1,1),(1,3,1,1),(3,2,1,0),(2,2,2,0)
    ]
    # Requested ABCD allocation after an A draw: C4 / B3 / A2 / D1.
    # State order from S1 transition is normally S4,S4,S4,S3,S3,S3,S2,S2,S1,S1.
    layer_plan=['C','B','A','C','B','C','B','C','D','A']
    bo_plan=[0,0,0,0,0,0,0,0,0,1]

    candidates=[]
    for row in itertools.combinations(range(1,44),6):
        if 23 in row or 24 in row or has_three_consecutive(row): continue
        R=set(row); c1=len(R&f1); c2=len(R&f2); c3=len(R&f3); c4=len(R&f4); co=len(R&outside)
        if not (0<=c1<=2 and 1<=c2<=2 and 1<=c3<=4 and 1<=c4<=2): continue
        if sum(x<=31 for x in row)<2 or sum(x>=32 for x in row)<1: continue
        sh=band(row); meta=sm[sh]
        # Hard temporal band-shape filter: absent in last20 and no more than once in last50.
        if not (meta['c20']==0 and meta['c50']<=1): continue
        viol,detail=cluster_audit(row,sets)
        mc=Counter(mult[n] for n in row); comp=(mc[0],mc[1],mc[2],mc[3]+mc[4])
        candidates.append({
            'row':row,'state':sum_state(sum(row)),'layer':meta['layer'],'shape':sh,
            'sum':sum(row),'c20':meta['c20'],'c50':meta['c50'],'c1':c1,'c2':c2,'c3':c3,'c4':c4,
            'outside_count':co,'bo':int(prevbo in R),'cluster_viol':viol,'cluster_detail':detail,
            'overlap_comp':comp,'overlap_values':[mult[n] for n in row],
            'high_overlap_numbers':[n for n in row if mult[n]>=3]
        })

    selected=[]; numuse=Counter(); pairuse=Counter(); shapeuse=Counter(); highuse=Counter()
    state_targets={'S1':102,'S2':119,'S3':139,'S4':156}
    for pos,st in enumerate(state_plan):
        target=comp_plan[pos]; best=None; bestk=None
        for relax in range(4):
            for x in candidates:
                if x['state']!=st or x['bo']!=bo_plan[pos]: continue
                if x['row'] in [y['row'] for y in selected]: continue
                if x['cluster_viol']>(0 if relax<2 else 1): continue
                if relax==0 and x['layer']!=layer_plan[pos]: continue
                if len(x['high_overlap_numbers'])>1: continue
                if x['high_overlap_numbers'] and sum(highuse.values())>=2: continue
                comp=x['overlap_comp']; compdist=sum(abs(comp[i]-target[i]) for i in range(4))
                if relax<2 and compdist>0: continue
                if relax==2 and compdist>2: continue
                pairs=list(itertools.combinations(x['row'],2))
                k=(
                    compdist, x['cluster_viol'], 0 if x['layer']==layer_plan[pos] else 1,
                    shapeuse[x['shape']], sum(pairuse[p] for p in pairs),
                    max((numuse[n] for n in x['row']),default=0), sum(numuse[n] for n in x['row']),
                    abs(x['sum']-state_targets[st]), x['row']
                )
                if bestk is None or k<bestk: bestk=k; best=x
            if best is not None: break
        assert best is not None,(pos,st,target,layer_plan[pos])
        selected.append(best); shapeuse[best['shape']]+=1
        for n in best['row']:
            numuse[n]+=1
            if mult[n]>=3: highuse[n]+=1
        for p in itertools.combinations(best['row'],2): pairuse[p]+=1

    scores=committee_scores(np.asarray(hist,np.int16),bonus,[x['row'] for x in selected])
    out={
        'draw':2138,
        'previous':{'draw':2137,'numbers':list(prev),'bonus':prevbo,'sum':sum(prev),'state':sum_state(sum(prev))},
        'requested_layer_allocation':{'C':4,'B':3,'A':2,'D':1},
        'band_shape_rule':{'last20_max':0,'last50_max':1},
        'flow_sets':{'prev_same':sorted(f1),'prev_pm1':sorted(f2),'prev2_same_pm1':sorted(f3),'draws3to5_same':sorted(f4),'outside_four':sorted(outside)},
        'number_overlap_count':{str(n):mult[n] for n in range(1,44)},
        'calibration':cal,'sum_transition':trans,'sum_state_allocation':alloc,'portfolio':[]
    }
    for i,(x,s) in enumerate(zip(selected,scores),1):
        z=dict(x); z['row']=list(z['row']); z['shape']=list(z['shape']); z['overlap_comp']=list(z['overlap_comp'])
        z.update(s); z['slot']=i; out['portfolio'].append(z)
    out['actual_layer_allocation']=dict(Counter(x['layer'] for x in selected))
    out['actual_state_allocation']=dict(Counter(x['state'] for x in selected))
    p=OUT/'loto6_2138_overlap_flow_v2.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
