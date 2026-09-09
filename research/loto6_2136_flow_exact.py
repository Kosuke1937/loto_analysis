#!/usr/bin/env python3
from __future__ import annotations
import itertools, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)

BANDS=((1,9),(10,19),(20,29),(30,39),(40,43))

def load_rows():
    rows=[]
    for k in range(1,13):
        p=ROOT/'data'/f'loto6-chunk-{k}.js'
        if not p.exists(): continue
        s=p.read_text(encoding='utf-8')
        payload=s.split('push(',1)[1].rsplit(');',1)[0]
        for r in json.loads(payload):
            rows.append((int(r[0]), tuple(int(x) for x in r[2:8]), int(r[8])))
    rows.sort()
    return rows

def band(row):
    return tuple(sum(lo<=x<=hi for x in row) for lo,hi in BANDS)

def layer(freq):
    return 'A' if freq>=0.02 else ('B' if freq>=0.01 else ('C' if freq>=0.005 else 'D'))

def adj(nums):
    out=set()
    for x in nums:
        if x>1: out.add(x-1)
        if x<43: out.add(x+1)
    return out-set(nums)

def profile_bounds(hist):
    # Broad structural guard only; not a predictive score.
    sums=sorted(sum(r) for r in hist)
    ranges=sorted(max(r)-min(r) for r in hist)
    n=len(hist)
    q=lambda a,p:a[min(n-1,max(0,int(round((n-1)*p))))]
    oddc=Counter(sum(x%2 for x in r) for r in hist)
    consec=sorted(sum(1 for a,b in zip(r,r[1:]) if b-a==1) for r in hist)
    common_odd={k for k,v in oddc.items() if v>=max(3,0.05*n)}
    return {'sum_lo':q(sums,.10),'sum_hi':q(sums,.90),'range_lo':q(ranges,.10),'range_hi':q(ranges,.90),
            'consec_hi':q(consec,.90),'odd_common':common_odd}

def prof_ok(row,p):
    su=sum(row); rg=row[-1]-row[0]; odd=sum(x%2 for x in row); con=sum(1 for a,b in zip(row,row[1:]) if b-a==1)
    return p['sum_lo']<=su<=p['sum_hi'] and p['range_lo']<=rg<=p['range_hi'] and odd in p['odd_common'] and con<=p['consec_hi']

def sig(row,S1,S2,S3,S4,OUTSIDE):
    R=set(row)
    return (len(R&S1),len(R&S2),len(R&S3),len(R&S4),len(R&OUTSIDE))

def main():
    rows=load_rows(); assert rows[-1][0]==2135, rows[-1][0]
    nums=[r[1] for r in rows]
    bonus=[r[2] for r in rows]
    prev=nums[-1]; prev2=nums[-2]; prev3to5=nums[-5:-2]
    S1=set(prev)
    S2=adj(S1)
    S3=set(prev2)|adj(set(prev2))
    S4=set().union(*(set(r) for r in prev3to5))
    FOUR=S1|S2|S3|S4
    OUTSIDE=set(range(1,44))-FOUR
    prevbo=bonus[-1]

    hall=Counter(band(r) for r in nums)
    c20=Counter(band(r) for r in nums[-20:])
    c50=Counter(band(r) for r in nums[-50:])
    shape_meta={}
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3; sh=(b0,b1,b2,b3,b4); f=hall[sh]/len(nums)
         shape_meta[sh]={'hist_count':hall[sh],'freq':f,'layer':layer(f),'c20':c20[sh],'c50':c50[sh],
                         'temporal_ok':c20[sh]==0 and c50[sh]<=1}
    prev_shape=band(prev); prev_layer=shape_meta[prev_shape]['layer']
    prof=profile_bounds(nums[-500:])

    counts=Counter(); counts_no_bo=Counter(); counts_profile=Counter(); counts_2324=Counter(); shape_counts=Counter()
    candidates=[]
    for row in itertools.combinations(range(1,44),6):
        R=set(row)
        c1=len(R&S1); c2=len(R&S2); c3=len(R&S3); c4=len(R&S4); co=len(R&OUTSIDE)
        if not (c1<=1 and c2==1 and 1<=c3<=4 and 1<=c4<=2 and co==1): continue
        sh=band(row); meta=shape_meta[sh]
        if not meta['temporal_ok']: continue
        ly=meta['layer']; counts_no_bo[ly]+=1
        if prevbo in R: continue
        counts[ly]+=1; shape_counts[(ly,sh)]+=1
        if 23 in R or 24 in R: counts_2324[ly]+=1
        if prof_ok(row,prof): counts_profile[ly]+=1
        if ly!='D':
            candidates.append({'row':row,'layer':ly,'shape':sh,'c1':c1,'c2':c2,'c3':c3,'c4':c4,
                               'outside':next(iter(R&OUTSIDE)),'cold':int(23 in R or 24 in R),'profile':prof_ok(row,prof)})

    # Prefer broad historical profile, but fall back to all A/B/C flow-valid combinations.
    maincand=[x for x in candidates if x['profile']]
    bylayer=defaultdict(list)
    for x in maincand: bylayer[x['layer']].append(x)
    quota={'A':4,'B':4,'C':2}
    # If a layer unexpectedly lacks enough profile candidates, allow non-profile from same layer.
    allby=defaultdict(list)
    for x in candidates: allby[x['layer']].append(x)
    for ly in quota:
        if len(bylayer[ly])<quota[ly]: bylayer[ly]=allby[ly]

    # Deterministic rule-based portfolio assembly. No Stat/Committee score.
    # Targets diversify previous overlap / 2-back flow / 3-5-back flow and minimize pair/triple reuse.
    target_c1=[0,0,0,0,0,0,1,1,1,1]
    target_c3=[1,1,2,2,2,2,3,3,4,1]
    target_c4=[1,2,1,2,1,2,1,2,1,2]
    target_layers=['A','B','A','B','C','A','B','C','A','B']
    selected=[]; numuse=Counter(); pairuse=Counter(); triuse=Counter(); cold_used=0
    for pos in range(10):
        ly=target_layers[pos]; pool=bylayer[ly]
        best=None; bestkey=None
        for x in pool:
            row=x['row']
            if any(row==y['row'] for y in selected): continue
            if x['c1']!=target_c1[pos]: continue
            # Keep 23/24 very limited across the final portfolio.
            if x['cold'] and cold_used>=1: continue
            pairs=list(itertools.combinations(row,2)); tris=list(itertools.combinations(row,3))
            # hard-ish diversity first; relax automatically through key rather than rejecting all.
            rep_tri=sum(triuse[t] for t in tris); rep_pair=sum(pairuse[p] for p in pairs); load=sum(numuse[n] for n in row)
            maxnum=max((numuse[n] for n in row),default=0)
            key=(abs(x['c3']-target_c3[pos]),abs(x['c4']-target_c4[pos]),rep_tri,rep_pair,maxnum,load,x['cold'],abs(sum(row)-132),row)
            if bestkey is None or key<bestkey: bestkey=key; best=x
        if best is None:
            # fallback within layer, relaxing c1 target only; still minimize diversity reuse.
            for x in pool:
                row=x['row']
                if any(row==y['row'] for y in selected): continue
                if x['cold'] and cold_used>=1: continue
                pairs=list(itertools.combinations(row,2)); tris=list(itertools.combinations(row,3))
                key=(abs(x['c1']-target_c1[pos]),abs(x['c3']-target_c3[pos]),abs(x['c4']-target_c4[pos]),
                     sum(triuse[t] for t in tris),sum(pairuse[p] for p in pairs),sum(numuse[n] for n in row),x['cold'],abs(sum(row)-132),row)
                if bestkey is None or key<bestkey: bestkey=key; best=x
        assert best is not None
        selected.append(best); cold_used+=best['cold']
        for n in best['row']: numuse[n]+=1
        for p in itertools.combinations(best['row'],2): pairuse[p]+=1
        for t in itertools.combinations(best['row'],3): triuse[t]+=1

    # audit
    union=sorted(set().union(*(set(x['row']) for x in selected)))
    layers=Counter(x['layer'] for x in selected)
    cold_lines=sum(x['cold'] for x in selected)
    prev_lines=Counter(x['c1'] for x in selected)
    out={
      'draw':2136,
      'history_last_draw':rows[-1][0],
      'recent_draws':{
        'prev_2135':list(prev),'prev2_2134':list(prev2),'prev3to5':[list(r) for r in prev3to5],'prev_bonus':prevbo,
        'prev_shape':list(prev_shape),'prev_layer':prev_layer},
      'sets':{'S1_prev_same':sorted(S1),'S2_prev_pm1':sorted(S2),'S3_prev2_same_pm1':sorted(S3),'S4_draws3to5_same':sorted(S4),
              'outside_four_sets':sorted(OUTSIDE)},
      'rules':{'prev_same':'0-1','prev_pm1':'exactly 1','prev2_same_pm1':'1-4','draws3to5_same':'1-2','outside_four_sets':'exactly 1',
               'band':'c20=0 and c50<=1','previous_bonus_excluded':prevbo,'D_layer_excluded_from_final':True,'23_24':'soft; max one final line'},
      'profile_tiebreak':{'sum':[prof['sum_lo'],prof['sum_hi']],'range':[prof['range_lo'],prof['range_hi']],'consec_max':prof['consec_hi'],'odd_common':sorted(prof['odd_common'])},
      'layer_counts_after_flow_before_D_exclusion':dict(counts),
      'layer_counts_without_previous_bonus_exclusion':dict(counts_no_bo),
      'layer_profile_counts':dict(counts_profile),
      'layer_counts_containing_23_or_24':dict(counts_2324),
      'eligible_total_before_D_exclusion':sum(counts.values()),
      'eligible_total_after_D_exclusion':sum(counts[x] for x in ('A','B','C')),
      'allowed_shape_count_by_layer':dict(Counter(v['layer'] for v in shape_meta.values() if v['temporal_ok'])),
      'top_shapes_by_layer':{},
      'portfolio':[],
      'portfolio_audit':{'layers':dict(layers),'union_size':len(union),'union':union,'cold23_24_lines':cold_lines,'prev_overlap_line_counts':dict(prev_lines),
                         'max_number_use':max(numuse.values()),'repeated_pairs':sum(v>1 for v in pairuse.values()),'repeated_triples':sum(v>1 for v in triuse.values())}
    }
    for ly in ('A','B','C','D'):
        arr=[(cnt,sh,shape_meta[sh]) for (l,sh),cnt in shape_counts.items() if l==ly]
        arr.sort(reverse=True)
        out['top_shapes_by_layer'][ly]=[{'shape':list(sh),'candidate_count':cnt,'hist_freq':m['freq'],'hist_count':m['hist_count'],'c20':m['c20'],'c50':m['c50']} for cnt,sh,m in arr[:12]]
    for i,x in enumerate(selected,1):
        out['portfolio'].append({'no':i,'numbers':list(x['row']),'sum':sum(x['row']),'layer':x['layer'],'shape':list(x['shape']),
                                 'prev_same':x['c1'],'prev_pm1':x['c2'],'prev2_same_pm1':x['c3'],'draws3to5_same':x['c4'],
                                 'outside_number':x['outside'],'contains_23_24':bool(x['cold']),'profile':bool(x['profile'])})
    path=OUT/'loto6_2136_flow_exact.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
