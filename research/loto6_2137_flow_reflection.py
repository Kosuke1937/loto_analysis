#!/usr/bin/env python3
from __future__ import annotations
import itertools, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
BANDS=((1,9),(10,19),(20,29),(30,39),(40,43))

def sum_state(s):
    if s<=109: return 'S1'
    if s<=129: return 'S2'
    if s<=149: return 'S3'
    return 'S4'

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

def has_three_consecutive(row):
    return any(row[i+1]==row[i]+1 and row[i+2]==row[i]+2 for i in range(len(row)-2))

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

def shape_meta(hist):
    hall=Counter(band(r) for r in hist)
    c20=Counter(band(r) for r in hist[-20:])
    c50=Counter(band(r) for r in hist[-50:])
    out={}
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3
         sh=(b0,b1,b2,b3,b4); f=hall[sh]/len(hist)
         out[sh]={'count':hall[sh],'freq':f,'layer':layer(f),'c20':c20[sh],'c50':c50[sh],
                  'temporal_ok':c20[sh]==0 and c50[sh]<=1}
    return out

def flow_sets(hist):
    prev=hist[-1]; prev2=hist[-2]; prev3to5=hist[-5:-2]
    f1=set(prev)
    f2=adj(f1)
    f3=set(prev2)|adj(set(prev2))
    f4=set().union(*(set(r) for r in prev3to5))
    outside=set(range(1,44))-(f1|f2|f3|f4)
    return f1,f2,f3,f4,outside

def audit_target(hist_before,target,target_bonus_prev):
    f1,f2,f3,f4,outside=flow_sets(hist_before)
    R=set(target); sm=shape_meta(hist_before); sh=band(target); m=sm[sh]
    return {
      'numbers':list(target),'sum':sum(target),'sum_state':sum_state(sum(target)),'shape':list(sh),'layer':m['layer'],
      'shape_prior20':m['c20'],'shape_prior50':m['c50'],'shape_temporal_ok':m['temporal_ok'],
      'prev_same':len(R&f1),'prev_pm1':len(R&f2),'prev2_same_pm1':len(R&f3),'draws3to5_same':len(R&f4),
      'outside_four_count':len(R&outside),'outside_four_numbers':sorted(R&outside),
      'count_1_31':sum(x<=31 for x in target),'count_32_43':sum(x>=32 for x in target),
      'previous_bonus_in_main':target_bonus_prev in R,
      'contains_23_24':bool(R&{23,24}),'three_consecutive':has_three_consecutive(target),
    }

def transition_counts(hist, source='S4'):
    c=Counter()
    for i in range(1,len(hist)):
        if sum_state(sum(hist[i-1]))==source:
            c[sum_state(sum(hist[i]))]+=1
    return c

def main():
    rows=load_rows(); assert rows[-1][0]==2136, rows[-1][0]
    nums=[r[1] for r in rows]; bonuses=[r[2] for r in rows]

    audit2136=audit_target(nums[:-1],nums[-1],bonuses[-2])

    prev=nums[-1]; prevbo=bonuses[-1]
    f1,f2,f3,f4,outside=flow_sets(nums)
    sm=shape_meta(nums)
    prev_sh=band(prev); prev_layer=sm[prev_sh]['layer']

    counts=Counter(); candidates=[]
    for row in itertools.combinations(range(1,44),6):
        if 23 in row or 24 in row: continue
        if has_three_consecutive(row): continue
        R=set(row)
        c1=len(R&f1); c2=len(R&f2); c3=len(R&f3); c4=len(R&f4); co=len(R&outside)
        if not (0<=c1<=2 and 1<=c2<=2 and 1<=c3<=4 and 1<=c4<=2 and co==1): continue
        low31=sum(x<=31 for x in row); high=6-low31
        if low31<2 or high<1: continue
        sh=band(row); m=sm[sh]
        if not m['temporal_ok']: continue
        ly=m['layer']
        if prev_layer=='D' and ly=='D': continue
        st=sum_state(sum(row))
        if st=='S3': continue
        has_bo=int(prevbo in R)
        counts[(st,ly,c1,c2,has_bo)]+=1
        candidates.append({'row':row,'state':st,'layer':ly,'shape':sh,'c1':c1,'c2':c2,'c3':c3,'c4':c4,
                           'outside':next(iter(R&outside)),'has_bo':has_bo,'low31':low31,'high':high})

    state_plan=['S4']*5+['S2']*3+['S1']*2
    layer_plan=['A','B','C','A','B','A','B','C','A','B']
    c1_plan=[0,0,1,1,2,0,1,2,0,1]
    c2_plan=[1,1,1,1,2,1,1,2,1,1]
    bo_plan=[0,0,0,0,0,0,0,0,0,1]
    c3_target=[2,2,3,1,3,2,3,2,1,3]
    c4_target=[2,1,2,1,2,1,2,1,2,1]

    selected=[]; numuse=Counter(); pairuse=Counter(); triuse=Counter(); prevuse=Counter()

    def candidate_key(x,pos,layer_exact=True,c1_exact=True,c2_exact=True):
        row=x['row']; pairs=list(itertools.combinations(row,2)); tris=list(itertools.combinations(row,3))
        return (
          0 if (layer_exact and x['layer']==layer_plan[pos]) else (0 if not layer_exact else 1),
          0 if (c1_exact and x['c1']==c1_plan[pos]) else (0 if not c1_exact else 1),
          0 if (c2_exact and x['c2']==c2_plan[pos]) else (0 if not c2_exact else 1),
          abs(x['c3']-c3_target[pos]),abs(x['c4']-c4_target[pos]),
          sum(triuse[t] for t in tris),sum(pairuse[p] for p in pairs),
          max((numuse[n] for n in row),default=0),sum(numuse[n] for n in row),
          abs(sum(row)-({'S4':158,'S2':119,'S1':101}[x['state']])),row
        )

    for pos,st in enumerate(state_plan):
        best=None; bestkey=None
        for relax in range(3):
            for x in candidates:
                if x['state']!=st: continue
                row=x['row']
                if any(row==y['row'] for y in selected): continue
                if relax<2 and x['c1']!=c1_plan[pos]: continue
                if relax<2 and x['c2']!=c2_plan[pos]: continue
                if x['has_bo']!=bo_plan[pos]: continue
                if relax==0 and x['layer']!=layer_plan[pos]: continue
                prevnums=[n for n in row if n in f1]
                if any(prevuse[n]>=2 for n in prevnums): continue
                k=candidate_key(x,pos,layer_exact=(relax==0),c1_exact=(relax<2),c2_exact=(relax<2))
                if bestkey is None or k<bestkey: bestkey=k; best=x
            if best is not None: break
        assert best is not None,(pos,st,layer_plan[pos],c1_plan[pos],c2_plan[pos],bo_plan[pos])
        selected.append(best)
        for n in best['row']:
            numuse[n]+=1
            if n in f1: prevuse[n]+=1
        for p in itertools.combinations(best['row'],2): pairuse[p]+=1
        for t in itertools.combinations(best['row'],3): triuse[t]+=1

    out={
      'draw':2137,
      'history_last_draw':2136,
      'postmortem_2136':audit2136,
      'previous':{'draw':2136,'numbers':list(prev),'bonus':prevbo,'sum':sum(prev),'sum_state':sum_state(sum(prev)),
                  'shape':list(prev_sh),'layer':prev_layer},
      'flow_sets':{'prev_same':sorted(f1),'prev_pm1':sorted(f2),'prev2_same_pm1':sorted(f3),
                   'draws3to5_same':sorted(f4),'outside_four':sorted(outside)},
      's4_transitions_all_history':dict(transition_counts(nums,'S4')),
      'rules':{
        'kept':['outside_four exactly 1','prev2 same/pm1 1-4','draws3to5 same 1-2','1-31 >=2','32-43 >=1',
                'band c20=0 and c50<=1','23/24 excluded','3-consecutive excluded','2-consecutive allowed'],
        'adjusted':['prev same main scenario 0-1 + 2 rescue','prev +/-1 main scenario 1 + 2 rescue',
                    'previous bonus: 9 exclude + 1 rescue'],
        'sum_portfolio':'S4 5 / S2 3 / S1 2 / S3 0',
        'D_after_D':'excluded if previous layer is D'
      },
      'portfolio':[],
      'audit':{}
    }
    for i,x in enumerate(selected,1):
        out['portfolio'].append({'no':i,'numbers':list(x['row']),'sum':sum(x['row']),'sum_state':x['state'],
          'layer':x['layer'],'shape':list(x['shape']),'prev_same':x['c1'],'prev_pm1':x['c2'],
          'prev2_same_pm1':x['c3'],'draws3to5_same':x['c4'],'outside_number':x['outside'],
          'previous_bonus_included':bool(x['has_bo']),'count_1_31':x['low31'],'count_32_43':x['high']})
    union=sorted(set().union(*(set(x['row']) for x in selected)))
    out['audit']={
      'state_counts':dict(Counter(x['state'] for x in selected)),
      'layer_counts':dict(Counter(x['layer'] for x in selected)),
      'prev_same_counts':dict(Counter(x['c1'] for x in selected)),
      'prev_pm1_counts':dict(Counter(x['c2'] for x in selected)),
      'previous_bonus_lines':sum(x['has_bo'] for x in selected),
      'outside_failures':sum(len(set(x['row'])&outside)!=1 for x in selected),
      'three_consecutive_lines':sum(has_three_consecutive(x['row']) for x in selected),
      'excluded_23_24_present':sum(bool(set(x['row'])&{23,24}) for x in selected),
      'union_size':len(union),'union':union,
      'repeated_pairs':sum(v>1 for v in pairuse.values()),'repeated_triples':sum(v>1 for v in triuse.values()),
      'previous_number_use':{str(k):v for k,v in sorted(prevuse.items())}
    }
    path=OUT/'loto6_2137_flow_reflection.json'
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
