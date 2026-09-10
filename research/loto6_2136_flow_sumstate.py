#!/usr/bin/env python3
from __future__ import annotations
import itertools, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
BANDS=((1,9),(10,19),(20,29),(30,39),(40,43))

# Sum-state definition used for #2136.
def sum_state(s):
    if s<=105: return 'S1'
    if s<=120: return 'S2'
    if s<=145: return 'S3'
    return 'S4'

def has_three_consecutive(row):
    """Allow ordinary consecutive pairs, but reject any run of 3+ consecutive numbers."""
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
    sums=sorted(sum(r) for r in hist); ranges=sorted(max(r)-min(r) for r in hist); n=len(hist)
    q=lambda a,p:a[min(n-1,max(0,int(round((n-1)*p))))]
    oddc=Counter(sum(x%2 for x in r) for r in hist)
    consec=sorted(sum(1 for a,b in zip(r,r[1:]) if b-a==1) for r in hist)
    common_odd={k for k,v in oddc.items() if v>=max(3,0.05*n)}
    return {'sum_lo':q(sums,.10),'sum_hi':q(sums,.90),'range_lo':q(ranges,.10),'range_hi':q(ranges,.90),
            'consec_hi':q(consec,.90),'odd_common':common_odd}

def prof_ok(row,p):
    su=sum(row); rg=row[-1]-row[0]; odd=sum(x%2 for x in row); con=sum(1 for a,b in zip(row,row[1:]) if b-a==1)
    return p['sum_lo']<=su<=p['sum_hi'] and p['range_lo']<=rg<=p['range_hi'] and odd in p['odd_common'] and con<=p['consec_hi']

def transition_counts(nums, start=0):
    c=Counter()
    for i in range(max(1,start),len(nums)):
        a=sum_state(sum(nums[i-1])); b=sum_state(sum(nums[i]))
        if a=='S4': c[b]+=1
    return c

def main():
    rows=load_rows(); assert rows[-1][0]==2135, rows[-1][0]
    nums=[r[1] for r in rows]; bonus=[r[2] for r in rows]
    prev=nums[-1]; prev2=nums[-2]; prev3to5=nums[-5:-2]; prevbo=bonus[-1]
    assert sum_state(sum(prev))=='S4'

    S1=set(prev); S2=adj(S1); S3=set(prev2)|adj(set(prev2)); S4=set().union(*(set(r) for r in prev3to5))
    OUTSIDE=set(range(1,44))-(S1|S2|S3|S4)

    hall=Counter(band(r) for r in nums); c20=Counter(band(r) for r in nums[-20:]); c50=Counter(band(r) for r in nums[-50:])
    shape_meta={}
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3; sh=(b0,b1,b2,b3,b4); f=hall[sh]/len(nums)
         shape_meta[sh]={'hist_count':hall[sh],'freq':f,'layer':layer(f),'c20':c20[sh],'c50':c50[sh],
                         'temporal_ok':c20[sh]==0 and c50[sh]<=1}
    prof=profile_bounds(nums[-500:])

    counts=Counter(); candidates=[]
    for row in itertools.combinations(range(1,44),6):
        if has_three_consecutive(row): continue
        R=set(row); c1=len(R&S1); c2=len(R&S2); c3=len(R&S3); c4=len(R&S4); co=len(R&OUTSIDE)
        if not (c1<=1 and c2==1 and 1<=c3<=4 and 1<=c4<=2 and co==1): continue
        sh=band(row); meta=shape_meta[sh]
        if not meta['temporal_ok']: continue
        low31=sum(x<=31 for x in row); high3243=6-low31
        if low31<2 or high3243<1: continue
        if prevbo in R: continue
        ly=meta['layer']
        if ly=='D': continue
        st=sum_state(sum(row))
        counts[(st,ly)]+=1
        candidates.append({'row':row,'state':st,'layer':ly,'shape':sh,'c1':c1,'c2':c2,'c3':c3,'c4':c4,
                           'outside':next(iter(R&OUTSIDE)),'cold':int(23 in R or 24 in R),'profile':prof_ok(row,prof),
                           'low31':low31,'high3243':high3243})

    state_plan=['S4']*5+['S2']*3+['S1']*2
    layer_plan=['A','B','C','A','B','A','B','C','A','B']
    c1_plan=[0,0,0,1,1,0,0,1,0,1]
    c3_plan=[2,2,3,1,4,2,3,2,1,3]
    c4_plan=[2,1,2,1,2,1,2,1,2,1]

    pools=defaultdict(list)
    for x in candidates: pools[(x['state'],x['layer'])].append(x)
    selected=[]; numuse=Counter(); pairuse=Counter(); triuse=Counter(); cold_used=0; prev_num_use=Counter()

    def key_for(x,pos,relax_layer=False,relax_c1=False):
        row=x['row']; pairs=list(itertools.combinations(row,2)); tris=list(itertools.combinations(row,3))
        prevn=[n for n in row if n in S1]
        prev_over=max((prev_num_use[n] for n in prevn),default=0)
        return (0 if x['profile'] else 1,
                0 if (relax_c1 or x['c1']==c1_plan[pos]) else 1,
                abs(x['c3']-c3_plan[pos]),abs(x['c4']-c4_plan[pos]),
                sum(triuse[t] for t in tris),sum(pairuse[p] for p in pairs),
                prev_over,max((numuse[n] for n in row),default=0),sum(numuse[n] for n in row),
                x['cold'],abs(sum(row)-({'S4':157,'S2':113,'S1':100}[x['state']])),row)

    for pos,(st,ly) in enumerate(zip(state_plan,layer_plan)):
        options=pools[(st,ly)]
        best=None; bestkey=None
        for x in options:
            row=x['row']
            if any(row==y['row'] for y in selected): continue
            if x['c1']!=c1_plan[pos]: continue
            if x['cold'] and cold_used>=1: continue
            prevn=[n for n in row if n in S1]
            if any(prev_num_use[n]>=2 for n in prevn): continue
            k=key_for(x,pos)
            if bestkey is None or k<bestkey: bestkey=k; best=x
        if best is None:
            for x in candidates:
                if x['state']!=st: continue
                row=x['row']
                if any(row==y['row'] for y in selected): continue
                if x['c1']!=c1_plan[pos]: continue
                if x['cold'] and cold_used>=1: continue
                prevn=[n for n in row if n in S1]
                if any(prev_num_use[n]>=2 for n in prevn): continue
                k=(0 if x['layer']==ly else 1,)+key_for(x,pos,relax_layer=True)
                if bestkey is None or k<bestkey: bestkey=k; best=x
        if best is None:
            for x in candidates:
                if x['state']!=st: continue
                row=x['row']
                if any(row==y['row'] for y in selected): continue
                if x['cold'] and cold_used>=1: continue
                prevn=[n for n in row if n in S1]
                if any(prev_num_use[n]>=2 for n in prevn): continue
                k=(0 if x['layer']==ly else 1,)+key_for(x,pos,relax_layer=True,relax_c1=True)
                if bestkey is None or k<bestkey: bestkey=k; best=x
        assert best is not None,(pos,st,ly)
        selected.append(best); cold_used+=best['cold']
        for n in best['row']:
            numuse[n]+=1
            if n in S1: prev_num_use[n]+=1
        for p in itertools.combinations(best['row'],2): pairuse[p]+=1
        for t in itertools.combinations(best['row'],3): triuse[t]+=1

    state_counts=Counter(x['state'] for x in selected); layer_counts=Counter(x['layer'] for x in selected); c1_counts=Counter(x['c1'] for x in selected)
    union=sorted(set().union(*(set(x['row']) for x in selected)))
    out={
      'draw':2136,
      'sum_state_definition':{'S1':'<=105','S2':'106-120','S3':'121-145','S4':'>=146'},
      'previous':{'draw':2135,'numbers':list(prev),'bonus':prevbo,'sum':sum(prev),'sum_state':'S4'},
      's4_transition_counts_all_history':dict(transition_counts(nums)),
      's4_transition_counts_recent500':dict(transition_counts(nums,max(1,len(nums)-500))),
      'user_priority':['S4','S2','S1'],
      'portfolio_state_plan':{'S4':5,'S2':3,'S1':2,'S3':0},
      'rules':{'three_consecutive':'excluded','ordinary_consecutive_pair':'allowed'},
      'eligible_counts_by_state_layer':{st:{ly:counts[(st,ly)] for ly in ('A','B','C')} for st in ('S1','S2','S3','S4')},
      'eligible_counts_by_state':{st:sum(counts[(st,ly)] for ly in ('A','B','C')) for st in ('S1','S2','S3','S4')},
      'portfolio':[],
      'portfolio_audit':{'states':dict(state_counts),'layers':dict(layer_counts),'prev_overlap':dict(c1_counts),
                         'prev_number_use':{str(k):v for k,v in sorted(prev_num_use.items())},'cold23_24_lines':cold_used,
                         'union_size':len(union),'union':union,'repeated_pairs':sum(v>1 for v in pairuse.values()),
                         'repeated_triples':sum(v>1 for v in triuse.values()),
                         'three_consecutive_lines':sum(has_three_consecutive(x['row']) for x in selected)},
      'caution':'5/3/2 is a portfolio allocation reflecting the user-provided order S4>S2>S1; it is not interpreted as calibrated lottery probability.'
    }
    for i,x in enumerate(selected,1):
        out['portfolio'].append({'no':i,'numbers':list(x['row']),'sum':sum(x['row']),'sum_state':x['state'],'layer':x['layer'],
                                 'shape':list(x['shape']),'prev_same':x['c1'],'prev_pm1':x['c2'],'prev2_same_pm1':x['c3'],
                                 'draws3to5_same':x['c4'],'outside_number':x['outside'],'contains_23_24':bool(x['cold']),
                                 'count_1_31':x['low31'],'count_32_43':x['high3243']})
    path=OUT/'loto6_2136_flow_sumstate.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
