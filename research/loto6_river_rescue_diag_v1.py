#!/usr/bin/env python3
from __future__ import annotations
import json, math, re
from pathlib import Path
from collections import Counter, defaultdict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)


def load_history():
    rows=[]
    for p in sorted((ROOT/'data').glob('loto6-chunk-*.js'), key=lambda x:int(re.search(r'(\d+)',x.stem).group(1))):
        s=p.read_text(encoding='utf-8')
        m=re.search(r'push\((.*)\);\s*$',s,re.S)
        if not m: continue
        rows.extend(json.loads(m.group(1)))
    rows.sort(key=lambda r:r[0])
    return [{'draw':r[0],'date':r[1],'nums':list(map(int,r[2:8])),'bonus':r[8]} for r in rows]

def band(n):
    if n<=9:return 0
    if n<=19:return 1
    if n<=29:return 2
    if n<=39:return 3
    return 4

def min_dist(n, nums):
    return min(abs(n-x) for x in nums) if nums else 99

def rel_flags(H,t,n):
    p1=H[t-1]['nums']; p2=H[t-2]['nums']; p35=sum((H[t-k]['nums'] for k in (3,4,5)),[])
    same=n in p1
    near=any(abs(n-x)==1 for x in p1)
    p2near=any(abs(n-x)<=1 for x in p2)
    p35same=n in p35
    outside=not (same or near or p2near or p35same)
    return {'prevSame':same,'prevNear':near,'prev2Near':p2near,'prev35Same':p35same,'outside':outside}

def same_gap(H,t,n,cap=100):
    for g in range(1,min(cap,t)+1):
        if n in H[t-g]['nums']: return g
    return cap+1

def local_density(H,t,n,w,rad):
    c=0
    for u in range(max(0,t-w),t):
        c += sum(abs(x-n)<=rad for x in H[u]['nums'])
    return c

def local_silence(H,t,n,rad,cap=30):
    g=0
    for u in range(t-1,max(-1,t-cap-1),-1):
        if any(abs(x-n)<=rad for x in H[u]['nums']): return g
        g+=1
    return g

def band_density(H,t,n,w):
    b=band(n); c=0
    for u in range(max(0,t-w),t): c+=sum(band(x)==b for x in H[u]['nums'])
    return c

def center_slope(H,t,w=5):
    vals=[sum(H[u]['nums'])/6 for u in range(max(0,t-w),t)]
    if len(vals)<2:return vals[-1] if vals else 22,0.0
    xs=list(range(len(vals))); xm=sum(xs)/len(xs); ym=sum(vals)/len(vals)
    den=sum((x-xm)**2 for x in xs) or 1.0
    slope=sum((x-xm)*(y-ym) for x,y in zip(xs,vals))/den
    return vals[-1],slope

def feat(H,t,n):
    p1=H[t-1]['nums']; p2=H[t-2]['nums']; p35=sum((H[t-k]['nums'] for k in (3,4,5)),[]); p5=sum((H[t-k]['nums'] for k in range(1,6)),[])
    c0,sl=center_slope(H,t,5); proj=c0+sl
    b=band(n)
    last3=sum(sum(band(x)==b for x in H[u]['nums']) for u in range(t-3,t))
    prev3=sum(sum(band(x)==b for x in H[u]['nums']) for u in range(t-6,t-3))
    direction=1 if sl>0.35 else (-1 if sl<-0.35 else 0)
    ahead=(n-c0)*direction if direction else 0.0
    return {
        'same_gap':same_gap(H,t,n,100),
        'd_prev1':min_dist(n,p1),
        'd_prev2':min_dist(n,p2),
        'd_prev35':min_dist(n,p35),
        'd_prev5_union':min_dist(n,p5),
        'local_silence2':local_silence(H,t,n,2,30),
        'local_silence4':local_silence(H,t,n,4,30),
        'local_density2_10':local_density(H,t,n,10,2),
        'local_density4_10':local_density(H,t,n,10,4),
        'band_density5':band_density(H,t,n,5),
        'band_momentum3':last3-prev3,
        'center_dist':abs(n-c0),
        'projected_center_dist':abs(n-proj),
        'macro_ahead':ahead,
    }

def auc(vals):
    pos=[s for s,y in vals if y]; neg=[s for s,y in vals if not y]
    if not pos or not neg:return None
    arr=sorted([(s,y) for s,y in vals], key=lambda z:z[0])
    ranks=[]; i=0; rank=1
    rank_sum=0.0
    while i<len(arr):
        j=i+1
        while j<len(arr) and arr[j][0]==arr[i][0]: j+=1
        avg=(rank + (rank+(j-i)-1))/2
        for k in range(i,j):
            if arr[k][1]: rank_sum+=avg
        rank += (j-i); i=j
    np=len(pos); nn=len(neg)
    return (rank_sum - np*(np+1)/2)/(np*nn)

def topk_capture(records, feature, sign=1, k=5):
    by=defaultdict(list)
    for r in records: by[r['draw']].append(r)
    hit=tot=0; expected=0.0
    for draw,rs in by.items():
        wins=sum(r['y'] for r in rs)
        if not wins: continue
        kk=min(k,len(rs)); top=sorted(rs,key=lambda r:sign*r[feature], reverse=True)[:kk]
        hit += sum(r['y'] for r in top); tot += wins
        expected += wins*kk/len(rs)
    return {'captured':hit,'total_outside_winners':tot,'recall':hit/tot if tot else 0,'random_expected':expected/tot if tot else 0,'lift':(hit/expected if expected else None)}

def bin_report(records, getter):
    bins=defaultdict(lambda:[0,0])
    for r in records:
        key=getter(r); bins[key][0]+=1; bins[key][1]+=int(r['y'])
    base=sum(v[1] for v in bins.values())/max(1,sum(v[0] for v in bins.values()))
    return {str(k):{'candidates':v[0],'winners':v[1],'rate':v[1]/v[0] if v[0] else 0,'lift_vs_pool':(v[1]/v[0]/base if v[0] and base else None)} for k,v in sorted(bins.items(),key=lambda kv:str(kv[0]))}

def future_growth(H,t,n,rad):
    if t+5>=len(H): return None
    past=sum(sum(abs(x-n)<=rad for x in H[u]['nums']) for u in range(t-5,t))
    fut=sum(sum(abs(x-n)<=rad for x in H[u]['nums']) for u in range(t+1,t+6))
    return fut-past

def main():
    H=load_history(); assert len(H)>=1000
    start=len(H)-1000; dev_end=start+500; test_start=dev_end
    records=[]; flow={'outside':[],'inside':[]}
    coverage=[]
    for t in range(start,len(H)):
        outside=[n for n in range(1,44) if rel_flags(H,t,n)['outside']]
        W=set(H[t]['nums']); ow=[n for n in W if n in outside]
        coverage.append({'draw':H[t]['draw'],'outside_candidates':len(outside),'outside_winners':len(ow)})
        part='dev' if t<dev_end else 'test'
        for n in outside:
            f=feat(H,t,n); records.append({'draw':H[t]['draw'],'part':part,'n':n,'y':n in W,**f})
        if t+5<len(H):
            for n in W:
                target='outside' if n in outside else 'inside'
                g2=future_growth(H,t,n,2); g4=future_growth(H,t,n,4)
                flow[target].append((g2,g4))
    dev=[r for r in records if r['part']=='dev']; test=[r for r in records if r['part']=='test']
    feats=['same_gap','d_prev1','d_prev2','d_prev35','d_prev5_union','local_silence2','local_silence4','local_density2_10','local_density4_10','band_density5','band_momentum3','center_dist','projected_center_dist','macro_ahead']
    feature_eval={}
    for f in feats:
        ad=auc([(r[f],r['y']) for r in dev]); sign=1 if (ad is None or ad>=.5) else -1
        at=auc([(sign*r[f],r['y']) for r in test]); cap=topk_capture(test,f,sign,5)
        feature_eval[f]={'dev_auc_raw':ad,'direction':'high' if sign==1 else 'low','test_auc_direction_fixed':at,'test_top5':cap}
    # simple dev-trained weighted blend from directional AUC strength; per-feature global z from dev
    chosen=[]; stats={}
    for f in feats:
        a=feature_eval[f]['dev_auc_raw'];
        if a is None: continue
        s=1 if a>=.5 else -1; strength=abs(a-.5)
        if strength>=.015:
            xs=[r[f] for r in dev]; mu=sum(xs)/len(xs); sd=(sum((x-mu)**2 for x in xs)/len(xs))**.5 or 1
            chosen.append((f,s,strength)); stats[f]=(mu,sd)
    for r in test:
        r['blend']=sum(w*s*((r[f]-stats[f][0])/stats[f][1]) for f,s,w in chosen)
    blend_auc=auc([(r['blend'],r['y']) for r in test]) if chosen else None
    blend_cap=topk_capture(test,'blend',1,5) if chosen else None
    cov_all={'draws':len(coverage),'mean_outside_candidates':sum(x['outside_candidates'] for x in coverage)/len(coverage),
             'outside_winners':sum(x['outside_winners'] for x in coverage),'winner_numbers':6*len(coverage),
             'outside_winner_share':sum(x['outside_winners'] for x in coverage)/(6*len(coverage)),
             'draws_with_outside':sum(x['outside_winners']>0 for x in coverage)}
    def flow_summary(arr):
        if not arr:return {}
        return {'n':len(arr),'mean_growth_pm2':sum(x[0] for x in arr)/len(arr),'growth_pm2_ge2':sum(x[0]>=2 for x in arr)/len(arr),
                'mean_growth_pm4':sum(x[1] for x in arr)/len(arr),'growth_pm4_ge2':sum(x[1]>=2 for x in arr)/len(arr)}
    bins={
      'same_gap':bin_report(test,lambda r:'6-7' if r['same_gap']<=7 else '8-10' if r['same_gap']<=10 else '11-15' if r['same_gap']<=15 else '16-25' if r['same_gap']<=25 else '26+'),
      'd_prev1':bin_report(test,lambda r:'2' if r['d_prev1']==2 else '3' if r['d_prev1']==3 else '4' if r['d_prev1']==4 else '5+'),
      'd_prev2':bin_report(test,lambda r:'2' if r['d_prev2']==2 else '3' if r['d_prev2']==3 else '4' if r['d_prev2']==4 else '5+'),
      'local_silence2':bin_report(test,lambda r:'0' if r['local_silence2']==0 else '1' if r['local_silence2']==1 else '2' if r['local_silence2']==2 else '3+'),
      'band_momentum3':bin_report(test,lambda r:'<=-2' if r['band_momentum3']<=-2 else '-1' if r['band_momentum3']==-1 else '0' if r['band_momentum3']==0 else '1' if r['band_momentum3']==1 else '>=2'),
    }
    out={'method':'Loto6 River outside-number Rescue Diagnostic v1','latest_draw':H[-1]['draw'],
         'eval_range':f"{H[start]['draw']}-{H[-1]['draw']} (1000 draws)",'dev_range':f"{H[start]['draw']}-{H[dev_end-1]['draw']} (500)",'test_range':f"{H[test_start]['draw']}-{H[-1]['draw']} (500)",
         'outside_definition':'not prevSame, not prev±1, not prev2 same/±1, not same in prev3-5',
         'coverage':cov_all,'feature_eval':feature_eval,'blend':{'features':[x[0] for x in chosen],'test_auc':blend_auc,'test_top5':blend_cap},
         'test_bins':bins,'future_flow_outcome':{'outside_winners':flow_summary(flow['outside']),'inside_winners':flow_summary(flow['inside']),
           'note':'future windows are outcome labels only, never used as pre-draw features'},
         'coverage_detail':coverage}
    path=OUT/'loto6_river_rescue_diag_v1_summary.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='coverage_detail'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
