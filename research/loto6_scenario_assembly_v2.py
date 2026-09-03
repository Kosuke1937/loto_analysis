#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v1',ROOT/'research'/'loto6_portfolio_multilane_v1.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
p=v1.p
v4=v1.v4
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877

PAIR_POS=list(itertools.combinations(range(6),2))
SUB4=list(itertools.combinations(range(6),4))
SUB5=list(itertools.combinations(range(6),5))

def z(x):
    x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)

def band_class(freq):
    if freq>=0.025:return 'A'
    if freq>=0.010:return 'B'
    if freq>=0.005:return 'C'
    return 'D'

def shape_freq(draws,t,w=500):
    c=Counter(v4.band_tuple(draws[u]) for u in range(max(0,t-w),t));n=max(1,min(w,t))
    return {k:v/n for k,v in c.items()}

def make_scores(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors):
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
    pc=ppref[t]-ppref[max(0,t-300)]
    c5,c4=p.cores(C,pc)
    comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    ns=v1.num_support(agents,C,topn=1500)
    return ss,pc,comm,ns

def candidate_pool(base,ns):
    U=set(x for t in base for x in t)
    # Low/Mid rescue: add four high-support 1..29 numbers absent from the current 10-ticket union.
    lowextra=[x for x in sorted(range(1,30),key=lambda n:(ns[n],-n),reverse=True) if x not in U][:4]
    A=sorted(U|set(lowextra))
    pool=set(tuple(sorted(map(int,t))) for t in base)
    # Recombine stable 5- and 4-number fragments from the six strongest base tickets.
    for t in base[:6]:
        tt=tuple(map(int,t))
        for core in itertools.combinations(tt,5):
            rem=[x for x in A if x not in core]
            for x in rem: pool.add(tuple(sorted(core+(x,))))
        for core in itertools.combinations(tt,4):
            rem=[x for x in A if x not in core]
            for x,y in itertools.combinations(rem,2): pool.add(tuple(sorted(core+(x,y))))
    return np.array(sorted(pool),dtype=np.int16),sorted(U),lowextra

def core_features(arr,pc):
    n=len(arr)
    pair_total=np.zeros(n,float)
    for i,j in PAIR_POS: pair_total += pc[arr[:,i],arr[:,j]]
    c4=np.full(n,-1e18,float)
    for sub in SUB4:
        s=np.zeros(n,float)
        for a,b in itertools.combinations(sub,2): s += pc[arr[:,a],arr[:,b]]
        c4=np.maximum(c4,s)
    c5=np.full(n,-1e18,float)
    for sub in SUB5:
        s=np.zeros(n,float)
        for a,b in itertools.combinations(sub,2): s += pc[arr[:,a],arr[:,b]]
        c5=np.maximum(c5,s)
    return pair_total,c4,c5

def scenario_scores(arr,ns,pc,sf):
    nz=(ns-np.mean(ns[1:]))/(np.std(ns[1:])+1e-9)
    num=nz[arr].sum(1)
    pair=np.zeros(len(arr),float)
    for i,j in PAIR_POS: pair += pc[arr[:,i],arr[:,j]]
    fast=.45*z(num)+.55*z(pair)
    keep=np.argsort(-fast)[:min(8000,len(arr))]
    arr=arr[keep];num=num[keep];pair=pair[keep]
    _,c4,c5=core_features(arr,pc)
    generic=.25*z(num)+.20*z(pair)+.20*z(c4)+.35*z(c5)
    sums=arr.sum(1)
    n30=(arr>=30).sum(1);n40=(arr>=40).sum(1)
    shapes=[v4.band_tuple(r) for r in arr]
    freqs=np.array([sf.get(s,0.0) for s in shapes])
    cls=np.array([band_class(f) for f in freqs])
    used=np.array([sum(x>0 for x in s) for s in shapes])
    fit_high=-np.abs(sums-160)/20.0
    fit_bal=-np.abs(sums-140)/18.0
    fit_low=-np.abs(sums-115)/15.0
    high=generic+.30*fit_high+.20*((n30>=3)&(n30<=5))+.15*((n40>=1)&(n40<=2))+.12*np.isin(cls,['B','D'])
    bal=generic+.28*fit_bal+.22*(used>=4)+.12*np.isin(cls,['A','B'])
    low=generic+.30*fit_low+.30*(n30<=1)+.10*np.isin(cls,['B','C','D'])
    rare=generic+.30*(-np.abs(sums-165)/22.0)+.35*(cls=='D')+.20*(n30>=3)
    explore=generic+.08*z(np.abs(sums-140))
    meta={'sum':sums,'shape':shapes,'class':cls,'used':used,'n30':n30,'n40':n40}
    return arr,{'high':high,'balanced':bal,'low':low,'rareD':rare,'explorer':explore},meta

def pick_lane(arr,score,n,sel,max_common=4):
    out=[]
    for i in np.argsort(-score):
        c=tuple(map(int,arr[int(i)]));cs=set(c)
        if c in sel or c in out:continue
        if any(len(cs&set(x))>max_common for x in sel+out):continue
        out.append(c)
        if len(out)>=n:break
    return out

def assembled10(base,arr,scores):
    sel=[]
    plan=[('high',3,4),('balanced',3,4),('low',2,4),('rareD',1,3),('explorer',1,3)]
    for lane,n,mc in plan: sel+=pick_lane(arr,scores[lane],n,sel,mc)
    if len(sel)<10: sel+=pick_lane(arr,scores['explorer'],10-len(sel),sel,4)
    return sel[:10]

def hybrid10(base,arr,scores):
    # Preserve four current Main tickets; replace six tickets with scenario-aware recombinations.
    sel=[tuple(map(int,x)) for x in base[:4]]
    plan=[('high',2,4),('balanced',2,4),('low',1,4),('rareD',1,3)]
    for lane,n,mc in plan: sel+=pick_lane(arr,scores[lane],n,sel,mc)
    if len(sel)<10: sel+=pick_lane(arr,scores['explorer'],10-len(sel),sel,4)
    return sel[:10]

def metrics(tickets,winner):
    ws=set(map(int,winner));hits=[len(ws&set(x)) for x in tickets]
    union=set(x for t in tickets for x in t)
    return {'best':max(hits) if hits else 0,'recall':len(ws&union),'union_n':len(union),'hits':hits}

def summarize(rows,key):
    best=np.array([r[key]['best'] for r in rows]);rec=np.array([r[key]['recall'] for r in rows])
    out={'n':len(rows),'mean_best':float(best.mean()),'mean_recall':float(rec.mean()),'median_recall':float(np.median(rec)),
         'recall6':int(np.sum(rec==6)),'recall5plus':int(np.sum(rec>=5)),'mean_union_n':float(np.mean([r[key]['union_n'] for r in rows]))}
    for k in (3,4,5,6):out[f'd{k}plus']=int(np.sum(best>=k))
    return out

def conditional(rows,newkey):
    out={}
    for label,maskfun in [('base_union6',lambda r:r['base']['recall']==6),('base_union5',lambda r:r['base']['recall']==5),('base_union_le2',lambda r:r['base']['recall']<=2)]:
        s=[r for r in rows if maskfun(r)];nb=np.array([r[newkey]['best'] for r in s]) if s else np.array([]);bb=np.array([r['base']['best'] for r in s]) if s else np.array([])
        out[label]={'n':len(s),'base_mean_best':float(bb.mean()) if len(s) else None,'new_mean_best':float(nb.mean()) if len(s) else None,
                    'new_d4plus':int(np.sum(nb>=4)) if len(s) else 0,'new_d5plus':int(np.sum(nb>=5)) if len(s) else 0,'new_d6':int(np.sum(nb>=6)) if len(s) else 0,
                    'improved_best':int(np.sum(nb>bb)) if len(s) else 0,'worsened_best':int(np.sum(nb<bb)) if len(s) else 0}
    return out

def main():
    hist=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(hist)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(hist,q);sizes,priors=p.prepare_priors(st)
    rec=[]
    for draw in range(START,END+1):
        t=di[draw]
        base,lanes,_=v1.make_portfolio(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors)
        ss,pc,comm,ns=make_scores(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors)
        pool,U,lowextra=candidate_pool(base,ns);sf=shape_freq(draws,t,500)
        arr,scores,meta=scenario_scores(pool,ns,pc,sf)
        full=assembled10(base,arr,scores);hyb=hybrid10(base,arr,scores)
        winner=draws[t]
        rec.append({'draw':draw,'part':'dev' if draw<=DEV_END else 'hold','winner':list(map(int,winner)),
                    'base':metrics(base,winner),'full':metrics(full,winner),'hybrid':metrics(hyb,winner),'lowextra':lowextra})
        if draw%25==0:print(draw,rec[-1]['base']['best'],rec[-1]['hybrid']['best'],rec[-1]['full']['best'],flush=True)
    dev=[r for r in rec if r['part']=='dev'];hold=[r for r in rec if r['part']=='hold']
    # Model choice is dev-only. Holdout is reported after fixing the better dev variant by d4, then d3, then mean_best.
    cand=[]
    for key in ('hybrid','full'):
        s=summarize(dev,key);cand.append((s['d4plus'],s['d3plus'],s['mean_best'],key))
    chosen=max(cand)[-1]
    out={'method':'Scenario-aware Assembly v2: sum x ABCD/band x core recombination','range':f'{START}-{END}',
         'caution':'Exploratory: scenario definitions were motivated by prior analysis of this same 500-draw era; dev-only chooses hybrid vs full, so hold is useful but not fully pristine.',
         'plan_full':'3 high + 3 balanced + 2 low/mid rescue + 1 high-sum D + 1 explorer',
         'plan_hybrid':'4 preserved Main + 2 high + 2 balanced + 1 low/mid rescue + 1 high-sum D',
         'chosen_by_dev':chosen,
         'dev':{k:summarize(dev,k) for k in ('base','hybrid','full')},
         'hold':{k:summarize(hold,k) for k in ('base','hybrid','full')},
         'all':{k:summarize(rec,k) for k in ('base','hybrid','full')},
         'conditional_hold':conditional(hold,chosen),'conditional_all':conditional(rec,chosen),'detail':rec}
    path=OUT/'loto6_scenario_assembly_v2_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
