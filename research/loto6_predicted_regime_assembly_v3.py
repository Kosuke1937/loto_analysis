#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v2',ROOT/'research'/'loto6_scenario_assembly_v2.py')
v2=importlib.util.module_from_spec(spec);spec.loader.exec_module(v2)
v1=v2.v1; p=v2.p; v4=v2.v4
OUT=ROOT/'research'/'results'; OUT.mkdir(parents=True,exist_ok=True)
START,DEV_END,TEST_START,END=1828,1927,1928,2127
SUM_BINS=[(-10**9,109,'low'),(110,129,'midlow'),(130,149,'mid'),(150,10**9,'high')]
CLASSES=('A','B','C','D')


def sum_bin(s):
    for lo,hi,n in SUM_BINS:
        if lo<=s<=hi:return n
    return 'mid'

def sfreq(draws,t,w=500):
    c=Counter(v4.band_tuple(draws[u]) for u in range(max(0,t-w),t)); n=max(1,min(w,t))
    return {k:v/n for k,v in c.items()}

def cls_at(draws,t):
    sf=sfreq(draws,t,500); sh=v4.band_tuple(draws[t]); return v2.band_class(sf.get(sh,0.0))

def hist_labels(draws,t):
    # labels are computed as they would have been known after each historical draw, using only its prior history for the class threshold.
    out=[]
    for u in range(max(501,t-500),t):
        out.append((u,cls_at(draws,u),sum_bin(int(np.sum(draws[u])))))
    return out

def smooth_probs(counter,keys,alpha=1.0):
    tot=sum(counter.get(k,0) for k in keys)+alpha*len(keys)
    return {k:(counter.get(k,0)+alpha)/tot for k in keys}

def predict_probs(draws,t,mode):
    H=hist_labels(draws,t)
    cC=Counter(c for _,c,_ in H); cS=Counter(s for _,_,s in H)
    pC=smooth_probs(cC,CLASSES,2.0); pS=smooth_probs(cS,('low','midlow','mid','high'),2.0)
    if not H:return pC,pS,{'confidence':0.0}
    prevC=H[-1][1]; prevS=H[-1][2]
    transC=Counter(); transS=Counter(); recentC=Counter(); recentS=Counter()
    for i in range(1,len(H)):
        if H[i-1][1]==prevC: transC[H[i][1]]+=1
        if H[i-1][2]==prevS: transS[H[i][2]]+=1
    for _,c,s in H[-50:]:recentC[c]+=1;recentS[s]+=1
    ptC=smooth_probs(transC,CLASSES,4.0); ptS=smooth_probs(transS,('low','midlow','mid','high'),4.0)
    prC=smooth_probs(recentC,CLASSES,2.0); prS=smooth_probs(recentS,('low','midlow','mid','high'),2.0)
    if mode=='marginal': pass
    elif mode=='transition':
        pC={k:.55*pC[k]+.45*ptC[k] for k in CLASSES}; pS={k:.55*pS[k]+.45*ptS[k] for k in pS}
    elif mode=='recent':
        pC={k:.60*pC[k]+.40*prC[k] for k in CLASSES}; pS={k:.60*pS[k]+.40*prS[k] for k in pS}
    elif mode=='ensemble':
        pC={k:.45*pC[k]+.30*ptC[k]+.25*prC[k] for k in CLASSES}; pS={k:.45*pS[k]+.30*ptS[k]+.25*prS[k] for k in pS}
    elif mode=='twostep':
        if len(H)>=2:
            key=(H[-2][1],H[-1][1]); keyS=(H[-2][2],H[-1][2]); cc=Counter();ss=Counter()
            for i in range(2,len(H)):
                if (H[i-2][1],H[i-1][1])==key: cc[H[i][1]]+=1
                if (H[i-2][2],H[i-1][2])==keyS: ss[H[i][2]]+=1
            p2C=smooth_probs(cc,CLASSES,8.0); p2S=smooth_probs(ss,('low','midlow','mid','high'),8.0)
            pC={k:.65*pC[k]+.35*p2C[k] for k in CLASSES};pS={k:.65*pS[k]+.35*p2S[k] for k in pS}
    valsC=sorted(pC.values(),reverse=True); valsS=sorted(pS.values(),reverse=True)
    conf=(valsC[0]-valsC[1])+(valsS[0]-valsS[1])
    return pC,pS,{'confidence':float(conf),'prev_class':prevC,'prev_sum_bin':prevS}

def choose_plan(pC,pS,policy,conf=0.0):
    # Plan contains only replacement tickets; remaining slots preserve the strongest Base tickets.
    high=pS['high']; low=pS['low']+0.45*pS['midlow']; d=pC['D']; ab=pC['A']+pC['B'];
    if policy=='P1':
        return [('high',1),('balanced',1),('low',1)]
    if policy=='P2':
        if d>=.20 and high>=.22:return [('rareD',2),('high',2),('balanced',1)]
        if low>=.38:return [('low',2),('balanced',2),('high',1)]
        return [('balanced',2),('high',1),('low',1)]
    if policy=='P3':
        if d*high>=.055:return [('rareD',2),('high',2)]
        if ab>=.70 and pS['mid']+.5*pS['midlow']>=.35:return [('balanced',3),('high',1)]
        if low>=.42:return [('low',3),('balanced',1)]
        return [('high',1),('balanced',2),('low',1)]
    if policy=='P4':
        # Conservative confidence gate: if prediction is weak, change only two tickets.
        if conf<.12:return [('balanced',1),('low',1)]
        if d>=.20 and high>=.22:return [('rareD',1),('high',2),('balanced',1)]
        if low>=.38:return [('low',2),('balanced',1),('high',1)]
        return [('balanced',2),('high',1)]
    if policy=='P5':
        # Always preserve at least 6 Base tickets; targeted one-sided boosts.
        pl=[]
        if d>=.18 and high>=.20:pl += [('rareD',1),('high',1)]
        elif high>=.30:pl += [('high',2)]
        if low>=.34:pl += [('low',1)]
        if ab>=.62:pl += [('balanced',1)]
        if not pl:pl=[('balanced',1),('low',1)]
        return pl[:4]
    return [('balanced',1),('low',1)]

def build_dynamic(base,arr,scores,plan):
    # Preserve Base first, then replace the weakest tail slots with scenario tickets.
    nrep=min(5,sum(n for _,n in plan)); keep=max(5,10-nrep)
    sel=[tuple(map(int,x)) for x in base[:keep]]
    for lane,n in plan:
        sel += v2.pick_lane(arr,scores[lane],n,sel,4 if lane!='rareD' else 3)
    if len(sel)<10:
        sel += v2.pick_lane(arr,scores['explorer'],10-len(sel),sel,4)
    return sel[:10]

def summarize(rows,key):return v2.summarize(rows,key)

def pred_accuracy(rows,mode):
    out={'n':len(rows),'class_top1':0,'sum_top1':0,'D_high_n':0,'D_high_hit':0}
    for r in rows:
        z=r['pred'][mode]; pc=z['class'];ps=z['sum']; ac=r['actual_class'];asb=r['actual_sum_bin']
        out['class_top1']+=max(pc,key=pc.get)==ac;out['sum_top1']+=max(ps,key=ps.get)==asb
        if pc['D']>=.20 and ps['high']>=.22:
            out['D_high_n']+=1;out['D_high_hit']+=(ac=='D' and asb=='high')
    out['class_acc']=out['class_top1']/max(1,out['n']);out['sum_acc']=out['sum_top1']/max(1,out['n'])
    return out

def main():
    hist=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(hist)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(hist,q);sizes,priors=p.prepare_priors(st)
    modes=('marginal','transition','recent','ensemble','twostep'); policies=('P1','P2','P3','P4','P5')
    rec=[]
    for draw in range(START,END+1):
        t=di[draw]
        base,_,_=v1.make_portfolio(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors)
        ss,pc,comm,ns=v2.make_scores(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors)
        pool,U,lowextra=v2.candidate_pool(base,ns);sf=v2.shape_freq(draws,t,500);arr,scores,meta=v2.scenario_scores(pool,ns,pc,sf)
        winner=draws[t]; actual_class=v2.band_class(sf.get(v4.band_tuple(winner),0.0)); actual_sum_bin=sum_bin(int(np.sum(winner)))
        row={'draw':draw,'part':'dev' if draw<=DEV_END else 'test','winner':list(map(int,winner)),'actual_class':actual_class,'actual_sum_bin':actual_sum_bin,'base':v2.metrics(base,winner),'pred':{}}
        for mode in modes:
            pC,pS,info=predict_probs(draws,t,mode);row['pred'][mode]={'class':pC,'sum':pS,**info}
            for pol in policies:
                plan=choose_plan(pC,pS,pol,info['confidence']);tickets=build_dynamic(base,arr,scores,plan)
                row[f'{mode}_{pol}']=v2.metrics(tickets,winner);row[f'{mode}_{pol}']['plan']=plan
        rec.append(row)
        if draw%20==0:print(draw,row['base']['best'],flush=True)
    dev=[r for r in rec if r['part']=='dev']; test=[r for r in rec if r['part']=='test']
    # Select using only 100-draw dev: prioritize d4, then d3, then mean best, then union recall.
    keys=[f'{m}_{p0}' for m in modes for p0 in policies]
    table={k:summarize(dev,k) for k in keys}
    chosen=max(keys,key=lambda k:(table[k]['d4plus'],table[k]['d3plus'],table[k]['mean_best'],table[k]['mean_recall']))
    out={'method':'Predicted-regime Scenario Assembly v3','range':f'{START}-{END}','dev_range':f'{START}-{DEV_END} (100 draws)','test_range':f'{TEST_START}-{END} (200 draws)',
         'principle':'For every draw, predict ABCD-class probabilities and sum-bin probabilities from prior draws only; condition how many Base tickets are replaced by high/balanced/low/rare-D assemblies. No realized target class/sum is used in assembly.',
         'modes':list(modes),'policies':list(policies),'chosen_dev_only':chosen,
         'prediction_dev':{m:pred_accuracy(dev,m) for m in modes},'prediction_test':{m:pred_accuracy(test,m) for m in modes},
         'base_dev':summarize(dev,'base'),'base_test':summarize(test,'base'),
         'chosen_dev':summarize(dev,chosen),'chosen_test':summarize(test,chosen),
         'all_dev_variants':table,'all_test_variants':{k:summarize(test,k) for k in keys},'detail':rec}
    (OUT/'loto6_predicted_regime_assembly_v3_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('detail','all_dev_variants','all_test_variants')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
