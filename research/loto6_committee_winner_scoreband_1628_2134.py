#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START=1628; END=2134

def local_rows():
    out=[]
    for k in range(1,13):
        s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8')
        payload=s.split('push(',1)[1].rsplit(');',1)[0]
        for r in json.loads(payload):
            out.append((int(r[0]),tuple(sorted(int(x) for x in r[2:8])),int(r[8])))
    out.sort(key=lambda x:x[0]); return out

def band(a):
    a=np.asarray(a)
    return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))

def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def layer(f):return 'A' if f>=.02 else ('B' if f>=.01 else ('C' if f>=.005 else 'D'))

def direct_static(X,q):
    X=np.asarray(X,np.int16);s=X.sum(1);sb=np.clip((s-21)//5,0,50).astype(np.int16);odd=(X%2).sum(1).astype(np.int8);con=(np.diff(X,axis=1)==1).sum(1).astype(np.int8)
    bc=np.array([bcode(band(r)) for r in X],np.int32);gap=np.digitize(np.std(np.diff(X,axis=1),axis=1),q).astype(np.int8)
    inc=np.zeros((len(X),44),np.uint8);rr=np.arange(len(X))
    for j in range(6):inc[rr,X[:,j]]=1
    return {'sum':sb,'odd':odd,'band':bc,'consec':con,'gap':gap},inc

def winner_score(t,win,W,q,draws,bonus,npref,pairc,ref_stat,ref_c5,ref_c4):
    X=np.asarray([win],np.int16);st,inc=direct_static(X,q)
    ws=p.stat_score(t,500,W,st,inc,draws,bonus,npref)
    w5,w4=p.cores(X,pairc)
    zstat=(ws[0]-ref_stat.mean())/(ref_stat.std()+1e-9)
    z5=(w5[0]-ref_c5.mean())/(ref_c5.std()+1e-9)
    z4=(w4[0]-ref_c4.mean())/(ref_c4.std()+1e-9)
    return float(zstat+.20*z5+.15*z4),float(ws[0]),float(w5[0]),float(w4[0])

def pct_bins(vals):
    edges=[0,.1,1,5,10,25,50,75,100.000001]
    labels=['0-0.1%','0.1-1%','1-5%','5-10%','10-25%','25-50%','50-75%','75-100%']
    return {labels[i]:sum(edges[i] <= x < edges[i+1] for x in vals) for i in range(len(labels))}

def summary(rows):
    if not rows:return {'n':0}
    pct=np.array([r['committee_top_percent_60k'] for r in rows],float);sc=np.array([r['committee_score'] for r in rows],float)
    return {
      'n':len(rows),
      'top0_1_count':int(np.sum(pct<=.1)),'top1_count':int(np.sum(pct<=1)),'top5_count':int(np.sum(pct<=5)),'top10_count':int(np.sum(pct<=10)),'top25_count':int(np.sum(pct<=25)),
      'percentile_bins':pct_bins(pct.tolist()),
      'percentile_quantiles':{'min':float(np.min(pct)),'q10':float(np.quantile(pct,.10)),'q25':float(np.quantile(pct,.25)),'median':float(np.median(pct)),'q75':float(np.quantile(pct,.75)),'q90':float(np.quantile(pct,.90)),'max':float(np.max(pct))},
      'score_quantiles':{'min':float(np.min(sc)),'q10':float(np.quantile(sc,.10)),'q25':float(np.quantile(sc,.25)),'median':float(np.median(sc)),'q75':float(np.quantile(sc,.75)),'q90':float(np.quantile(sc,.90)),'max':float(np.max(sc))},
    }

def main():
    rows=local_rows(); rows=[r for r in rows if r[0]<=END]
    di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    recs=[]
    for draw in range(START,END+1):
        if draw not in di:continue
        t=di[draw]
        if t<500:continue
        W=p.weights(t,500,actual,sizes,priors);ref_stat=p.stat_score(t,500,W,st,inc,draws,bonus,npref)
        pairc=ppref[t]-ppref[max(0,t-300)];ref_c5,ref_c4=p.cores(C,pairc)
        z=lambda x:(x-x.mean())/(x.std()+1e-9)
        ref_comm=z(ref_stat)+.20*z(ref_c5)+.15*z(ref_c4)
        win=tuple(map(int,draws[t]));ws,raws,w5,w4=winner_score(t,win,W,q,draws,bonus,npref,pairc,ref_stat,ref_c5,ref_c4)
        rank=int(np.count_nonzero(ref_comm>ws)+1);pct=rank/(len(C)+1)*100
        sh=band(win);prev=set(map(int,draws[t-1]));prevsum=int(draws[t-1].sum())
        c20=Counter(band(r) for r in draws[t-20:t]);c2150=Counter(band(r) for r in draws[t-50:t-20]);c50=Counter(band(r) for r in draws[t-50:t]);hall=Counter(band(r) for r in draws[:t])
        temporal=(c20[sh]==0 and (c2150[sh]==1 or c50[sh]==0));pover=len(set(win)&prev);pbonus=int(bonus[t-1]) in win;sumup=sum(win)>prevsum
        ly=layer(hall[sh]/t)
        rec={'draw':draw,'winner':list(win),'sum':sum(win),'prev_sum':prevsum,'sum_up':sumup,'shape':list(sh),'layer':ly,'shape_long_freq_pre':hall[sh]/t,'temporal_shape_ok':temporal,'prev_overlap':pover,'prev_bonus_absent':not pbonus,'committee_score':ws,'committee_rank_60k':rank,'committee_top_percent_60k':pct,'stat_raw':raws,'core5':w5,'core4':w4}
        rec['structural_current_like']=bool(temporal and pover<=1 and not pbonus)
        rec['structural_plus_sumup']=bool(rec['structural_current_like'] and sumup)
        recs.append(rec)
        if draw%25==0:print(draw,round(pct,3),round(ws,3),rec['structural_current_like'],flush=True)
    allr=recs;struct=[r for r in recs if r['structural_current_like']];structsum=[r for r in recs if r['structural_plus_sumup']]
    bylayer={ly:summary([r for r in struct if r['layer']==ly]) for ly in 'ABCD'}
    out={
      'definition':'Pre-draw leakage-safe Committee percentile versus deterministic fixed 60k reference sample. Committee=Z(Stat500)+0.20Z(5core300)+0.15Z(4core300). Structural-current-like means temporal shape rule + previous-main overlap<=1 + previous bonus absent. structural_plus_sumup additionally requires current sum > previous draw sum. This is approximate percentile, not exact all-6,096,454 rank.',
      'range':[START,END],'reference_sample_size':len(C),
      'all':summary(allr),'structural_current_like':summary(struct),'structural_plus_sumup':summary(structsum),'structural_by_layer':bylayer,
      'rows':recs
    }
    (OUT/'loto6_committee_winner_scoreband_1628_2134.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
