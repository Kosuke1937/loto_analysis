#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json,math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p',ROOT/'research'/'loto6_consensus_phase1_sample.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
CAL_START=1628; EVAL_START=1828; END=2135
POOL_SIZES=(22,26,28,30,32)


def local_rows():
    out=[]
    for k in range(1,13):
        s=(ROOT/'data'/f'loto6-chunk-{k}.js').read_text(encoding='utf-8')
        payload=s.split('push(',1)[1].rsplit(');',1)[0]
        for r in json.loads(payload):
            out.append((int(r[0]),tuple(sorted(int(x) for x in r[2:8])),int(r[8])))
    out.sort(key=lambda x:x[0]);return [r for r in out if r[0]<=END]


def band(a):
    a=np.asarray(a)
    return (int(np.sum(a<=9)),int(np.sum((a>=10)&(a<=19))),int(np.sum((a>=20)&(a<=29))),int(np.sum((a>=30)&(a<=39))),int(np.sum(a>=40)))

def bcode(b):return b[0]*2401+b[1]*343+b[2]*49+b[3]*7+b[4]
def layer(f):return 'A' if f>=.02 else ('B' if f>=.01 else ('C' if f>=.005 else 'D'))

def direct_static(X,q):
    X=np.asarray(X,np.int16);s=X.sum(1);sb=np.clip((s-21)//5,0,50).astype(np.int16);odd=(X%2).sum(1).astype(np.int8);con=(np.diff(X,axis=1)==1).sum(1).astype(np.int8);bc=np.array([bcode(band(r)) for r in X],np.int32);gap=np.digitize(np.std(np.diff(X,axis=1),axis=1),q).astype(np.int8);inc=np.zeros((len(X),44),np.uint8);rr=np.arange(len(X))
    for j in range(6):inc[rr,X[:,j]]=1
    return {'sum':sb,'odd':odd,'band':bc,'consec':con,'gap':gap},inc

def winner_score(t,win,W,q,draws,bonus,npref,pairc,rs,r5,r4):
    X=np.asarray([win],np.int16);st,inc=direct_static(X,q);s=p.stat_score(t,500,W,st,inc,draws,bonus,npref);c5,c4=p.cores(X,pairc)
    return float((s[0]-rs.mean())/(rs.std()+1e-9)+.20*(c5[0]-r5.mean())/(r5.std()+1e-9)+.15*(c4[0]-r4.mean())/(r4.std()+1e-9))

def number_support(agents,C):
    sup=np.zeros(44,float)
    for idx in agents.values():
        for rank,i in enumerate(idx[:500],1):
            w=1.0/math.log2(rank+2)
            for x in C[int(i)]: sup[int(x)]+=w
    return sup

def top_pool(sup,k):
    nums=np.arange(1,44)
    order=np.lexsort((nums,-sup[1:]))
    return tuple(map(int,nums[order[:k]]))

def build_layer_map(hall,t):
    lm=np.empty(16807,dtype='<U1');allowed=[]
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3;sh=(b0,b1,b2,b3,b4);code=bcode(sh);lm[code]=layer(hall[sh]/t)
    return lm

def allowed_shapes(draws,t):
    c20=Counter(band(r) for r in draws[t-20:t]);c2150=Counter(band(r) for r in draws[t-50:t-20]);c50=Counter(band(r) for r in draws[t-50:t]);out=[]
    for b0 in range(7):
      for b1 in range(7-b0):
       for b2 in range(7-b0-b1):
        for b3 in range(7-b0-b1-b2):
         b4=6-b0-b1-b2-b3;sh=(b0,b1,b2,b3,b4)
         if c20[sh]==0 and (c2150[sh]==1 or c50[sh]==0):out.append(bcode(sh))
    return np.asarray(out,np.int32),c20,c2150,c50

def admissible(i,C,ly,shape,lc,sc,nc,pc,tc,strict=True):
    quota={'A':3,'B':3,'C':2,'D':2};row=tuple(map(int,C[i]))
    if lc[ly]>=quota[ly] or sc[shape]>=2:return False
    pa=list(itertools.combinations(row,2));tr=list(itertools.combinations(row,3))
    if strict and (any(nc[x]>=4 for x in row) or any(pc[x]>=2 for x in pa) or any(tc[x]>=1 for x in tr)):return False
    return True

def add_sel(i,C,ly,shape,sel,lc,sc,nc,pc,tc):
    row=tuple(map(int,C[i]));sel.append(int(i));lc[ly]+=1;sc[shape]+=1
    for x in row:nc[x]+=1
    for x in itertools.combinations(row,2):pc[x]+=1
    for x in itertools.combinations(row,3):tc[x]+=1

def select_order(order,C,layers,shapes,n=10):
    sel=[];lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter()
    for strict in (True,False):
        for ii in order:
            i=int(ii)
            if i in sel:continue
            if not admissible(i,C,layers[i],shapes[i],lc,sc,nc,pc,tc,strict):continue
            add_sel(i,C,layers[i],shapes[i],sel,lc,sc,nc,pc,tc)
            if len(sel)>=n:return sel
    return sel

def select_coverage(ids,dist,C,layers,shapes,pool,sup,n=10):
    # Coverage is lexicographic: new unique pool numbers first, support of new numbers second,
    # then closeness to historical winner-score target. Same quotas/diversity as baseline.
    # Restrict search to the 4000 closest-score candidates only to keep the comparison tractable.
    if len(ids)==0:return []
    ord0=ids[np.argsort(dist)][:min(4000,len(ids))]
    sel=[];used=set();lc=Counter();sc=Counter();nc=Counter();pc=Counter();tc=Counter();poolset=set(pool)
    for strict in (True,False):
        while len(sel)<n:
            best=None;bestkey=None
            for ii in ord0:
                i=int(ii)
                if i in sel:continue
                if not admissible(i,C,layers[i],shapes[i],lc,sc,nc,pc,tc,strict):continue
                row=set(map(int,C[i]));new=(row&poolset)-used
                key=(len(new),sum(sup[x] for x in new),-float(dist[np.where(ids==i)[0][0]]))
                if bestkey is None or key>bestkey:bestkey=key;best=i
            if best is None:break
            add_sel(best,C,layers[best],shapes[best],sel,lc,sc,nc,pc,tc);used.update(map(int,C[best]))
        if len(sel)>=n:break
    return sel

def metrics(win,sel,C):
    W=set(map(int,win));tickets=[set(map(int,C[i])) for i in sel]
    best=max([len(W&t) for t in tickets] or [0]);U=set().union(*tickets) if tickets else set();union=len(W&U)
    wp=set(itertools.combinations(sorted(W),2));wt=set(itertools.combinations(sorted(W),3));sp=set();st=set()
    for i in sel:
        row=tuple(map(int,C[i]));sp.update(itertools.combinations(row,2));st.update(itertools.combinations(row,3))
    return {'n_tickets':len(sel),'best':best,'union':union,'pair_recall':len(wp&sp),'triple_recall':len(wt&st),'d3':int(best>=3),'d4':int(best>=4),'d5':int(best>=5),'d6':int(best>=6)}

def summarize(rr):
    if not rr:return {}
    return {'n':len(rr),'mean_tickets':float(np.mean([x['n_tickets'] for x in rr])),'mean_best':float(np.mean([x['best'] for x in rr])),'mean_union':float(np.mean([x['union'] for x in rr])),'union5plus':sum(x['union']>=5 for x in rr),'union6':sum(x['union']==6 for x in rr),'mean_pair_recall':float(np.mean([x['pair_recall'] for x in rr])),'mean_triple_recall':float(np.mean([x['triple_recall'] for x in rr])),'d3':sum(x['d3'] for x in rr),'d4':sum(x['d4'] for x in rr),'d5':sum(x['d5'] for x in rr),'d6':sum(x['d6'] for x in rr)}
def paired(a,b,key):
    gain=sum(int(y[key]>x[key]) for x,y in zip(a,b));loss=sum(int(y[key]<x[key]) for x,y in zip(a,b));tie=len(a)-gain-loss
    return {'gain':gain,'loss':loss,'tie':tie}

def main():
    rows=local_rows();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);sums=C.sum(1);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st)
    poolR={k:[] for k in POOL_SIZES};R={'committee_top28':[],'scorematch28':[],'coverage28':[]};details=[];past=[]
    for draw in range(CAL_START,END+1):
        if draw not in di:continue
        t=di[draw]
        ss={}
        for h in (200,500,800):
            W_h=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W_h,st,inc,draws,bonus,npref)
        W=p.weights(t,500,actual,sizes,priors);pairc=ppref[t]-ppref[max(0,t-300)];r5,r4=p.cores(C,pairc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(r5)+.15*z(r4)
        agents={'stat200':p.topidx(ss[200],500),'stat500':p.topidx(ss[500],500),'stat800':p.topidx(ss[800],500),'committee':p.topidx(comm,500)}
        sup=number_support(agents,C);pools={k:top_pool(sup,k) for k in POOL_SIZES};win=tuple(map(int,draws[t]));Wset=set(win)
        if draw>=EVAL_START:
            for k,pool in pools.items(): poolR[k].append(len(Wset&set(pool)))
        wsc=winner_score(t,win,W,q,draws,bonus,npref,pairc,ss[500],r5,r4)
        ash,c20,c2150,c50=allowed_shapes(draws,t);hall=Counter(band(r) for r in draws[:t]);wsh=band(win);prev=set(map(int,draws[t-1]));prevsum=int(draws[t-1].sum());wqual=(c20[wsh]==0 and (c2150[wsh]==1 or c50[wsh]==0) and len(Wset&prev)<=1 and int(bonus[t-1]) not in Wset and sum(win)>prevsum)
        if draw>=EVAL_START and len(past)>=20:
            pool28=pools[28];poolmask=inc[:,list(pool28)].sum(1)==6;po=inc[:,list(prev)].sum(1);valid=np.isin(st['band'],ash)&(po<=1)&(inc[:,int(bonus[t-1])]==0)&(sums>prevsum)&poolmask;ids=np.where(valid)[0]
            lm=build_layer_map(hall,t);layers=lm[st['band']];shapes=st['band'];target=float(np.median(past))
            if len(ids)>=10:
                d=np.abs(comm[ids]-target);toporder=ids[np.argsort(comm[ids])[::-1]];matchorder=ids[np.argsort(d)]
                sels={'committee_top28':select_order(toporder,C,layers,shapes),'scorematch28':select_order(matchorder,C,layers,shapes),'coverage28':select_coverage(ids,d,C,layers,shapes,pool28,sup)}
                rec={'draw':draw,'winner':list(win),'winner_qualifies':bool(wqual),'winner_score':wsc,'target_median':target,'pool28':list(pool28),'pool28_winner_recall':len(Wset&set(pool28)),'valid_count':len(ids)}
                for name,sel in sels.items():
                    m=metrics(win,sel,C);R[name].append(m);rec[name]=m;rec[name+'_tickets']=[list(map(int,C[i])) for i in sel]
                details.append(rec)
        if wqual:past.append(wsc)
        if draw%25==0:print(draw,'past',len(past),flush=True)
    pool_summary={}
    for k,v in poolR.items():pool_summary[str(k)]={'n':len(v),'mean_recall':float(np.mean(v)),'recall4plus':sum(x>=4 for x in v),'recall5plus':sum(x>=5 for x in v),'recall6':sum(x==6 for x in v),'distribution':{str(i):sum(x==i for x in v) for i in range(7)}}
    methods={k:summarize(v) for k,v in R.items()};pairs={'scorematch_vs_top':{q:paired(R['committee_top28'],R['scorematch28'],q) for q in ('best','union','pair_recall','triple_recall','d3','d4')},'coverage_vs_scorematch':{q:paired(R['scorematch28'],R['coverage28'],q) for q in ('best','union','pair_recall','triple_recall','d3','d4')}}
    d2135=next((x for x in details if x['draw']==2135),None)
    out={'definition':'Leakage-safe deterministic 60k sample comparison. Pool support = rank-weighted number support from pre-draw Stat200/500/800/Committee Top500. Pool recall is exact against each historical winner. Integrated 10-ticket comparison uses identical Pool28, temporal shape rule, previous overlap<=1, previous bonus absent, candidate sum>previous sum, A3/B3/C2/D2, shape<=2, and diversity caps. Committee-top sorts descending score; scorematch sorts by distance to rolling median score of prior qualifying winners; coverage uses the same winner-score target but lexicographically maximizes new Pool28 number coverage before score distance. Frequency/last-digit soft score is intentionally omitted to isolate these three design choices.','calibration_start':CAL_START,'evaluation_start':EVAL_START,'end':END,'pool_size_summary':pool_summary,'portfolio_methods':methods,'paired':pairs,'draw2135':d2135,'details':details}
    (OUT/'loto6_pool_coverage_scorematch_compare.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pool_size_summary':pool_summary,'portfolio_methods':methods,'paired':pairs,'draw2135':d2135},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
