#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loto6 Consensus-Assembly Phase 1 (frozen, reproducible, leakage-safe)

Goal:
Compare a conventional Committee diversified 10-ticket portfolio against a
Selective Consensus-Assembly 10-ticket portfolio on draws 1628-2127.

This phase intentionally uses only agents whose definitions are recoverable:
  - Stat200
  - Stat500
  - Stat800
  - Committee = Z(Stat500)+0.20 Z(5core)+0.15 Z(4core)

Candidate universe is a deterministic fixed 60,000-combination sample drawn
without using any target winner. Therefore this is an approximate portfolio
backtest, not an exact all-6,096,454 ranking test.

KPI:
  union coverage, core3/core4 capture, assembly ratio, d3/d4/d5/d6,
  mean best match.
"""
from __future__ import annotations
import csv, io, itertools, json, math, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np

SEED=20260825
NSAMPLE=60000
START_DRAW=1628
END_DRAW=2127
TOP_SUPPORT=500
ALPHA=75.0
CLIP=1.5
URL='https://www.mk-mode.com/rails/loto/LOTO6_ALL.csv'
OUT=Path('research/results')
OUT.mkdir(parents=True,exist_ok=True)


def fetch_history():
    raw=urllib.request.urlopen(URL,timeout=60).read()
    try:text=raw.decode('cp932')
    except UnicodeDecodeError:text=raw.decode('utf-8')
    rd=csv.reader(io.StringIO(text)); header=next(rd)
    rows=[]
    for r in rd:
        try:
            draw=int(r[0]); nums=tuple(sorted(int(r[2+i]) for i in range(6))); bonus=int(r[8])
            rows.append((draw,nums,bonus))
        except Exception:pass
    rows.sort()
    return rows


def fixed_sample(n=NSAMPLE):
    rng=np.random.default_rng(SEED); seen=set(); out=[]
    while len(out)<n:
        # batch generation; sorting removes within-row order
        X=np.sort(rng.choice(np.arange(1,44),size=(min(20000,n-len(out))*2,6),replace=True),axis=1)
        for a in X:
            t=tuple(map(int,a))
            if len(set(t))==6 and t not in seen:
                seen.add(t);out.append(t)
                if len(out)>=n:break
    return np.asarray(out,np.int16)


def build_static(C):
    s=C.sum(1); sb=np.clip((s-21)//5,0,50).astype(np.int16)
    odd=(C%2).sum(1).astype(np.int8)
    con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
    b0=(C<=9).sum(1);b1=((C>=10)&(C<=19)).sum(1);b2=((C>=20)&(C<=29)).sum(1);b3=((C>=30)&(C<=39)).sum(1);b4=(C>=40).sum(1)
    band=(b0*2401+b1*343+b2*49+b3*7+b4).astype(np.int32)
    gs=np.std(np.diff(C,axis=1),axis=1)
    q=np.quantile(gs,[.2,.4,.6,.8]);gap=np.digitize(gs,q).astype(np.int8)
    inc=np.zeros((len(C),44),np.uint8)
    rr=np.arange(len(C))
    for j in range(6):inc[rr,C[:,j]]=1
    return {'sum':sb,'odd':odd,'band':band,'consec':con,'gap':gap},inc,q


def hist_actual(rows,q):
    draws=np.asarray([r[1] for r in rows],np.int16);bonus=np.asarray([r[2] for r in rows],np.int16)
    T=len(draws)
    npref=np.zeros((T+1,44),np.int32);ppref=np.zeros((T+1,44,44),np.int16)
    for t,row in enumerate(draws):
        npref[t+1]=npref[t];npref[t+1,row]+=1;ppref[t+1]=ppref[t]
        for a,b in itertools.combinations(map(int,row),2):ppref[t+1,a,b]+=1;ppref[t+1,b,a]+=1
    def rnum(t,w):return npref[t]-npref[max(0,t-w)]
    actual=[]
    for t,row in enumerate(draws):
        s=int(row.sum()); odd=int((row%2).sum()); con=int((np.diff(row)==1).sum())
        ba=[np.sum(row<=9),np.sum((row>=10)&(row<=19)),np.sum((row>=20)&(row<=29)),np.sum((row>=30)&(row<=39)),np.sum(row>=40)]
        band=int(ba[0]*2401+ba[1]*343+ba[2]*49+ba[3]*7+ba[4])
        gap=int(np.digitize(np.std(np.diff(row)),q))
        cur=set(map(int,row));prev=set(map(int,draws[t-1])) if t else set();prev2=set(map(int,draws[t-2])) if t>=2 else set()
        c300=rnum(t,300);hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]) if t else set()
        actual.append({'sum':int(np.clip((s-21)//5,0,50)),'odd':odd,'band':band,'consec':con,'gap':gap,
            'prev':len(cur&prev),'prev2':len(cur&prev2),'pbonus':int(t>=1 and int(bonus[t-1]) in cur),'hot':sum(int(x) in hot for x in row)})
    return draws,bonus,npref,ppref,actual


def hyp(K):
    den=math.comb(43,6);a=np.zeros(7,float)
    for k in range(7):
        if k<=K and 6-k<=43-K:a[k]=math.comb(K,k)*math.comb(43-K,6-k)/den
    return a


def prepare_priors(st):
    sizes={'sum':51,'odd':7,'band':16807,'consec':6,'gap':5,'prev':7,'prev2':7,'pbonus':2,'hot':7}
    p={k:np.bincount(v,minlength=sizes[k]).astype(float) for k,v in st.items()}
    for k in p:p[k]/=p[k].sum()
    p['prev']=hyp(6);p['prev2']=hyp(6);p['hot']=hyp(15);p['pbonus']=np.array([37/43,6/43],float)
    return sizes,p


def weights(t,hist,actual,sizes,priors):
    lo=max(0,t-hist);n=max(1,t-lo);W={}
    for f in ('sum','odd','band','consec','gap','prev','prev2','pbonus','hot'):
        wins=np.bincount([actual[u][f] for u in range(lo,t)],minlength=sizes[f]).astype(float)
        p=priors[f];q=(wins+ALPHA*p)/(n+ALPHA)
        W[f]=np.clip(np.log(np.maximum(q,1e-15)/np.maximum(p,1e-15)),-CLIP,CLIP)
    return W


def stat_score(t,hist,W,st,inc,draws,bonus,npref):
    prev=draws[t-1];prev2=draws[t-2] if t>=2 else np.array([],np.int16)
    c300=npref[t]-npref[max(0,t-300)];hot=(np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15]
    po=inc[:,prev].sum(1);p2=inc[:,prev2].sum(1) if len(prev2) else np.zeros(len(inc),np.int8)
    pb=inc[:,int(bonus[t-1])] if t else np.zeros(len(inc),np.uint8);hh=inc[:,hot].sum(1)
    return (W['sum'][st['sum']]+W['odd'][st['odd']]+W['band'][st['band']]+W['consec'][st['consec']]+W['gap'][st['gap']]+
            W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)


def cores(C,pc):
    total=np.zeros(len(C),np.float32);incident=np.zeros((len(C),6),np.float32)
    for i,j in itertools.combinations(range(6),2):
        v=pc[C[:,i],C[:,j]].astype(np.float32);total+=v;incident[:,i]+=v;incident[:,j]+=v
    c5=np.max(total[:,None]-incident,axis=1)
    c4=np.full(len(C),-1e9,np.float32)
    for i,j in itertools.combinations(range(6),2):c4=np.maximum(c4,total-incident[:,i]-incident[:,j]+pc[C[:,i],C[:,j]])
    return c5,c4


def topidx(s,k):
    k=min(k,len(s));x=np.argpartition(s,-k)[-k:];return x[np.argsort(s[x])[::-1]]


def diverse(indices,C,n=10,triple_cap=1,pair_cap=2,num_cap=4):
    sel=[];tc=Counter();pc=Counter();nc=Counter()
    for i in indices:
        row=tuple(map(int,C[i]));tr=list(itertools.combinations(row,3));pa=list(itertools.combinations(row,2))
        if any(tc[x]>=triple_cap for x in tr) or any(pc[x]>=pair_cap for x in pa) or any(nc[x]>=num_cap for x in row):continue
        sel.append(int(i));
        for x in tr:tc[x]+=1
        for x in pa:pc[x]+=1
        for x in row:nc[x]+=1
        if len(sel)==n:break
    if len(sel)<n:
        used=set(sel)
        for i in indices:
            if int(i) not in used:sel.append(int(i));used.add(int(i))
            if len(sel)==n:break
    return sel


def support(agents,C):
    v3=Counter();v4=Counter();rw3=Counter();rw4=Counter()
    for name,idx in agents.items():
        seen3={};seen4={}
        for rank,i in enumerate(idx[:TOP_SUPPORT],1):
            w=1/np.log2(rank+2);row=tuple(map(int,C[i]))
            for c in itertools.combinations(row,3):seen3[c]=max(seen3.get(c,0),w)
            for c in itertools.combinations(row,4):seen4[c]=max(seen4.get(c,0),w)
        for c,w in seen3.items():v3[c]+=1;rw3[c]+=w
        for c,w in seen4.items():v4[c]+=1;rw4[c]+=w
    return v3,rw3,v4,rw4


def consensus_portfolio(agents,scores,C):
    v3,rw3,v4,rw4=support(agents,C)
    pool=np.unique(np.concatenate([x[:1500] for x in agents.values()]))
    top3=max(v3,key=lambda c:(v3[c],rw3[c])) if v3 else None
    def cscore(i):
        row=tuple(map(int,C[i]));b3=max((v3[c],rw3[c]) for c in itertools.combinations(row,3));b4=max((v4[c],rw4[c]) for c in itertools.combinations(row,4))
        return 2*b4[0]+b3[0]+.35*b4[1]+.15*b3[1]
    order=sorted(map(int,pool),key=cscore,reverse=True)
    sel=[]
    def allowed(i):
        row=tuple(map(int,C[i]))
        if any(len(set(row)&set(C[j]))>=5 for j in sel):return False
        cnt=Counter(c for j in sel for c in itertools.combinations(tuple(map(int,C[j])),3))
        for c in itertools.combinations(row,3):
            lim=2 if c==top3 else 1
            if cnt[c]>=lim:return False
        return True
    # 4 consensus
    for i in order:
        if allowed(i):sel.append(i)
        if len(sel)>=4:break
    # 2 committee
    for i in agents['committee']:
        if i not in sel and allowed(int(i)):sel.append(int(i))
        if len(sel)>=6:break
    # one stat200, one stat800 with low overlap
    for name in ('stat200','stat800'):
        cand=[int(i) for i in agents[name] if int(i) not in sel]
        cand.sort(key=lambda i:(max([len(set(C[i])&set(C[j])) for j in sel] or [0]), list(agents[name]).index(i)))
        for i in cand:
            if allowed(i):sel.append(i);break
    # two structural-tail rescues from candidate pool
    def extreme(i):
        a=C[i];su=a.sum();odd=(a%2).sum();low=(a<=21).sum();bands=[np.sum(a<=9),np.sum((a>=10)&(a<=19)),np.sum((a>=20)&(a<=29)),np.sum((a>=30)&(a<=39)),np.sum(a>=40)]
        return abs(su-132)/20+abs(odd-3)*.35+abs(low-3)*.35+max(bands)*.15
    tail=sorted(map(int,pool),key=extreme,reverse=True)
    for i in tail:
        if i not in sel and allowed(i):sel.append(i)
        if len(sel)>=10:break
    for i in order:
        if i not in sel:sel.append(i)
        if len(sel)>=10:break
    return sel[:10]


def metrics(win,sel,C):
    W=set(map(int,win));U=set().union(*(set(map(int,C[i])) for i in sel));uc=len(W&U);best=max(len(W&set(map(int,C[i]))) for i in sel)
    w3=set(itertools.combinations(tuple(sorted(W)),3));w4=set(itertools.combinations(tuple(sorted(W)),4))
    s3=set(c for i in sel for c in itertools.combinations(tuple(map(int,C[i])),3));s4=set(c for i in sel for c in itertools.combinations(tuple(map(int,C[i])),4))
    return {'union':uc,'core3':int(bool(w3&s3)),'core4':int(bool(w4&s4)),'best':best,'assembly':best/uc if uc else 0.0}


def summarize(R):
    return {'draws':len(R),'mean_union':float(np.mean([x['union'] for x in R])),'union5plus':sum(x['union']>=5 for x in R),'union6':sum(x['union']==6 for x in R),
            'core3_capture':sum(x['core3'] for x in R),'core4_capture':sum(x['core4'] for x in R),'mean_assembly':float(np.mean([x['assembly'] for x in R])),
            'mean_best':float(np.mean([x['best'] for x in R])),'d3':sum(x['best']>=3 for x in R),'d4':sum(x['best']>=4 for x in R),'d5':sum(x['best']>=5 for x in R),'d6':sum(x['best']>=6 for x in R)}


def main():
    rows=fetch_history();draw_to_idx={d:i for i,(d,_,_) in enumerate(rows)}
    C=fixed_sample();st,inc,q=build_static(C);draws,bonus,npref,ppref,actual=hist_actual(rows,q);sizes,priors=prepare_priors(st)
    baseR=[];conR=[];detail=[]
    for draw in range(START_DRAW,END_DRAW+1):
        if draw not in draw_to_idx:continue
        t=draw_to_idx[draw]
        ss={}
        for h in (200,500,800):
            W=weights(t,h,actual,sizes,priors);ss[h]=stat_score(t,h,W,st,inc,draws,bonus,npref)
        pc=ppref[t]-ppref[max(0,t-300)];c5,c4=cores(C,pc)
        z=lambda x:(x-x.mean())/(x.std()+1e-9)
        comm=z(ss[500])+.20*z(c5)+.15*z(c4)
        agents={'stat200':topidx(ss[200],1500),'stat500':topidx(ss[500],1500),'stat800':topidx(ss[800],1500),'committee':topidx(comm,1500)}
        base=diverse(agents['committee'],C,10,1,2,4);con=consensus_portfolio(agents,{'committee':comm},C)
        mb=metrics(draws[t],base,C);mc=metrics(draws[t],con,C);baseR.append(mb);conR.append(mc)
        detail.append({'draw':draw,'baseline':mb,'consensus':mc})
        if len(detail)%25==0:print(draw,summarize(baseR),summarize(conR),flush=True)
    out={'method':'fixed 60k uniform candidate sample; no target winner injection','range':f'{START_DRAW}-{END_DRAW}','seed':SEED,'sample_size':NSAMPLE,
         'baseline_committee_diverse10':summarize(baseR),'consensus_assembly_v1_phase1':summarize(conR),'detail':detail,
         'caveat':'Approximate candidate-sample portfolio test. Not exact all-6,096,454 ranking; specialized Repair-Core/R2/Trajectory agents are deferred to phase2.'}
    Path(OUT/'loto6_consensus_phase1_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
