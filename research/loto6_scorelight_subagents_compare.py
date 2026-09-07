#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('f',ROOT/'research'/'loto6_pool_coverage_scorematch_compare_fixed10.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
m=f.m
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
CAL_START=1628;EVAL_START=1828;END=2135
PAIR_POS=list(itertools.combinations(range(6),2));TRI_POS=list(itertools.combinations(range(6),3))
QUOTA={'A':3,'B':3,'C':2,'D':2}


def precompute(C):
    pairids=np.empty((len(C),len(PAIR_POS)),np.int32)
    for j,(a,b) in enumerate(PAIR_POS):pairids[:,j]=C[:,a]*44+C[:,b]
    tripids=np.empty((len(C),len(TRI_POS)),np.int32)
    for j,(a,b,c) in enumerate(TRI_POS):tripids[:,j]=(C[:,a]*44+C[:,b])*44+C[:,c]
    rawsum=C.sum(1).astype(np.int16);rng=(C[:,-1]-C[:,0]).astype(np.int8);odd=(C%2).sum(1).astype(np.int8);con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
    return pairids,tripids,rawsum,rng,odd,con

def pair_total_for_rows(X,pc):
    out=[]
    for row in X:
        s=0
        for a,b in itertools.combinations(map(int,row),2):s+=int(pc[a,b])
        out.append(s)
    return np.asarray(out,float)

def profile_mask(draws,t,rawsum,rng,odd,con):
    H=draws[max(0,t-500):t]
    hs=H.sum(1);hr=H[:,-1]-H[:,0];ho=(H%2).sum(1);hc=(np.diff(H,axis=1)==1).sum(1)
    slo,shi=np.quantile(hs,[.10,.90]);rlo,rhi=np.quantile(hr,[.10,.90]);clim=int(np.quantile(hc,.90))
    cnt=np.bincount(ho,minlength=7);common={i for i,x in enumerate(cnt) if x>=max(3,.05*len(H))}
    mask=(rawsum>=slo)&(rawsum<=shi)&(rng>=rlo)&(rng<=rhi)&(con<=clim)&np.isin(odd,list(common))
    return mask,{'sum':[float(slo),float(shi)],'range':[float(rlo),float(rhi)],'consec_max':clim,'odd_common':sorted(common)}

def freq_strata_mask(C,inc,npref,t,prevbo):
    freq=npref[t]-npref[max(0,t-100)]
    nums=np.asarray([x for x in range(1,44) if x!=prevbo],int)
    order=nums[np.lexsort((nums,freq[nums]))]
    low=set(map(int,order[:14]));mid=set(map(int,order[14:28]));high=set(map(int,order[28:]))
    cl=inc[:,list(low)].sum(1);cm=inc[:,list(mid)].sum(1);ch=inc[:,list(high)].sum(1)
    return (cl>=1)&(cm>=1)&(ch>=1)&(cl<=4)&(cm<=4)&(ch<=4),{'low':sorted(low),'mid':sorted(mid),'high':sorted(high)}

def greedy(ids,C,pairids,tripids,layers,shapes,mode='pair',ptotal=None,pmed=0.0,n=10):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)==0:return []
    sel=[];chosen=set();usedn=np.zeros(44,bool);usedp=np.zeros(44*44,bool);usedt=np.zeros(44*44*44,bool)
    nc=np.zeros(44,np.int16);pc=np.zeros(44*44,np.int16);tc=np.zeros(44*44*44,np.int16);lc=Counter();sc=Counter()
    passes=[(True,True,True),(False,True,True),(False,False,True),(False,False,False)]
    for strict,layer_cap,shape_cap in passes:
        while len(sel)<n:
            av=np.asarray([i for i in ids if int(i) not in chosen],dtype=np.int64)
            if len(av)==0:break
            ok=np.ones(len(av),bool)
            if layer_cap:ok &= np.asarray([lc[str(layers[i])] < QUOTA[str(layers[i])] for i in av])
            if shape_cap:ok &= np.asarray([sc[int(shapes[i])] < 2 for i in av])
            if strict:
                ok &= (nc[C[av]].max(1)<4)
                ok &= (pc[pairids[av]].max(1)<2)
                ok &= (tc[tripids[av]].max(1)<1)
            av=av[ok]
            if len(av)==0:break
            newn=(~usedn[C[av]]).sum(1);newp=(~usedp[pairids[av]]).sum(1);newt=(~usedt[tripids[av]]).sum(1);load=nc[C[av]].sum(1)
            if mode=='number': keys=(av,load,-newt,-newp,-newn)
            elif mode=='graph' and ptotal is not None:
                pd=np.abs(ptotal[av]-pmed);keys=(av,load,pd,-newn,-newt,-newp)
            else: keys=(av,load,-newn,-newt,-newp)
            j=int(np.lexsort(keys)[0]);i=int(av[j]);chosen.add(i);sel.append(i);row=C[i];pp=pairids[i];tt=tripids[i]
            usedn[row]=True;usedp[pp]=True;usedt[tt]=True;nc[row]+=1;pc[pp]+=1;tc[tt]+=1;lc[str(layers[i])]+=1;sc[int(shapes[i])]+=1
        if len(sel)>=n:return sel[:n]
    # deterministic final fallback
    for i in ids:
        i=int(i)
        if i not in chosen:sel.append(i);chosen.add(i)
        if len(sel)>=n:break
    return sel[:n]

def round_robin(routes,n=10):
    names=['structure','profile','graph','freq'];sel=[];used=set();pos=defaultdict(int)
    while len(sel)<n:
        moved=False
        for name in names:
            arr=routes[name]
            while pos[name]<len(arr) and arr[pos[name]] in used:pos[name]+=1
            if pos[name]<len(arr):
                i=int(arr[pos[name]]);pos[name]+=1;sel.append(i);used.add(i);moved=True
                if len(sel)>=n:break
        if not moved:break
    for name in names:
        for i in routes[name]:
            if int(i) not in used:sel.append(int(i));used.add(int(i))
            if len(sel)>=n:return sel[:n]
    return sel[:n]

def adaptive_vote_pool(routes,base_ids,profile_ids,C,inc,prevbo,min_candidates=250):
    vote=Counter()
    for arr in routes.values():
        for i in arr:
            for x in map(int,C[i]):vote[x]+=1
    nums=[x for x in range(1,44) if x!=prevbo]
    nums.sort(key=lambda x:(-vote[x],x))
    chosen_ids=None;eff=None;pool=None
    for k in range(22,33):
        p=nums[:k];pm=(inc[:,p].sum(1)==6)
        ids=np.asarray([i for i in profile_ids if pm[i]],dtype=np.int64)
        if len(ids)<min_candidates:ids=np.asarray([i for i in base_ids if pm[i]],dtype=np.int64)
        if len(ids)>=min_candidates or k==32:
            chosen_ids=ids;eff=k;pool=p;break
    return chosen_ids,eff,pool,dict(vote)

def met(win,sel,C,pairids,tripids):
    W=set(map(int,win));tickets=[set(map(int,C[i])) for i in sel];best=max([len(W&t) for t in tickets] or [0]);U=set().union(*tickets) if tickets else set();wu=len(W&U)
    wp=set(itertools.combinations(sorted(W),2));wt=set(itertools.combinations(sorted(W),3));sp=set();st=set()
    for i in sel:
        row=tuple(map(int,C[i]));sp.update(itertools.combinations(row,2));st.update(itertools.combinations(row,3))
    return {'n_tickets':len(sel),'best':best,'winner_union':wu,'portfolio_numbers':len(U),'winner_pairs':len(wp&sp),'winner_triples':len(wt&st),'portfolio_pairs':len(sp),'portfolio_triples':len(st),'d3':int(best>=3),'d4':int(best>=4),'d5':int(best>=5),'d6':int(best>=6)}
def summary(rr):
    if not rr:return {}
    keys=['best','winner_union','portfolio_numbers','winner_pairs','winner_triples','portfolio_pairs','portfolio_triples']
    out={'n':len(rr),'mean_tickets':float(np.mean([x['n_tickets'] for x in rr]))}
    for k in keys:out['mean_'+k]=float(np.mean([x[k] for x in rr]))
    out.update({'union5plus':sum(x['winner_union']>=5 for x in rr),'union6':sum(x['winner_union']==6 for x in rr),'d3':sum(x['d3'] for x in rr),'d4':sum(x['d4'] for x in rr),'d5':sum(x['d5'] for x in rr),'d6':sum(x['d6'] for x in rr)})
    return out

def paired(a,b,key):
    return {'gain':sum(y[key]>x[key] for x,y in zip(a,b)),'loss':sum(y[key]<x[key] for x,y in zip(a,b)),'tie':sum(y[key]==x[key] for x,y in zip(a,b))}

def main():
    rows=m.local_rows();di={d:i for i,(d,_,_) in enumerate(rows)}
    C=m.p.fixed_sample();st,inc,q=m.p.build_static(C);draws,bonus,npref,ppref,actual=m.p.hist_actual(rows,q);pairids,tripids,rawsum,rng,odd,con=precompute(C)
    R={k:[] for k in ('structure','profile','graph','freq','consensus','adaptive_vote')};details=[];effp=[]
    for draw in range(CAL_START,END+1):
        if draw not in di:continue
        t=di[draw];prev=set(map(int,draws[t-1]));prevbo=int(bonus[t-1]);ash,_,_,_=m.allowed_shapes(draws,t);hall=Counter(m.band(r) for r in draws[:t]);lm=m.build_layer_map(hall,t);layers=lm[st['band']];shapes=st['band']
        po=inc[:,list(prev)].sum(1);base=np.isin(st['band'],ash)&(po<=1)&(inc[:,prevbo]==0);base_ids=np.where(base)[0]
        profmask,prof=m.profile_mask(draws,t,rawsum,rng,odd,con) if False else profile_mask(draws,t,rawsum,rng,odd,con)
        profile_ids=np.where(base&profmask)[0]
        fmask,strata=freq_strata_mask(C,inc,npref,t,prevbo);freq_ids=np.where(base&fmask)[0]
        pairc=ppref[t]-ppref[max(0,t-300)];flat=pairc.reshape(-1);ptotal=flat[pairids].sum(1)
        H=draws[max(0,t-200):t];hp=pair_total_for_rows(H,pairc);plo,phi=np.quantile(hp,[.10,.90]);pmed=float(np.median(hp));graph_ids=np.where(base&profmask&(ptotal>=plo)&(ptotal<=phi))[0]
        # independent score-light agents. No Stat/Committee score is used here.
        routes={}
        routes['structure']=greedy(base_ids,C,pairids,tripids,layers,shapes,'pair')
        routes['profile']=greedy(profile_ids if len(profile_ids)>=20 else base_ids,C,pairids,tripids,layers,shapes,'pair')
        routes['graph']=greedy(graph_ids if len(graph_ids)>=20 else profile_ids,C,pairids,tripids,layers,shapes,'graph',ptotal,pmed)
        routes['freq']=greedy(freq_ids if len(freq_ids)>=20 else base_ids,C,pairids,tripids,layers,shapes,'number')
        cons=round_robin(routes,10)
        avid,ek,pool,vote=adaptive_vote_pool(routes,base_ids,profile_ids,C,inc,prevbo)
        avsel=greedy(avid,C,pairids,tripids,layers,shapes,'pair')
        if draw>=EVAL_START:
            win=tuple(map(int,draws[t]));sels={**routes,'consensus':cons,'adaptive_vote':avsel}
            rec={'draw':draw,'winner':list(win),'previous_bonus':prevbo,'base_count':len(base_ids),'profile_count':len(profile_ids),'graph_count':len(graph_ids),'freq_count':len(freq_ids),'adaptive_pool_size':ek,'adaptive_pool':pool}
            effp.append(ek)
            for name,sel in sels.items():
                mm=met(win,sel,C,pairids,tripids);R[name].append(mm);rec[name]=mm
                if draw in (2134,2135):rec[name+'_tickets']=[list(map(int,C[i])) for i in sel]
            details.append(rec)
        if draw%25==0:print(draw,flush=True)
    def block(lo,hi):
        return {name:summary([x for x,d in zip(rr,[r['draw'] for r in details]) if lo<=d<=hi]) for name,rr in R.items()}
    allsum={k:summary(v) for k,v in R.items()}
    early={k:summary(v[:154]) for k,v in R.items()};late={k:summary(v[154:]) for k,v in R.items()}
    out={'definition':'Score-light/no-Committee final-selection study on fixed deterministic 60k sample. All six agents use only pre-draw data. Base structural gate: temporal allowed band shape, previous main-number overlap <=1, previous bonus excluded. No hard sum>previous-sum rule. Structure maximizes portfolio pair/triple/number novelty. Profile additionally gates to rolling historical winner central structural ranges. Graph adds rolling pair-cooccurrence range and uses pair-profile only as a lexicographic tie-break, not a Committee score. Freq balances low/mid/high recent-100 frequency strata. Consensus round-robins the four independent routes. Adaptive_vote forms the smallest 22-32 number pool with sufficient candidates from sub-agent ticket votes, then assembles 10 by pair coverage.','evaluation':[EVAL_START,END],'n_draws':len(details),'summary':allsum,'early_1828_1981':early,'late_1982_2135':late,'adaptive_pool':{'mean':float(np.mean(effp)),'median':float(np.median(effp)),'distribution':{str(k):sum(x==k for x in effp) for k in range(22,33)}},'paired_vs_structure':{name:{k:paired(R['structure'],R[name],k) for k in ('best','winner_union','winner_pairs','winner_triples','d3','d4')} for name in R if name!='structure'},'draw2134':next((x for x in details if x['draw']==2134),None),'draw2135':next((x for x in details if x['draw']==2135),None)}
    (OUT/'loto6_scorelight_subagents_compare.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
