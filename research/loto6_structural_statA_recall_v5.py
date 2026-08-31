#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json,math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v4',ROOT/'research'/'loto6_pool_shell_rescue_v4.py')
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
p=v4.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
K,R,M=32,10,3
A_THR=.025
THRESH=(100000,50000,20000,10000,5000,1000,500,100)

def all_combos():
    arr=np.fromiter((x for c in itertools.combinations(range(1,44),6) for x in c),dtype=np.uint8,count=6096454*6)
    return arr.reshape(-1,6)

def band_code_from_counts(b):
    b=np.asarray(b,dtype=np.int32)
    return (b[:,0]*2401+b[:,1]*343+b[:,2]*49+b[:,3]*7+b[:,4]).astype(np.int32)

def exact_static(C,q):
    s=C.sum(1).astype(np.int16); sb=np.clip((s-21)//5,0,50).astype(np.int16)
    odd=(C%2).sum(1).astype(np.int8); con=(np.diff(C,axis=1)==1).sum(1).astype(np.int8)
    b=np.stack([(C<=9).sum(1),((C>=10)&(C<=19)).sum(1),((C>=20)&(C<=29)).sum(1),((C>=30)&(C<=39)).sum(1),(C>=40).sum(1)],axis=1).astype(np.int32)
    band=band_code_from_counts(b); gap=np.digitize(np.std(np.diff(C,axis=1),axis=1),q).astype(np.int8)
    return {'rawsum':s,'sum':sb,'odd':odd,'band':band,'consec':con,'gap':gap},b

def shape_code(a):
    a=np.asarray(a); b=np.array([[np.sum(a<=9),np.sum((a>=10)&(a<=19)),np.sum((a>=20)&(a<=29)),np.sum((a>=30)&(a<=39)),np.sum(a>=40)]],dtype=np.int32)
    return int(band_code_from_counts(b)[0])

def shape_freq_codes(draws,t,w=500):
    lo=max(0,t-w); c=Counter(shape_code(draws[u]) for u in range(lo,t)); n=max(1,t-lo)
    return {k:v/n for k,v in c.items()}

def num_support(agents,Csample,topn=500):
    s=np.zeros(44,float)
    for idx in agents.values():
        for rank,i in enumerate(idx[:topn],1):
            w=1/np.log2(rank+2)
            for x in Csample[int(i)]:s[int(x)]+=w
    return s

def rank_nums(s): return sorted(range(1,44),key=lambda x:(s[x],-x),reverse=True)

def row_dynamic(row,prev,prev2,pbonus,hot):
    st=set(map(int,row));return len(st&prev),len(st&prev2),int(pbonus in st),len(st&hot)

def score_winner(row,W,q,prev,prev2,pbonus,hot):
    a=np.asarray(row,dtype=np.int16); raw=int(a.sum()); sb=int(np.clip((raw-21)//5,0,50));odd=int((a%2).sum());con=int((np.diff(a)==1).sum())
    band=shape_code(a); gap=int(np.digitize(np.std(np.diff(a)),q));po,p2,pb,hh=row_dynamic(a,prev,prev2,pbonus,hot)
    return float(W['sum'][sb]+W['odd'][odd]+W['band'][band]+W['consec'][con]+W['gap'][gap]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh])

def main():
    rows=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(rows)}
    Csample=p.fixed_sample();st_s,inc_s,q=p.build_static(Csample);draws,bonus,npref,ppref,actual=p.hist_actual(rows,q);sizes,priors=p.prepare_priors(st_s)
    C=all_combos();st,_=exact_static(C,q)
    rec=[]
    for draw in range(START,END+1):
        t=di[draw]
        ss={}
        for h in (200,500,800):
            W_h=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W_h,st_s,inc_s,draws,bonus,npref)
        pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(Csample,pc);z=lambda x:(x-x.mean())/(x.std()+1e-9);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
        agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
        rank=rank_nums(num_support(agents,Csample)); P=set(rank[:K]); S=set(rank[K:K+R]); universe=P|S
        inU=np.zeros(44,np.uint8);inS=np.zeros(44,np.uint8)
        for x in universe:inU[x]=1
        for x in S:inS[x]=1
        ucnt=inU[C].sum(1); scnt=inS[C].sum(1)
        sf=shape_freq_codes(draws,t,500);recent={shape_code(draws[u]) for u in range(max(0,t-20),t)}
        allowed_codes=np.array([code for code,f in sf.items() if f>=A_THR and code not in recent],dtype=np.int32)
        mask=(ucnt==6)&(scnt<=M)&(st['rawsum']>=120)&np.isin(st['band'],allowed_codes)
        idx=np.flatnonzero(mask);Ct=C[idx]
        W=p.weights(t,500,actual,sizes,priors)
        prev=set(map(int,draws[t-1]));prev2=set(map(int,draws[t-2])) if t>=2 else set();pbonus=int(bonus[t-1]);c300=npref[t]-npref[max(0,t-300)];hot=set((np.lexsort((np.arange(1,44),-c300[1:]))+1)[:15])
        prev_l=np.zeros(44,np.uint8);prev2_l=np.zeros(44,np.uint8);hot_l=np.zeros(44,np.uint8)
        for x in prev:prev_l[x]=1
        for x in prev2:prev2_l[x]=1
        for x in hot:hot_l[x]=1
        po=prev_l[Ct].sum(1);p2=prev2_l[Ct].sum(1);pb=(Ct==pbonus).any(1).astype(np.uint8);hh=hot_l[Ct].sum(1)
        sc=(W['sum'][st['sum'][idx]]+W['odd'][st['odd'][idx]]+W['band'][st['band'][idx]]+W['consec'][st['consec'][idx]]+W['gap'][st['gap'][idx]]+W['prev'][po]+W['prev2'][p2]+W['pbonus'][pb]+W['hot'][hh]).astype(np.float32)
        win=draws[t];wset=set(map(int,win));win_scnt=len(wset&S);win_struct=(wset<=universe and win_scnt<=M and int(win.sum())>=120 and shape_code(win) in set(map(int,allowed_codes)))
        rankA=None
        if win_struct:
            ws=score_winner(win,W,q,prev,prev2,pbonus,hot);rankA=1+int(np.sum(sc>ws))
        rec.append({'draw':draw,'part':'dev' if draw<=DEV_END else 'hold','structural_survive':bool(win_struct),'structural_candidates':int(len(idx)),'statA_rank':rankA})
        if draw%25==0:print(draw,len(idx),win_struct,rankA,flush=True)
    def summary(part):
        r=[x for x in rec if x['part']==part]; surv=[x for x in r if x['structural_survive']]
        out={'n':len(r),'structural_survive':len(surv),'structural_survive_rate':len(surv)/len(r),'mean_structural_candidates':float(np.mean([x['structural_candidates'] for x in r])),'median_structural_candidates':float(np.median([x['structural_candidates'] for x in r]))}
        for k in THRESH:
            c=sum(x['statA_rank'] is not None and x['statA_rank']<=k for x in r);out[f'recall_at_{k}']=c;out[f'recall_at_{k}_rate_all']=c/len(r);out[f'recall_at_{k}_conditional_struct']=c/len(surv) if surv else 0.0
        if surv:out['median_statA_rank_conditional']=float(np.median([x['statA_rank'] for x in surv]))
        return out
    out={'method':'Exact structural gate -> canonical Stat_A(A1/Stat500) ranking','range':f'{START}-{END}','definition':{'primary':K,'satellite':R,'max_satellite':M,'A_freq_min':A_THR,'sum_min':120,'shape_absent_prev20':True,'Stat_A':'rolling500, alpha75, clip +/-1.5; sum/odd/band/consecutive/gap/prev/prev2/prev_bonus/hot15'},'dev':summary('dev'),'hold':summary('hold'),'detail':rec}
    path=OUT/'loto6_structural_statA_recall_v5_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
