#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,itertools,json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v4',ROOT/'research'/'loto6_pool_shell_rescue_v4.py')
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
p=v4.p
OUT=ROOT/'research'/'results';OUT.mkdir(parents=True,exist_ok=True)
START,END,DEV_END=1628,2127,1877
A_THR=.025

def z(x):
    x=np.asarray(x,float);return (x-x.mean())/(x.std()+1e-9)
def shape_freq(draws,t,w=500):
    c=Counter(v4.band_tuple(draws[u]) for u in range(max(0,t-w),t));n=max(1,min(w,t));return {k:v/n for k,v in c.items()}
def num_support(agents,C,topn=500):
    s=np.zeros(44,float)
    for idx in agents.values():
        for rank,i in enumerate(idx[:topn],1):
            w=1/np.log2(rank+2)
            for x in C[int(i)]:s[int(x)]+=w
    return s
def rank_nums(s):return sorted(range(1,44),key=lambda x:(s[x],-x),reverse=True)
def overlap_ok(c,sel,max_common=4):
    cs=set(map(int,c));return all(len(cs & set(map(int,x)))<=max_common for x in sel)
def greedy_from_order(order,C,n,sel,max_common=4):
    out=[]
    for i in order:
        c=tuple(map(int,C[int(i)]))
        if c in out or c in sel:continue
        if not overlap_ok(c,sel+out,max_common):continue
        out.append(c)
        if len(out)>=n:break
    return out
def baseline10(comm,C):return greedy_from_order(np.argsort(-comm)[:10000],C,10,[],4)
def make_portfolio(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors):
    ss={}
    for h in (200,500,800):
        W=p.weights(t,h,actual,sizes,priors);ss[h]=p.stat_score(t,h,W,st,inc,draws,bonus,npref)
    pc=ppref[t]-ppref[max(0,t-300)];c5,c4=p.cores(C,pc);comm=z(ss[500])+.20*z(c5)+.15*z(c4)
    agents={'stat200':p.topidx(ss[200],1500),'stat500':p.topidx(ss[500],1500),'stat800':p.topidx(ss[800],1500),'committee':p.topidx(comm,1500)}
    rank=rank_nums(num_support(agents,C));P=set(rank[:32]);S=set(rank[32:42])
    sf=shape_freq(draws,t,500);recent={v4.band_tuple(draws[u]) for u in range(max(0,t-20),t)}
    arr=C.astype(np.int16);sums=arr.sum(1)
    sel=greedy_from_order(np.argsort(-comm)[:8000],C,4,[],4);lanes=['main']*len(sel)
    struct_idx=[]
    for i,row in enumerate(arr):
        sh=v4.band_tuple(row);f=sf.get(sh,0.0)
        if sums[i]>=120 and f>=A_THR and sh not in recent:struct_idx.append(i)
    if struct_idx:
        order=np.array(struct_idx)[np.argsort(-comm[np.array(struct_idx)])]
        add=greedy_from_order(order[:12000],C,2,sel,4);sel+=add;lanes+=['structural']*len(add)
    resc=[]
    for i,row in enumerate(arr):
        r=set(map(int,row))
        if not r <= (P|S):continue
        sc=len(r&S)
        if sc in (1,2):resc.append(i)
    if resc:
        rr=np.array(resc);score=comm[rr]+0.10*z(ss[800][rr]);order=rr[np.argsort(-score)]
        add=greedy_from_order(order[:15000],C,2,sel,4);sel+=add;lanes+=['rescue']*len(add)
    co=np.argsort(-comm)
    add=greedy_from_order(co[2000:15000],C,1,sel,3);sel+=add;lanes+=['committee_tail']*len(add)
    topcomm=set(map(int,co[:2000]));eo=[int(i) for i in np.argsort(-ss[800])[:20000] if int(i) not in topcomm]
    add=greedy_from_order(eo,C,1,sel,3);sel+=add;lanes+=['explorer']*len(add)
    if len(sel)<10:
        add=greedy_from_order(co[:20000],C,10-len(sel),sel,4);sel+=add;lanes+=['fallback']*len(add)
    return sel[:10],lanes[:10],baseline10(comm,C)
def eval_set(tickets,winner):
    ws=set(map(int,winner));hits=[len(ws&set(x)) for x in tickets]
    union=set().union(*(set(map(int,x)) for x in tickets)) if tickets else set()
    recall=len(ws&union)
    return (max(hits) if hits else 0),hits,recall,sorted(ws&union),sorted(ws-union),len(union)
def summarize(rows,prefix):
    n=len(rows);best=np.array([r[f'{prefix}_best'] for r in rows]);recall=np.array([r[f'{prefix}_winner_recall'] for r in rows])
    out={'n':n,f'{prefix}_mean_best':float(best.mean()),f'{prefix}_mean_winner_recall':float(recall.mean()),f'{prefix}_median_winner_recall':float(np.median(recall))}
    for k in range(0,7):out[f'{prefix}_winner_recall_{k}of6']=int(np.sum(recall==k))
    out[f'{prefix}_winner_recall_5plus']=int(np.sum(recall>=5));out[f'{prefix}_winner_recall_6of6']=int(np.sum(recall==6))
    for k in (3,4,5,6):out[f'{prefix}_d{k}plus']=int(np.sum(best>=k))
    return out
def combine_summary(rows):
    p1=summarize(rows,'portfolio');b1=summarize(rows,'baseline');best=np.array([r['portfolio_best'] for r in rows]);base=np.array([r['baseline_best'] for r in rows])
    out={**p1,**{k:v for k,v in b1.items() if k!='n'}}
    out['paired_gain_d3']=int(np.sum((best>=3)&(base<3)));out['paired_loss_d3']=int(np.sum((best<3)&(base>=3)))
    out['paired_gain_d4']=int(np.sum((best>=4)&(base<4)));out['paired_loss_d4']=int(np.sum((best<4)&(base>=4)))
    return out
def main():
    hist=p.fetch_history();di={d:i for i,(d,_,_) in enumerate(hist)}
    C=p.fixed_sample();st,inc,q=p.build_static(C);draws,bonus,npref,ppref,actual=p.hist_actual(hist,q);sizes,priors=p.prepare_priors(st)
    rec=[]
    for draw in range(START,END+1):
        t=di[draw];port,lanes,base=make_portfolio(t,C,draws,bonus,npref,ppref,actual,st,inc,sizes,priors)
        pb,ph,pr,pr_in,pr_miss,pu=eval_set(port,draws[t]);bb,bh,br,br_in,br_miss,bu=eval_set(base,draws[t])
        lane_best={}
        for lane in set(lanes):
            vals=[ph[i] for i,x in enumerate(lanes) if x==lane];lane_best[lane]=max(vals) if vals else 0
        rec.append({'draw':draw,'part':'dev' if draw<=DEV_END else 'hold','winner':list(map(int,draws[t])),'portfolio_tickets':[list(x) for x in port],'baseline_tickets':[list(x) for x in base],'portfolio_best':pb,'baseline_best':bb,'portfolio_hits':ph,'baseline_hits':bh,'portfolio_winner_recall':pr,'baseline_winner_recall':br,'portfolio_winner_numbers_present':pr_in,'baseline_winner_numbers_present':br_in,'portfolio_winner_numbers_missing':pr_miss,'baseline_winner_numbers_missing':br_miss,'portfolio_union_size':pu,'baseline_union_size':bu,'lanes':lanes,'lane_best':lane_best})
        if draw%25==0:print(draw,pb,bb,pr,br,flush=True)
    out={'method':'4 Main + 2 Structural + 2 Rescue + 1 Committee-tail + 1 Explorer','range':f'{START}-{END}','candidate_base':'fixed60k sample; forward features only','dev':combine_summary([r for r in rec if r['part']=='dev']),'hold':combine_summary([r for r in rec if r['part']=='hold']),'all':combine_summary(rec),'detail':rec}
    path=OUT/'loto6_portfolio_multilane_v1_summary.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='detail'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
