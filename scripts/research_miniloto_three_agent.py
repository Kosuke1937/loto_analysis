import collections, itertools, json, math, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']; N=m['N']
a1_score=m['a1_score']; a2_score=m['a2_score']; z=m['z']; rpair=m['rpair']
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}

DEV=range(1000,1200)
TEST=range(1200,1400)

def committee(t):
    s1=a1_score(t); s2=a2_score(t); return z(s1)+0.15*z(s2)

def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def basic_greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for ii in indices:
        ii=int(ii); ns=tuple(map(int,combos[ii])); trs=list(itertools.combinations(ns,3)); prs=list(itertools.combinations(ns,2))
        if any(tc[x]>=triple_cap for x in trs): continue
        if any(pc[x]>=pair_cap for x in prs): continue
        if any(nc[x]>=num_cap for x in ns): continue
        sel.append(ii)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in ns: nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for ii in indices:
            ii=int(ii)
            if ii not in used:
                sel.append(ii); used.add(ii)
                if len(sel)>=K: break
    return sel

def current_state234(score,top5000):
    for cap in (2,3,4):
        x=basic_greedy(top5000[:1500],10,1,2,cap,False)
        if len(x)>=10: return x[:10]
    return basic_greedy(top5000[:1500],10,1,2,4,True)

def coverage_greedy(score,top5000,lam):
    cand=list(map(int,top5000[:1500])); s=score[cand]; sz=(s-s.mean())/(s.std()+1e-9)
    sel=[]; used=set(); pc=collections.Counter(); tc=collections.Counter()
    for _ in range(10):
        best=None; bestv=-1e99
        for pos,ii in enumerate(cand):
            if ii in sel: continue
            ns=tuple(map(int,combos[ii])); prs=list(itertools.combinations(ns,2)); trs=list(itertools.combinations(ns,3))
            if any(pc[x]>=2 for x in prs) or any(tc[x]>=1 for x in trs): continue
            new=sum(n not in used for n in ns)
            v=float(sz[pos])+lam*new
            if v>bestv: bestv=v; best=ii
        if best is None: break
        sel.append(best); ns=tuple(map(int,combos[best])); used.update(ns)
        for x in itertools.combinations(ns,2): pc[x]+=1
        for x in itertools.combinations(ns,3): tc[x]+=1
    if len(sel)<10:
        for ii in cand:
            if ii not in sel:
                sel.append(ii)
                if len(sel)>=10: break
    return sel[:10]

def union_hit(ids,t):
    u=set(); [u.update(map(int,combos[i])) for i in ids]
    return len(u & set(map(int,draws[t]))), len(u), u

def main_metrics(rows):
    return {'n':len(rows),'union5':sum(r['hit']==5 for r in rows),'union4plus':sum(r['hit']>=4 for r in rows),
            'avg_union':round(float(np.mean([r['usize'] for r in rows])),3)}

# ---------- Agent 1: choose Main10 method on DEV ----------
main_methods={'state234':None,'cov025':0.25,'cov050':0.50,'cov075':0.75,'cov100':1.00}
cache={}
for rr in list(DEV)+list(TEST):
    t=rr-1; cs=committee(t); top=top_indices(cs,5000)
    cache[rr]={'t':t,'score':cs,'top':top}

main_dev={}
for name,lam in main_methods.items():
    rows=[]
    for rr in DEV:
        c=cache[rr]; ids=current_state234(c['score'],c['top']) if lam is None else coverage_greedy(c['score'],c['top'],lam)
        hit,usize,u=union_hit(ids,c['t']); rows.append({'hit':hit,'usize':usize})
    main_dev[name]=main_metrics(rows)
# lexicographic: union5, union4+, then smaller union size
best_main=max(main_dev,key=lambda k:(main_dev[k]['union5'],main_dev[k]['union4plus'],-main_dev[k]['avg_union']))

def get_main(rr):
    c=cache[rr]; lam=main_methods[best_main]
    return current_state234(c['score'],c['top']) if lam is None else coverage_greedy(c['score'],c['top'],lam)

# ---------- Agent 2: compression methods ----------
def num_features(rr,ids):
    c=cache[rr]; top500=c['top'][:500]; score=c['score']
    mf=np.zeros(32,float); ts=np.zeros(32,float); pq=np.full(32,-10.0,float)
    for ii in ids:
        for n in map(int,combos[ii]):
            mf[n]+=1; pq[n]=max(pq[n],float(score[ii]))
    ts=inc[top500].sum(0).astype(float)
    return mf,ts,pq

def z31(x):
    y=x[1:32]; return np.r_[0.0,(y-y.mean())/(y.std()+1e-9)]

def compress_pool(rr,ids,size,method):
    _,_,u=union_hit(ids,cache[rr]['t']); mf,ts,pq=num_features(rr,ids)
    if method=='mainfreq': sc=z31(mf)
    elif method=='topsupport': sc=z31(ts)
    elif method=='parent': sc=z31(pq)
    elif method=='hybrid': sc=z31(mf)+0.5*z31(ts)+0.5*z31(pq)
    elif method=='satellite':
        sc=z31(mf)+0.4*z31(ts)+0.8*z31(pq)
        # protect singleton digits from strong parents
        sc += ((mf==1)&(z31(pq)>0.5))*0.75
    else: raise ValueError(method)
    ranked=sorted(u,key=lambda n:(-sc[n],n))
    return set(ranked[:min(size,len(ranked))])

pool_methods=['mainfreq','topsupport','parent','hybrid','satellite']
pool_sizes=[16,18,20,22]
pool_dev={}
for method in pool_methods:
    for size in pool_sizes:
        eligible=retain5=retain4=0; total4=0
        for rr in DEV:
            ids=get_main(rr); t=cache[rr]['t']; win=set(map(int,draws[t])); _,_,u=union_hit(ids,t)
            p=compress_pool(rr,ids,size,method); h=len(win&p)
            if win<=u:
                eligible+=1; retain5+=int(h==5)
            if len(win&u)>=4:
                total4+=1; retain4+=int(h>=4)
        pool_dev[f'{method}_{size}']={'eligible_union5':eligible,'retain5':retain5,'retain5_rate':round(retain5/max(1,eligible),4),
                                      'eligible_union4plus':total4,'retain4plus':retain4,'retain4plus_rate':round(retain4/max(1,total4),4),'size':size,'method':method}
best_pool=max(pool_dev,key=lambda k:(pool_dev[k]['retain5_rate'],pool_dev[k]['retain4plus_rate'],-pool_dev[k]['size']))
BP=pool_dev[best_pool]

# ---------- Agent 3: Assembly score grid ----------
def triple_counter(t,window=500):
    lo=max(0,t-window); cc=collections.Counter()
    for row in draws[lo:t]:
        for tri in itertools.combinations(map(int,row),3): cc[tri]+=1
    return cc

def assembly_features(rr,ids,pool):
    t=cache[rr]['t']; score=cache[rr]['score']; mf,ts,pq=num_features(rr,ids); pc=rpair(t,300); tc=triple_counter(t,500)
    arr=np.array(list(itertools.combinations(sorted(pool),5)),dtype=np.int16)
    idx=np.array([combo_index[tuple(map(int,a))] for a in arr],dtype=int)
    fC=score[idx].astype(float)
    fM=np.array([np.mean([mf[int(n)] for n in a]) for a in arr])
    fP=np.array([sum(pc[int(a[i]),int(a[j])] for i,j in itertools.combinations(range(5),2)) for a in arr],float)
    fT=np.array([sum(tc[tuple(sorted((int(a[i]),int(a[j]),int(a[k]))))] for i,j,k in itertools.combinations(range(5),3)) for a in arr],float)
    fQ=np.array([np.mean([pq[int(n)] for n in a]) for a in arr])
    def zz(v): return (v-v.mean())/(v.std()+1e-9)
    return arr,idx,{'C':zz(fC),'M':zz(fM),'P':zz(fP),'T':zz(fT),'Q':zz(fQ)}

weights={
 'committee':(1,0,0,0,0),
 'support':(0.7,0.8,0.2,0,0.4),
 'pair':(0.7,0.3,0.8,0.2,0.3),
 'triple':(0.6,0.2,0.6,0.8,0.3),
 'balanced':(0.7,0.5,0.5,0.3,0.5),
 'assembly':(0.5,0.6,0.7,0.5,0.5),
}

def eval_assembly(period,wname):
    wc,wm,wp,wt,wq=weights[wname]; ex5=0; hit4=0; condn=0; ranks=[]
    for rr in period:
        ids=get_main(rr); t=cache[rr]['t']; win=tuple(map(int,draws[t])); _,_,u=union_hit(ids,t)
        pool=compress_pool(rr,ids,BP['size'],BP['method'])
        if set(win)<=pool: condn+=1
        arr,idx,F=assembly_features(rr,ids,pool)
        sc=wc*F['C']+wm*F['M']+wp*F['P']+wt*F['T']+wq*F['Q']
        order=np.argsort(-sc); top10=arr[order[:10]]
        matches=np.array([len(set(map(int,a))&set(win)) for a in top10])
        ex5+=int(np.max(matches)==5); hit4+=int(np.max(matches)>=4)
        if set(win)<=pool:
            wi=np.where(np.all(arr==np.array(win),axis=1))[0]
            if len(wi): ranks.append(1+int(np.sum(sc>sc[wi[0]])))
    return {'n':len(list(period)),'pool_contains_winner':condn,'assembly_top10_exact5':ex5,'assembly_top10_4plus':hit4,
            'median_winner_rank_conditional':None if not ranks else float(np.median(ranks))}

assembly_dev={k:eval_assembly(DEV,k) for k in weights}
best_assembly=max(assembly_dev,key=lambda k:(assembly_dev[k]['assembly_top10_exact5'],assembly_dev[k]['assembly_top10_4plus'],-(assembly_dev[k]['median_winner_rank_conditional'] or 1e9)))

# ---------- Frozen TEST ----------
def eval_main(period):
    rows=[]
    for rr in period:
        ids=get_main(rr); h,s,_=union_hit(ids,cache[rr]['t']); rows.append({'hit':h,'usize':s})
    return main_metrics(rows)

def eval_pool(period):
    eligible=retain5=eligible4=retain4=0
    for rr in period:
        ids=get_main(rr); t=cache[rr]['t']; win=set(map(int,draws[t])); _,_,u=union_hit(ids,t); p=compress_pool(rr,ids,BP['size'],BP['method']); h=len(win&p)
        if win<=u: eligible+=1; retain5+=int(h==5)
        if len(win&u)>=4: eligible4+=1; retain4+=int(h>=4)
    return {'eligible_union5':eligible,'retain5':retain5,'retain5_rate':round(retain5/max(1,eligible),4),
            'eligible_union4plus':eligible4,'retain4plus':retain4,'retain4plus_rate':round(retain4/max(1,eligible4),4)}

out={
 'protocol':{'development':[1000,1199],'fixed_test':[1200,1399],'excluded':[1400,1401],
             'note':'A/B Specialist and Cluster exact implementations were not available in repository search; this run uses exact reproducible Committee baseline only.'},
 'agent1':{'development_all':main_dev,'selected':best_main,'fixed_test':eval_main(TEST)},
 'agent2':{'development_all':pool_dev,'selected':best_pool,'fixed_test':eval_pool(TEST)},
 'agent3':{'development_all':assembly_dev,'selected':best_assembly,'fixed_test':eval_assembly(TEST,best_assembly)},
}
path=ROOT/'data'/'miniloto-three-agent-research.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
print('WROTE',path)
