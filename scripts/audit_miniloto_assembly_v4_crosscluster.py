import collections,itertools,json,runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']
committee_score=m.get('committee_score')
if committee_score is None:
    def committee_score(t):
        s1=m['a1_score'](t); s2=m['a2_score'](t); return s1,s2,m['z'](s1)+0.15*m['z'](s2)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}
def z(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s+1e-9)
def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]; return idx[np.lexsort((idx,-score[idx]))]
def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[];tc=collections.Counter();pc=collections.Counter();nc=collections.Counter()
    for idx in indices:
        idx=int(idx);q=tuple(map(int,combos[idx]));trs=list(itertools.combinations(q,3));prs=list(itertools.combinations(q,2))
        if any(tc[x]>=triple_cap for x in trs) or any(pc[x]>=pair_cap for x in prs) or any(nc[x]>=num_cap for x in q): continue
        sel.append(idx)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in q:nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for idx in indices:
            idx=int(idx)
            if idx not in used:
                sel.append(idx);used.add(idx)
                if len(sel)>=K:break
    return sel
def final10(score,top):
    base=[];cap=4
    for c in (2,3,4):
        tr=greedy(top[:1500],10,1,2,c,False)
        if len(tr)>=10:base=tr[:10];cap=c;break
    if len(base)<10:base=greedy(top[:1500],10,1,2,4,True);cap=4
    support=inc[top[:500]].sum(0).astype(float); union=set().union(*[set(map(int,combos[i])) for i in base])
    allowed=np.zeros(32,bool);allowed[list(union)]=True;pool=top[allowed[combos[top]].all(axis=1)]
    pool=np.array([int(i) for i in pool if int(i) not in set(base)],int)
    if len(pool):
        zz=z(support[combos[pool]].mean(1)); add=int(pool[np.argmax(score[pool]-0.10*zz)])
        best=None;bk=None
        for rem in base:
            keep=[x for x in base if x!=rem]+[add];nu=set();pu=set()
            for idx in keep:
                q=tuple(map(int,combos[idx]));nu.update(q);pu.update(itertools.combinations(q,2))
            key=(len(nu),len(pu),float(np.sum(score[keep])))
            if bk is None or key>bk:bk=key;best=keep
        base=best
    return base,cap
def hist_pair(t,w=300):
    M=np.zeros((32,32),float)
    for row in draws[max(0,t-w):t]:
        for a,b in itertools.combinations(map(int,row),2):M[a,b]+=1;M[b,a]+=1
    return M
def top_pair_residual(top):
    M=np.zeros((32,32),float);ns=np.zeros(32,float)
    for idx in top[:500]:
        q=tuple(map(int,combos[int(idx)]))
        for n in q:ns[n]+=1
        for a,b in itertools.combinations(q,2):M[a,b]+=1;M[b,a]+=1
    R=np.zeros_like(M)
    for a in range(1,32):
        for b in range(a+1,32):R[a,b]=R[b,a]=M[a,b]/np.sqrt(max(ns[a],1)*max(ns[b],1))
    return R
# Cross-cluster variants: no raw Final10 frequency reward.
VAR={
 'C1_cross':      {'cover':.40,'newpair':.30,'newtriple':.20,'bridge':.10,'hist':0,'comm':0},
 'C2_crossbridge':{'cover':.30,'newpair':.20,'newtriple':.15,'bridge':.25,'hist':.10,'comm':0},
 'C3_novel':      {'cover':.20,'newpair':.35,'newtriple':.35,'bridge':.10,'hist':0,'comm':0},
 'C4_softcomm':   {'cover':.30,'newpair':.25,'newtriple':.20,'bridge':.10,'hist':.05,'comm':.10},
 'C5_bridge':     {'cover':.25,'newpair':.20,'newtriple':.10,'bridge':.35,'hist':.10,'comm':0},
}
def min_parent_cover(q,tickets):
    qs=set(q)
    for k in range(1,6):
        for ids in itertools.combinations(range(10),k):
            u=set()
            for j in ids:u.update(tickets[j])
            if qs<=u:return k
    return 6
def assemblies(t,score,top,base):
    tickets=[tuple(map(int,combos[i])) for i in base];union=sorted(set().union(*map(set,tickets)));bset=set(base)
    seenp=set();seent=set()
    for q in tickets:seenp.update(itertools.combinations(q,2));seent.update(itertools.combinations(q,3))
    R=top_pair_residual(top);H=hist_pair(t,300)
    idxs=[];cover=[];newp=[];newt=[];bridge=[];hist=[];comm=[]
    for q in itertools.combinations(union,5):
        idx=combo_index[q]
        if idx in bset:continue
        idxs.append(idx); cover.append(min_parent_cover(q,tickets))
        ps=list(itertools.combinations(q,2));ts=list(itertools.combinations(q,3))
        newp.append(sum(p not in seenp for p in ps));newt.append(sum(x not in seent for x in ts))
        bridge.append(sum(R[a,b] for a,b in ps));hist.append(sum(H[a,b] for a,b in ps));comm.append(score[idx])
    idxs=np.asarray(idxs,int); feats={'cover':z(cover),'newpair':z(newp),'newtriple':z(newt),'bridge':z(bridge),'hist':z(hist),'comm':z(comm)}
    out={}
    for name,w in VAR.items():
        s=sum(w[k]*feats[k] for k in w);order=np.argsort(-s,kind='stable');out[name]=[int(idxs[i]) for i in order[:5]]
    return out
def metrics(ids,win):
    ws=set(win);ts=[set(map(int,combos[i])) for i in ids];return max(len(ws&q) for q in ts),len(ws&set().union(*ts))
def safe_remove(base,a1,score):
    best=None;bk=None
    for rem in base:
        keep=[x for x in base if x!=rem]+[a1];nu=set();pu=set();tu=set()
        for idx in keep:
            q=tuple(map(int,combos[idx]));nu.update(q);pu.update(itertools.combinations(q,2));tu.update(itertools.combinations(q,3))
        key=(len(nu),len(pu),len(tu),float(np.sum(score[keep])))
        if bk is None or key>bk:bk=key;best=rem
    return best
def eval_range(a,b):
    rows=[]
    for rr in range(a,b+1):
        t=rr-1;_,_,cs=committee_score(t);top=top_indices(cs,5000);base,cap=final10(cs,top);win=tuple(map(int,draws[t]));bb,bu=metrics(base,win);aa=assemblies(t,cs,top,base);vr={}
        for name,cands in aa.items():
            a1=cands[0];ab,au=metrics(base+[a1],win);rem=safe_remove(base,a1,cs);repl=[x for x in base if x!=rem]+[a1];rb,ru=metrics(repl,win)
            vr[name]={'assembly1':list(map(int,combos[a1])),'top5':[list(map(int,combos[x])) for x in cands],'add1_best':ab,'replace1_best':rb,'add1_union':au,'replace1_union':ru,'removed_ticket':list(map(int,combos[rem]))}
        rows.append({'draw':rr,'winner':list(win),'baseline_best':bb,'baseline_union':bu,'cap':cap,'variants':vr})
        if rr%25==0:print('done',rr,flush=True)
    return rows
def summ(rows,name,mode):
    k='add1_best' if mode=='add1' else 'replace1_best';v=[r['variants'][name][k] for r in rows];b=[r['baseline_best'] for r in rows];focus=[i for i,r in enumerate(rows) if r['baseline_union']==5 and r['baseline_best']<=2]
    return {'n':len(rows),'best3plus':sum(x>=3 for x in v),'best4plus':sum(x>=4 for x in v),'best5':sum(x>=5 for x in v),'gain3':sum(x>=3 and y<3 for x,y in zip(v,b)),'loss3':sum(x<3 and y>=3 for x,y in zip(v,b)),'gain4':sum(x>=4 and y<4 for x,y in zip(v,b)),'loss4':sum(x<4 and y>=4 for x,y in zip(v,b)),'gain5':sum(x>=5 and y<5 for x,y in zip(v,b)),'loss5':sum(x<5 and y>=5 for x,y in zip(v,b)),'assembly_focus_n':len(focus),'assembly_focus_to3plus':sum(v[i]>=3 for i in focus),'assembly_focus_to4plus':sum(v[i]>=4 for i in focus)}
dev=eval_range(1100,1299);ds={n:{m:summ(dev,n,m) for m in ('add1','replace1')} for n in VAR}
def key(n):
    s=ds[n]['add1'];return (s['assembly_focus_to4plus'],s['assembly_focus_to3plus'],s['best5'],s['best4plus'],s['best3plus'],-s['loss3'])
sel=max(VAR,key=key);print('SELECTED',sel,ds[sel]['add1'],flush=True)
test=eval_range(1300,1399);ts={n:{m:summ(test,n,m) for m in ('add1','replace1')} for n in VAR}
base={'n':len(test),'best3plus':sum(r['baseline_best']>=3 for r in test),'best4plus':sum(r['baseline_best']>=4 for r in test),'best5':sum(r['baseline_best']>=5 for r in test),'assembly_focus_n':sum(r['baseline_union']==5 and r['baseline_best']<=2 for r in test)}
focus={str(rr):next(r for r in test if r['draw']==rr) for rr in (1381,1385,1399)}
out={'definition':'Assembly-v4 cross-cluster pre-draw audit. No raw Final10-frequency reward. Development 1100-1299 selects variant; test 1300-1399 untouched.','principle':'Recombine numbers across different parent tickets/clusters, reward unseen pair/triple links, and use only weak normalized bridge support.','variants':VAR,'selected_variant':sel,'development':{'range':[1100,1299],'summary':ds},'test':{'range':[1300,1399],'baseline':base,'summary':ts,'selected':ts[sel]},'focus_draws':focus,'test_rows':test}
p=ROOT/'data'/'miniloto-assembly-v4-crosscluster-audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('WROTE',p);print('TEST',ts[sel])