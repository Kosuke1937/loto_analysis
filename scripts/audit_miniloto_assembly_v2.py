import collections, itertools, json, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']; N=m['N']
committee_score=m['committee_score'] if 'committee_score' in m else None
if committee_score is None:
    def committee_score(t):
        s1=m['a1_score'](t); s2=m['a2_score'](t)
        return s1,s2,m['z'](s1)+0.15*m['z'](s2)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}

def z(x):
    x=np.asarray(x,dtype=float); s=x.std(); return (x-x.mean())/(s+1e-9)

def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        idx=int(idx); q=tuple(map(int,combos[idx]))
        trs=list(itertools.combinations(q,3)); prs=list(itertools.combinations(q,2))
        if any(tc[x]>=triple_cap for x in trs): continue
        if any(pc[x]>=pair_cap for x in prs): continue
        if any(nc[x]>=num_cap for x in q): continue
        sel.append(idx)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in q: nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for idx in indices:
            idx=int(idx)
            if idx not in used:
                sel.append(idx); used.add(idx)
                if len(sel)>=K: break
    return sel

def final10(score,top5000):
    base=[]; cap=4
    for c in (2,3,4):
        trial=greedy(top5000[:1500],10,1,2,c,False)
        if len(trial)>=10: base=trial[:10]; cap=c; break
    if len(base)<10: base=greedy(top5000[:1500],10,1,2,4,True); cap=4
    support=inc[top5000[:500]].sum(0).astype(float)
    union=set().union(*[set(map(int,combos[i])) for i in base])
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=top5000[allowed[combos[top5000]].all(axis=1)]
    pool=np.array([int(i) for i in pool if int(i) not in set(base)],dtype=int)
    if len(pool):
        zz=z(support[combos[pool]].mean(1))
        add=int(pool[np.argmax(score[pool]-0.10*zz)])
        best=None; bestkey=None
        for rem in base:
            keep=[x for x in base if x!=rem]+[add]
            nu=set(); pu=set()
            for idx in keep:
                q=tuple(map(int,combos[idx])); nu.update(q); pu.update(itertools.combinations(q,2))
            key=(len(nu),len(pu),float(np.sum(score[keep])))
            if bestkey is None or key>bestkey: bestkey=key; best=keep
        base=best
    return base,cap

def hist_pair(t,w=300):
    M=np.zeros((32,32),float); a=max(0,t-w)
    for row in draws[a:t]:
        for x,y in itertools.combinations(map(int,row),2): M[x,y]+=1; M[y,x]+=1
    return M

def pair500(top):
    M=np.zeros((32,32),float); ns=np.zeros(32,float)
    for idx in top[:500]:
        q=tuple(map(int,combos[int(idx)]))
        for n in q: ns[n]+=1
        for x,y in itertools.combinations(q,2): M[x,y]+=1; M[y,x]+=1
    # normalized residual-like support: avoids rewarding pairs only because both numbers are hubs
    R=np.zeros_like(M)
    for x in range(1,32):
        for y in range(x+1,32):
            denom=np.sqrt(max(ns[x],1)*max(ns[y],1))
            R[x,y]=R[y,x]=M[x,y]/denom
    return R,ns

def novelty(q,tickets):
    mx=max(len(set(q)&set(t)) for t in tickets)
    return -mx

def role_score(counts,target):
    # Compare sorted Final10 frequencies against a target role profile.
    c=sorted(counts)
    return -sum(abs(a-b) for a,b in zip(c,target))

# Targets intentionally keep 1-use satellite numbers eligible.
VARIANTS={
 'S1_sat_rich':  {'target':[1,1,1,2,3],'w_role':0.55,'w_bridge':0.25,'w_hist':0.10,'w_novel':0.10,'w_comm':0.00},
 'S2_balanced':  {'target':[1,1,2,3,3],'w_role':0.45,'w_bridge':0.25,'w_hist':0.10,'w_novel':0.15,'w_comm':0.05},
 'S3_broad':     {'target':[1,2,2,3,4],'w_role':0.35,'w_bridge':0.25,'w_hist':0.10,'w_novel':0.20,'w_comm':0.10},
 'S4_novel':     {'target':[1,1,2,3,4],'w_role':0.30,'w_bridge':0.20,'w_hist':0.10,'w_novel':0.35,'w_comm':0.05},
 'S5_bridge':    {'target':[1,1,1,3,4],'w_role':0.30,'w_bridge':0.40,'w_hist':0.15,'w_novel':0.15,'w_comm':0.00},
}

def assemblies(t,score,top,base):
    tickets=[tuple(map(int,combos[i])) for i in base]
    union=sorted(set().union(*map(set,tickets))); nf=collections.Counter(n for q in tickets for n in q)
    R,ns=pair500(top); H=hist_pair(t,300); bset=set(base)
    idxs=[]; role={k:[] for k in VARIANTS}; bridge=[]; hist=[]; nov=[]; comm=[]
    for q in itertools.combinations(union,5):
        idx=combo_index[q]
        if idx in bset: continue
        idxs.append(idx)
        counts=[nf[n] for n in q]
        for name,v in VARIANTS.items(): role[name].append(role_score(counts,v['target']))
        ps=list(itertools.combinations(q,2))
        bridge.append(sum(R[a,b] for a,b in ps))
        hist.append(sum(H[a,b] for a,b in ps))
        nov.append(novelty(q,tickets))
        comm.append(score[idx])
    idxs=np.asarray(idxs,dtype=int)
    zb=z(bridge); zh=z(hist); zn=z(nov); zc=z(comm)
    out={}
    for name,v in VARIANTS.items():
        zr=z(role[name])
        s=v['w_role']*zr+v['w_bridge']*zb+v['w_hist']*zh+v['w_novel']*zn+v['w_comm']*zc
        order=np.argsort(-s,kind='stable')
        out[name]=[int(idxs[i]) for i in order[:5]]
    return out

def metrics(ids,win):
    ws=set(win); ts=[set(map(int,combos[i])) for i in ids]
    return max(len(ws&q) for q in ts),len(ws&set().union(*ts))

def remove_for_replacement(base,a1,score):
    # Choose replacement by preserving number/pair coverage after insertion, not by baseline alone.
    best=None; keybest=None
    for rem in base:
        keep=[x for x in base if x!=rem]+[a1]
        nu=set(); pu=set()
        for idx in keep:
            q=tuple(map(int,combos[idx])); nu.update(q); pu.update(itertools.combinations(q,2))
        key=(len(nu),len(pu),float(np.sum(score[keep])))
        if keybest is None or key>keybest: keybest=key; best=rem
    return best

def eval_range(a,b):
    rows=[]
    for rr in range(a,b+1):
        t=rr-1; _,_,cs=committee_score(t); top=top_indices(cs,5000); base,cap=final10(cs,top)
        win=tuple(map(int,draws[t])); bb,bu=metrics(base,win); ass=assemblies(t,cs,top,base); vr={}
        for name,cands in ass.items():
            a1=cands[0]; ab,au=metrics(base+[a1],win); rem=remove_for_replacement(base,a1,cs)
            repl=[x for x in base if x!=rem]+[a1]; rb,ru=metrics(repl,win)
            vr[name]={'assembly1':list(map(int,combos[a1])),'top5':[list(map(int,combos[x])) for x in cands],
                      'add1_best':ab,'replace1_best':rb,'add1_union':au,'replace1_union':ru,
                      'removed_ticket':list(map(int,combos[rem]))}
        rows.append({'draw':rr,'winner':list(win),'baseline_best':bb,'baseline_union':bu,'cap':cap,'variants':vr})
        if rr%25==0: print('done',rr,flush=True)
    return rows

def summ(rows,name,mode):
    k='add1_best' if mode=='add1' else 'replace1_best'; vals=[r['variants'][name][k] for r in rows]; base=[r['baseline_best'] for r in rows]
    # assembly-specific KPI: baseline union=5 and baseline best<=2
    focus=[i for i,r in enumerate(rows) if r['baseline_union']==5 and r['baseline_best']<=2]
    return {'n':len(rows),'best3plus':sum(v>=3 for v in vals),'best4plus':sum(v>=4 for v in vals),'best5':sum(v>=5 for v in vals),
            'gain3':sum(v>=3 and b<3 for v,b in zip(vals,base)),'loss3':sum(v<3 and b>=3 for v,b in zip(vals,base)),
            'gain4':sum(v>=4 and b<4 for v,b in zip(vals,base)),'loss4':sum(v<4 and b>=4 for v,b in zip(vals,base)),
            'gain5':sum(v>=5 and b<5 for v,b in zip(vals,base)),'loss5':sum(v<5 and b>=5 for v,b in zip(vals,base)),
            'assembly_focus_n':len(focus),'assembly_focus_to3plus':sum(vals[i]>=3 for i in focus),'assembly_focus_to4plus':sum(vals[i]>=4 for i in focus)}

dev=eval_range(1100,1299); ds={n:{m:summ(dev,n,m) for m in ('add1','replace1')} for n in VARIANTS}
def key(n):
    s=ds[n]['replace1']; return (s['assembly_focus_to4plus'],s['assembly_focus_to3plus'],s['best5'],s['best4plus'],s['best3plus'],-s['loss3'])
sel=max(VARIANTS,key=key); print('SELECTED',sel,ds[sel]['replace1'],flush=True)
test=eval_range(1300,1399); ts={n:{m:summ(test,n,m) for m in ('add1','replace1')} for n in VARIANTS}
base={'n':len(test),'best3plus':sum(r['baseline_best']>=3 for r in test),'best4plus':sum(r['baseline_best']>=4 for r in test),'best5':sum(r['baseline_best']>=5 for r in test),
      'assembly_focus_n':sum(r['baseline_union']==5 and r['baseline_best']<=2 for r in test)}
focus={str(rr):next(r for r in test if r['draw']==rr) for rr in (1381,1385,1399)}
out={'definition':'Assembly-v2 pre-draw audit. High Final10 frequency is not assumed predictive. Development 1100-1299 selects one fixed role-balanced Hub/Mid/Satellite variant; test 1300-1399 untouched.',
     'principle':'Protect low-frequency satellite numbers; use normalized pair bridge support and novelty rather than simply selecting high-frequency numbers.',
     'variants':VARIANTS,'selected_variant':sel,'development':{'range':[1100,1299],'summary':ds},
     'test':{'range':[1300,1399],'baseline':base,'summary':ts,'selected':ts[sel]},'focus_draws':focus,'test_rows':test}
p=ROOT/'data'/'miniloto-assembly-v2-audit.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',p); print('TEST BASE',base); print('TEST SELECTED',ts[sel])
