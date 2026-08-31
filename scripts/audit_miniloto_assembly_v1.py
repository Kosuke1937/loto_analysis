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
        return s1, s2, m['z'](s1)+0.15*m['z'](s2)

combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}

def z(x):
    x=np.asarray(x,dtype=float); s=x.std()
    return (x-x.mean())/(s+1e-9)

def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        idx=int(idx); nums=tuple(map(int,combos[idx]))
        trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
        if any(tc[x]>=triple_cap for x in trs): continue
        if any(pc[x]>=pair_cap for x in prs): continue
        if any(nc[x]>=num_cap for x in nums): continue
        sel.append(idx)
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in nums: nc[x]+=1
        if len(sel)>=K: break
    if fallback and len(sel)<K:
        used=set(sel)
        for idx in indices:
            idx=int(idx)
            if idx not in used:
                sel.append(idx); used.add(idx)
                if len(sel)>=K: break
    return sel

def state234_final10(score,top5000):
    base=[]; used_cap=None
    for cap in (2,3,4):
        trial=greedy(top5000[:1500],10,1,2,cap,False)
        if len(trial)>=10:
            base=trial[:10]; used_cap=cap; break
    if len(base)<10:
        base=greedy(top5000[:1500],10,1,2,4,True); used_cap=4
    support=inc[top5000[:500]].sum(0).astype(float)
    union=set()
    for idx in base: union.update(map(int,combos[idx]))
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=top5000[allowed[combos[top5000]].all(axis=1)]
    bset=set(base); pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
    if len(pool):
        supp=support[combos[pool]].mean(1); zz=z(supp)
        add=int(pool[np.argmax(score[pool]-0.10*zz)])
        best=None; bestkey=None
        for rem in base:
            keep=[x for x in base if x!=rem]+[add]
            nu=set(); pu=set()
            for idx in keep:
                nums=tuple(map(int,combos[idx])); nu.update(nums); pu.update(itertools.combinations(nums,2))
            key=(len(nu),len(pu),float(np.sum(score[keep])))
            if bestkey is None or key>bestkey: bestkey=key; best=keep
        base=best
    return base,used_cap

def pair_matrix_from_indices(indices):
    M=np.zeros((32,32),dtype=float)
    for idx in indices:
        nums=tuple(map(int,combos[int(idx)]))
        for a,b in itertools.combinations(nums,2):
            M[a,b]+=1; M[b,a]+=1
    return M

def hist_pair_matrix(t,window=300):
    M=np.zeros((32,32),dtype=float)
    a=max(0,t-window)
    for row in draws[a:t]:
        nums=tuple(map(int,row))
        for x,y in itertools.combinations(nums,2):
            M[x,y]+=1; M[y,x]+=1
    return M

VARIANTS={
    'V1_freq_pair500': (0.70,0.30,0.00,0.00),
    'V2_freq_histpair':(0.60,0.00,0.40,0.00),
    'V3_balanced':     (0.40,0.30,0.20,0.10),
    'V4_structure':    (0.20,0.40,0.30,0.10),
    'V5_committee':    (0.30,0.20,0.10,0.40),
}

def assembly_candidates(t,score,top5000,base):
    tickets=[tuple(map(int,combos[i])) for i in base]
    union=sorted(set().union(*map(set,tickets)))
    nf=collections.Counter(n for q in tickets for n in q)
    p500=pair_matrix_from_indices(top5000[:500])
    ph=hist_pair_matrix(t,300)
    base_set=set(base)
    rows=[]; idxs=[]; f_num=[]; f_p500=[]; f_hist=[]; f_comm=[]
    for q in itertools.combinations(union,5):
        idx=combo_index[q]
        if idx in base_set: continue
        idxs.append(idx); rows.append(q)
        f_num.append(np.mean([nf[n] for n in q]))
        ps=list(itertools.combinations(q,2))
        f_p500.append(sum(p500[a,b] for a,b in ps))
        f_hist.append(sum(ph[a,b] for a,b in ps))
        f_comm.append(score[idx])
    idxs=np.asarray(idxs,dtype=int)
    feats=[z(f_num),z(f_p500),z(f_hist),z(f_comm)]
    out={}
    for name,w in VARIANTS.items():
        s=sum(wi*fi for wi,fi in zip(w,feats))
        order=np.argsort(-s,kind='stable')
        out[name]=[int(idxs[i]) for i in order[:5]]
    return out

def redundancy_removal(base,score):
    # Remove the ticket whose deletion preserves the largest number/pair coverage;
    # break ties by removing the lower-Committee ticket.
    best=None; bestkey=None
    for rem in base:
        keep=[x for x in base if x!=rem]
        nu=set(); pu=set()
        for idx in keep:
            q=tuple(map(int,combos[idx])); nu.update(q); pu.update(itertools.combinations(q,2))
        key=(len(nu),len(pu),-float(score[rem]))
        if bestkey is None or key>bestkey:
            bestkey=key; best=rem
    return best

def metrics(ids,win):
    ws=set(win); tickets=[set(map(int,combos[i])) for i in ids]
    best=max(len(ws&q) for q in tickets)
    union=set().union(*tickets)
    return {'best':best,'union':len(ws&union)}

def evaluate_range(start,end):
    rows=[]
    for rr in range(start,end+1):
        t=rr-1; _,_,cs=committee_score(t); top=top_indices(cs,5000)
        base,cap=state234_final10(cs,top)
        win=tuple(map(int,draws[t])); bm=metrics(base,win)
        variants=assembly_candidates(t,cs,top,base)
        rem=redundancy_removal(base,cs)
        vr={}
        for name,cands in variants.items():
            a1=cands[0]
            add=metrics(base+[a1],win)
            repl=[x for x in base if x!=rem]+[a1]
            rep=metrics(repl,win)
            vr[name]={
                'assembly1':list(map(int,combos[a1])),
                'assembly_top5':[list(map(int,combos[x])) for x in cands],
                'add1_best':add['best'],'replace1_best':rep['best'],
                'add1_union':add['union'],'replace1_union':rep['union']
            }
        rows.append({'draw':rr,'winner':list(win),'baseline_best':bm['best'],'baseline_union':bm['union'],
                     'cap':cap,'removed_ticket':list(map(int,combos[rem])),'variants':vr})
        if rr%25==0: print('done',rr,flush=True)
    return rows

def summary(rows,name,mode):
    key='add1_best' if mode=='add1' else 'replace1_best'
    vals=[r['variants'][name][key] for r in rows]
    base=[r['baseline_best'] for r in rows]
    return {
        'n':len(rows),
        'best3plus':sum(x>=3 for x in vals),'best4plus':sum(x>=4 for x in vals),'best5':sum(x>=5 for x in vals),
        'gain3':sum(v>=3 and b<3 for v,b in zip(vals,base)), 'loss3':sum(v<3 and b>=3 for v,b in zip(vals,base)),
        'gain4':sum(v>=4 and b<4 for v,b in zip(vals,base)), 'loss4':sum(v<4 and b>=4 for v,b in zip(vals,base)),
        'gain5':sum(v>=5 and b<5 for v,b in zip(vals,base)), 'loss5':sum(v<5 and b>=5 for v,b in zip(vals,base)),
    }

dev=evaluate_range(1100,1299)
dev_s={name:{mode:summary(dev,name,mode) for mode in ('add1','replace1')} for name in VARIANTS}
# Select only on replacement performance, lexicographically 5+,4+,3+, then lower loss3.
def selkey(name):
    s=dev_s[name]['replace1']; return (s['best5'],s['best4plus'],s['best3plus'],-s['loss3'])
selected=max(VARIANTS,key=selkey)
print('SELECTED',selected,dev_s[selected]['replace1'],flush=True)

test=evaluate_range(1300,1399)
test_s={name:{mode:summary(test,name,mode) for mode in ('add1','replace1')} for name in VARIANTS}
base_test={'n':len(test),'best3plus':sum(r['baseline_best']>=3 for r in test),
           'best4plus':sum(r['baseline_best']>=4 for r in test),'best5':sum(r['baseline_best']>=5 for r in test)}
focus={str(rr):next(r for r in test if r['draw']==rr) for rr in (1381,1385,1399)}
out={
 'definition':'Assembly-v1 pre-draw audit. Development 1100-1299 selects one fixed variant; test 1300-1399 is untouched by selection.',
 'baseline':'Current Committee -> state-dependent cap 2->3->4 -> anti-support Final10.',
 'assembly':'Recombine only numbers already present in baseline Final10 union. Features use Final10 number frequency, Committee Top500 pair support, prior-300 actual-draw pair support, and current Committee score; no current winning numbers.',
 'replacement':'Keep 10-ticket budget. Remove one baseline ticket chosen pre-draw to preserve maximal number union then pair union, with lower Committee score as tie-break; insert Assembly1.',
 'variants':{k:list(v) for k,v in VARIANTS.items()},
 'development':{'range':[1100,1299],'baseline':{'best3plus':sum(r['baseline_best']>=3 for r in dev),'best4plus':sum(r['baseline_best']>=4 for r in dev),'best5':sum(r['baseline_best']>=5 for r in dev)},'summary':dev_s},
 'selected_variant':selected,
 'test':{'range':[1300,1399],'baseline':base_test,'summary':test_s,'selected':test_s[selected]},
 'focus_draws':focus,
 'test_rows':test
}
p=ROOT/'data'/'miniloto-assembly-v1-audit.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',p)
print('TEST BASE',base_test)
print('TEST SELECTED',test_s[selected])
