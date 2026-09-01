import collections, itertools, json, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']
committee_score=m.get('committee_score')
if committee_score is None:
    def committee_score(t):
        s1=m['a1_score'](t); s2=m['a2_score'](t)
        return s1,s2,m['z'](s1)+0.15*m['z'](s2)
combo_index={tuple(map(int,c)):i for i,c in enumerate(combos)}

def z(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s+1e-9)

def top_indices(score,k):
    idx=np.argpartition(-score,k-1)[:k]
    return idx[np.lexsort((idx,-score[idx]))]

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4,fallback=True):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        idx=int(idx); q=tuple(map(int,combos[idx])); trs=list(itertools.combinations(q,3)); prs=list(itertools.combinations(q,2))
        if any(tc[x]>=triple_cap for x in trs) or any(pc[x]>=pair_cap for x in prs) or any(nc[x]>=num_cap for x in q): continue
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
    union=set().union(*[set(map(int,combos[i])) for i in base]); allowed=np.zeros(32,bool); allowed[list(union)]=True
    pool=top5000[allowed[combos[top5000]].all(axis=1)]; pool=np.array([int(i) for i in pool if int(i) not in set(base)],int)
    if len(pool):
        zz=z(support[combos[pool]].mean(1)); add=int(pool[np.argmax(score[pool]-0.10*zz)])
        best=None; bk=None
        for rem in base:
            keep=[x for x in base if x!=rem]+[add]; nu=set(); pu=set()
            for idx in keep:
                q=tuple(map(int,combos[idx])); nu.update(q); pu.update(itertools.combinations(q,2))
            key=(len(nu),len(pu),float(np.sum(score[keep])))
            if bk is None or key>bk: bk=key; best=keep
        base=best
    return base,cap

def hist_pair(t,w=300):
    M=np.zeros((32,32),float)
    for row in draws[max(0,t-w):t]:
        for a,b in itertools.combinations(map(int,row),2): M[a,b]+=1; M[b,a]+=1
    return M

def pair500(top):
    M=np.zeros((32,32),float); ns=np.zeros(32,float)
    for idx in top[:500]:
        q=tuple(map(int,combos[int(idx)]))
        for n in q: ns[n]+=1
        for a,b in itertools.combinations(q,2): M[a,b]+=1; M[b,a]+=1
    R=np.zeros_like(M)
    for a in range(1,32):
        for b in range(a+1,32):
            R[a,b]=R[b,a]=M[a,b]/np.sqrt(max(ns[a],1)*max(ns[b],1))
    return R

def assembly_s3(t,score,top,base):
    # S3_broad frozen from Assembly-v2 development selection.
    tickets=[tuple(map(int,combos[i])) for i in base]; union=sorted(set().union(*map(set,tickets)))
    nf=collections.Counter(n for q in tickets for n in q); R=pair500(top); H=hist_pair(t,300); bset=set(base)
    idxs=[]; role=[]; bridge=[]; hist=[]; novel=[]; comm=[]; target=[1,2,2,3,4]
    for q in itertools.combinations(union,5):
        idx=combo_index[q]
        if idx in bset: continue
        idxs.append(idx); cs=sorted(nf[n] for n in q); role.append(-sum(abs(a-b) for a,b in zip(cs,target)))
        ps=list(itertools.combinations(q,2)); bridge.append(sum(R[a,b] for a,b in ps)); hist.append(sum(H[a,b] for a,b in ps))
        novel.append(-max(len(set(q)&set(t)) for t in tickets)); comm.append(score[idx])
    idxs=np.asarray(idxs,int); s=.35*z(role)+.25*z(bridge)+.10*z(hist)+.20*z(novel)+.10*z(comm)
    order=np.argsort(-s,kind='stable')
    return int(idxs[order[0]])

def coverage(ids):
    nu=set(); pu=set(); tu=set()
    for idx in ids:
        q=tuple(map(int,combos[idx])); nu.update(q); pu.update(itertools.combinations(q,2)); tu.update(itertools.combinations(q,3))
    return nu,pu,tu

def choose_removal(base,a1,score):
    bnu,bpu,btu=coverage(base); best=None; bk=None; bestfeat=None
    base_scores=np.array([score[i] for i in base],float)
    for pos,rem in enumerate(base,1):
        other=[x for x in base if x!=rem]; onu,opu,otu=coverage(other)
        keep=other+[a1]; nu,pu,tu=coverage(keep)
        q=set(map(int,combos[rem])); unique_num=len(bnu-onu); unique_pair=len(bpu-opu); unique_triple=len(btu-otu)
        feat={
            'removed_pos':pos,'removed_ticket':list(map(int,combos[rem])),
            'number_loss':len(bnu)-len(nu),'pair_loss':len(bpu)-len(pu),'triple_loss':len(btu)-len(tu),
            'unique_num_before':unique_num,'unique_pair_before':unique_pair,'unique_triple_before':unique_triple,
            'removed_score_rank':int(np.sum(base_scores>score[rem])+1),
            'assembly_max_overlap':max(len(set(map(int,combos[a1]))&set(map(int,combos[x]))) for x in base),
        }
        # Safety-first removal: preserve number, then pair, then triple coverage; then remove later/lower score.
        key=(-feat['number_loss'],-feat['pair_loss'],-feat['triple_loss'],feat['removed_score_rank'],pos)
        if bk is None or key>bk: bk=key; best=rem; bestfeat=feat
    return best,bestfeat

GATES={
 'G0_always': lambda f: True,
 'G1_full_number': lambda f: f['number_loss']==0,
 'G2_pair3': lambda f: f['number_loss']==0 and f['pair_loss']<=3,
 'G3_pair2_triple6': lambda f: f['number_loss']==0 and f['pair_loss']<=2 and f['triple_loss']<=6,
 'G4_late_pair3': lambda f: f['number_loss']==0 and f['pair_loss']<=3 and f['removed_pos']>=6,
 'G5_late_pair2_triple6': lambda f: f['number_loss']==0 and f['pair_loss']<=2 and f['triple_loss']<=6 and f['removed_pos']>=6,
 'G6_novel_safe': lambda f: f['number_loss']==0 and f['pair_loss']<=3 and f['assembly_max_overlap']<=3,
 'G7_strict': lambda f: f['number_loss']==0 and f['pair_loss']<=1 and f['triple_loss']<=4 and f['removed_pos']>=6,
}

def metrics(ids,win):
    ws=set(win); ts=[set(map(int,combos[i])) for i in ids]; return max(len(ws&q) for q in ts),len(ws&set().union(*ts))

def eval_range(a,b):
    rows=[]
    for rr in range(a,b+1):
        t=rr-1; _,_,cs=committee_score(t); top=top_indices(cs,5000); base,cap=final10(cs,top); win=tuple(map(int,draws[t]))
        bb,bu=metrics(base,win); a1=assembly_s3(t,cs,top,base); rem,feat=choose_removal(base,a1,cs); repl=[x for x in base if x!=rem]+[a1]
        rb,ru=metrics(repl,win); gates={}
        for name,fn in GATES.items():
            on=bool(fn(feat)); ids=repl if on else base; vb,vu=metrics(ids,win)
            gates[name]={'activated':on,'best':vb,'union':vu}
        rows.append({'draw':rr,'winner':list(win),'baseline_best':bb,'baseline_union':bu,'cap':cap,
                     'assembly1':list(map(int,combos[a1])),'replacement_best':rb,'replacement_union':ru,
                     'removal':feat,'gates':gates})
        if rr%25==0: print('done',rr,flush=True)
    return rows

def summ(rows,g):
    vals=[r['gates'][g]['best'] for r in rows]; base=[r['baseline_best'] for r in rows]
    focus=[i for i,r in enumerate(rows) if r['baseline_union']==5 and r['baseline_best']<=2]
    return {'n':len(rows),'activated':sum(r['gates'][g]['activated'] for r in rows),
      'best3plus':sum(v>=3 for v in vals),'best4plus':sum(v>=4 for v in vals),'best5':sum(v>=5 for v in vals),
      'gain3':sum(v>=3 and b<3 for v,b in zip(vals,base)),'loss3':sum(v<3 and b>=3 for v,b in zip(vals,base)),
      'gain4':sum(v>=4 and b<4 for v,b in zip(vals,base)),'loss4':sum(v<4 and b>=4 for v,b in zip(vals,base)),
      'gain5':sum(v>=5 and b<5 for v,b in zip(vals,base)),'loss5':sum(v<5 and b>=5 for v,b in zip(vals,base)),
      'assembly_focus_n':len(focus),'assembly_focus_to3plus':sum(vals[i]>=3 for i in focus),'assembly_focus_to4plus':sum(vals[i]>=4 for i in focus)}

def baseline_s(rows):
    return {'n':len(rows),'best3plus':sum(r['baseline_best']>=3 for r in rows),'best4plus':sum(r['baseline_best']>=4 for r in rows),'best5':sum(r['baseline_best']>=5 for r in rows),
      'assembly_focus_n':sum(r['baseline_union']==5 and r['baseline_best']<=2 for r in rows)}

dev=eval_range(1100,1299); ds={g:summ(dev,g) for g in GATES}; db=baseline_s(dev)
# Selection is development-only and conservative: preserve 4+/3+ losses first, then gain/focus conversion, then activation.
def selkey(g):
    s=ds[g]
    return (-s['loss4'],-s['loss3'],s['gain4'],s['gain3'],s['assembly_focus_to3plus'],s['best4plus'],s['best3plus'],-s['activated'])
sel=max(GATES,key=selkey); print('SELECTED',sel,ds[sel],flush=True)
test=eval_range(1300,1399); ts={g:summ(test,g) for g in GATES}; tb=baseline_s(test)
focus={str(rr):next(r for r in test if r['draw']==rr) for rr in (1381,1385,1395,1399)}
out={'definition':'Assembly-v3 conditional replacement gate. Assembly candidate S3_broad is frozen from v2. Gate is selected only on 1100-1299; 1300-1399 untouched.',
 'principle':'Do not assume frequently proposed numbers are more likely. Replace only when pre-draw coverage-loss and redundancy features indicate acceptable safety.',
 'gates':list(GATES),'development':{'range':[1100,1299],'baseline':db,'summary':ds},'selected_gate':sel,
 'test':{'range':[1300,1399],'baseline':tb,'summary':ts,'selected':ts[sel]},'focus_draws':focus,'test_rows':test}
p=ROOT/'data'/'miniloto-assembly-v3-gate-audit.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',p); print('DEV BASE',db); print('TEST BASE',tb); print('TEST SELECTED',ts[sel])
