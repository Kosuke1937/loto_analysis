import collections,itertools,json,runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; inc=m['inc']; draws=m['draws']; NCAND=m['N']; H=len(draws)
committee_score=m['committee_score'] if 'committee_score' in m else None
if committee_score is None:
    def committee_score(t):
        a1=m['a1_score'](t); a2=m['a2_score'](t); return a1,a2,m['z'](a1)+0.15*m['z'](a2)

def z(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s+1e-9)

def ranks(score):
    order=np.argsort(-score,kind='stable'); r=np.empty(len(score),int); r[order]=np.arange(1,len(score)+1); return r

def greedy(indices,K=10,triple_cap=1,pair_cap=2,num_cap=4):
    sel=[]; tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter()
    for idx in indices:
        q=tuple(map(int,combos[int(idx)])); trs=list(itertools.combinations(q,3)); prs=list(itertools.combinations(q,2))
        if any(tc[x]>=triple_cap for x in trs): continue
        if any(pc[x]>=pair_cap for x in prs): continue
        if any(nc[x]>=num_cap for x in q): continue
        sel.append(int(idx))
        for x in trs: tc[x]+=1
        for x in prs: pc[x]+=1
        for x in q: nc[x]+=1
        if len(sel)>=K: break
    return sel

t=H
stat,core,committee=committee_score(t)
stat_rank=ranks(stat); committee_rank=ranks(committee)
sums=combos.sum(1)
elig=np.where(sums<=110)[0]
score_init=z(committee)+0.20*z(stat)
ord_elig=elig[np.argsort(-score_init[elig],kind='stable')]
base=[]; usedcap=None
for cap in (2,3,4):
    x=greedy(ord_elig[:8000],10,1,2,cap)
    if len(x)>=10: base=x[:10]; usedcap=cap; break
if len(base)<10:
    base=list(map(int,ord_elig[:10])); usedcap=4
base_t=[tuple(map(int,combos[i])) for i in base]
union=sorted(set().union(*map(set,base_t)))
base_pairs=set(); base_triples=set(); parents=collections.defaultdict(set)
for j,q in enumerate(base_t):
    for n in q: parents[n].add(j)
    base_pairs.update(itertools.combinations(q,2)); base_triples.update(itertools.combinations(q,3))
idxmap={tuple(map(int,q)):i for i,q in enumerate(combos)}
rows=[]
for q in itertools.combinations(union,5):
    if sum(q)>110 or q in base_t: continue
    idx=idxmap[q]
    prs=list(itertools.combinations(q,2)); trs=list(itertools.combinations(q,3))
    newp=sum(p not in base_pairs for p in prs); newt=sum(tr not in base_triples for tr in trs)
    parent_union=len(set().union(*[parents[n] for n in q]))
    max_overlap=max(len(set(q)&set(b)) for b in base_t)
    rows.append((idx,q,newp,newt,parent_union,max_overlap))
idxs=np.array([r[0] for r in rows],int)
feat_newp=z([r[2] for r in rows]); feat_newt=z([r[3] for r in rows]); feat_parent=z([r[4] for r in rows]); feat_overlap=z([-r[5] for r in rows]); feat_comm=z(committee[idxs]); feat_stat=z(stat[idxs])
reb_score=0.25*feat_newp+0.30*feat_newt+0.15*feat_parent+0.15*feat_overlap+0.05*feat_comm+0.10*feat_stat
order=np.argsort(-reb_score,kind='stable')
rebuild=[]; pc=collections.Counter(); tc=collections.Counter(); nc=collections.Counter()
for oi in order:
    idx=int(idxs[oi]); q=tuple(map(int,combos[idx])); maxbase=max(len(set(q)&set(b)) for b in base_t)
    if maxbase>2: continue
    prs=list(itertools.combinations(q,2)); trs=list(itertools.combinations(q,3))
    if any(pc[p]>=1 for p in prs): continue
    if any(tc[tr]>=1 for tr in trs): continue
    if any(nc[n]>=3 for n in q): continue
    rebuild.append(idx)
    for p in prs: pc[p]+=1
    for tr in trs: tc[tr]+=1
    for n in q: nc[n]+=1
    if len(rebuild)>=10: break
if len(rebuild)<10:
    used=set(rebuild)
    for oi in order:
        idx=int(idxs[oi]); q=tuple(map(int,combos[idx]))
        if idx in used or max(len(set(q)&set(b)) for b in base_t)>3: continue
        prs=list(itertools.combinations(q,2))
        if any(pc[p]>=2 for p in prs): continue
        if any(nc[n]>=4 for n in q): continue
        rebuild.append(idx); used.add(idx)
        for p in prs: pc[p]+=1
        for n in q: nc[n]+=1
        if len(rebuild)>=10: break

def info(ids,is_rebuild=False):
    out=[]
    for k,idx in enumerate(ids,1):
        q=list(map(int,combos[idx])); out.append({'no':k,'numbers':q,'sum':int(sum(q)),'committee_rank':int(committee_rank[idx]),'stat_rank':int(stat_rank[idx]),'committee_score':float(committee[idx]),'stat_score':float(stat[idx]),'max_overlap_initial':int(max(len(set(q)&set(b)) for b in base_t)) if is_rebuild else 5})
    return out
res={'target_draw':H+1,'history_last_draw':H,'constraint':'sum<=110','initial_cap':usedcap,'initial10':info(base),'initial_union':union,'initial_union_size':len(union),'rebuilt10':info(rebuild,True),'rebuilt_union':sorted(set().union(*[set(map(int,combos[i])) for i in rebuild])),'rebuilt_union_size':len(set().union(*[set(map(int,combos[i])) for i in rebuild])),'method':'Initial = Committee primary + 0.20 Stat z blend with state cap diversification; Rebuild = same-number-union cross-cluster, new pair/triple novelty, low overlap to initial, plus weak Committee/Stat support; all sums <=110.'}
p=ROOT/'data'/'miniloto-1402-fast-rebuild.json'; p.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))