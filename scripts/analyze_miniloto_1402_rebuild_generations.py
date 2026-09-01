import collections,itertools,json,runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
m=runpy.run_path(str(ROOT/'scripts'/'generate_miniloto_model_ranks.py'))
combos=m['combos']; H=len(m['draws'])
committee_score=m['committee_score'] if 'committee_score' in m else None
if committee_score is None:
    def committee_score(t):
        a1=m['a1_score'](t); a2=m['a2_score'](t); return a1,a2,m['z'](a1)+0.15*m['z'](a2)

def z(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/(s+1e-9)

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

stat,core,committee=committee_score(H)
sums=combos.sum(1)
# Reproduce the original initial 10 exactly from the pre-draw generator.
elig=np.where((sums>=85)&(sums<=105))[0]
score_init=z(committee)+0.40*z(stat)
ord_elig=elig[np.argsort(-score_init[elig],kind='stable')]
base=[]
for cap in (2,3,4):
    x=greedy(ord_elig[:10000],10,1,2,cap)
    if len(x)>=10:
        base=x[:10]; break
if len(base)<10: base=list(map(int,ord_elig[:10]))
base_t=[tuple(map(int,combos[i])) for i in base]
union=sorted(set().union(*map(set,base_t)))
base_pairs=set(); base_triples=set(); parents=collections.defaultdict(set)
for j,q in enumerate(base_t):
    for n in q: parents[n].add(j)
    base_pairs.update(itertools.combinations(q,2)); base_triples.update(itertools.combinations(q,3))
idxmap={tuple(map(int,q)):i for i,q in enumerate(combos)}
winner=(1,4,20,25,29); winner_idx=idxmap[winner]

# Candidate feature table relative to the original 10. This keeps the parent information fixed,
# so 'generation' means taking another non-overlapping batch from the same reconstructed candidate space.
allrows=[]
for q in itertools.combinations(union,5):
    idx=idxmap[q]
    prs=list(itertools.combinations(q,2)); trs=list(itertools.combinations(q,3))
    newp=sum(p not in base_pairs for p in prs); newt=sum(tr not in base_triples for tr in trs)
    parent_union=len(set().union(*[parents[n] for n in q]))
    max_overlap=max(len(set(q)&set(b)) for b in base_t)
    allrows.append((idx,q,newp,newt,parent_union,max_overlap))

def evaluate(name,min_sum,max_sum,max_overlap,stat_w=0.25,novelty_scale=1.0,comm_w=0.10,max_rounds=200):
    rows=[r for r in allrows if min_sum<=sum(r[1])<=max_sum and r[1] not in base_t and r[5]<=max_overlap]
    idxs=np.array([r[0] for r in rows],int)
    out={'name':name,'min_sum':min_sum,'max_sum':max_sum,'max_overlap':max_overlap,'stat_weight':stat_w,'novelty_scale':novelty_scale,'candidate_count':len(rows),'winner_eligible':False,'winner_raw_rank':None,'winner_generation':None,'winner_position_in_generation':None}
    if not rows or winner_idx not in set(map(int,idxs)): return out
    out['winner_eligible']=True
    fnp=z([r[2] for r in rows]); fnt=z([r[3] for r in rows]); fpa=z([r[4] for r in rows]); fov=z([-r[5] for r in rows]); fco=z(committee[idxs]); fst=z(stat[idxs])
    score=novelty_scale*(0.18*fnp+0.22*fnt)+0.12*fpa+0.13*fov+comm_w*fco+stat_w*fst
    order=np.argsort(-score,kind='stable')
    ranked=[int(idxs[i]) for i in order]
    out['winner_raw_rank']=ranked.index(winner_idx)+1
    used=set()
    for gen in range(1,max_rounds+1):
        pc=collections.Counter(); tc=collections.Counter(); nc=collections.Counter(); sel=[]
        for idx in ranked:
            if idx in used: continue
            q=tuple(map(int,combos[idx])); prs=list(itertools.combinations(q,2)); trs=list(itertools.combinations(q,3))
            if any(pc[p]>=1 for p in prs): continue
            if any(tc[t]>=1 for t in trs): continue
            if any(nc[n]>=3 for n in q): continue
            sel.append(idx)
            for p in prs: pc[p]+=1
            for t in trs: tc[t]+=1
            for n in q: nc[n]+=1
            if len(sel)>=10: break
        if len(sel)<10:
            # relaxed fallback mirrors the original generator.
            for idx in ranked:
                if idx in used or idx in sel: continue
                q=tuple(map(int,combos[idx])); prs=list(itertools.combinations(q,2))
                if any(pc[p]>=2 for p in prs): continue
                if any(nc[n]>=4 for n in q): continue
                sel.append(idx)
                for p in prs: pc[p]+=1
                for n in q: nc[n]+=1
                if len(sel)>=10: break
        if winner_idx in sel:
            out['winner_generation']=gen
            out['winner_position_in_generation']=sel.index(winner_idx)+1
            break
        used.update(sel)
        if not sel: break
    return out

variants=[]
variants.append(evaluate('V0_current_85_105_overlap2',85,105,2,0.25,1.0))
variants.append(evaluate('V1_no_floor_overlap2',0,105,2,0.25,1.0))
variants.append(evaluate('V2_no_floor_overlap3_current_weights',0,105,3,0.25,1.0))
variants.append(evaluate('V3_no_floor_overlap5_current_weights',0,105,5,0.25,1.0))
# Small oracle sensitivity grid. This is explicitly post-hoc and not a forward-valid model selection.
for ov in (3,5):
    for sw in (0.15,0.25,0.35,0.45):
        for ns in (0.0,0.5,1.0):
            variants.append(evaluate(f'GRID_ov{ov}_stat{sw:.2f}_nov{ns:.1f}',0,105,ov,sw,ns))
valid=[v for v in variants if v['winner_generation'] is not None]
best=min(valid,key=lambda v:(v['winner_generation'],v['winner_position_in_generation'],v['winner_raw_rank'])) if valid else None
res={
 'draw':1402,
 'winner':list(winner),
 'winner_sum':sum(winner),
 'initial10':[list(q) for q in base_t],
 'initial_union':union,
 'winner_in_initial_union':all(n in union for n in winner),
 'winner_max_overlap_initial':max(len(set(winner)&set(b)) for b in base_t),
 'blocking_conditions':{'sum_floor_85':sum(winner)<85,'max_overlap_le_2':max(len(set(winner)&set(b)) for b in base_t)>2},
 'generation_definition':'Each generation selects the next 10 reconstructed combinations from the same original-union candidate space, excluding combinations already selected in earlier generations. Portfolio pair/triple/number caps reset each generation.',
 'variants':variants,
 'oracle_best':best,
 'warning':'All tuning in this audit uses the known draw-1402 winner. Treat as mechanism diagnosis only; do not report as forward predictive evidence.'
}
p=ROOT/'data'/'miniloto-1402-rebuild-generations.json'; p.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
