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
    x=np.asarray(x,float); return (x-x.mean())/(x.std()+1e-9)
def ranks(x):
    o=np.argsort(-x,kind='stable'); r=np.empty(len(x),int); r[o]=np.arange(1,len(x)+1); return r
# Force pre-draw-1402 state: use history only through draw 1401.
t=min(H,1401)
stat,core,committee=committee_score(t); sr=ranks(stat); cr=ranks(committee)
base=[(9,11,15,30,31),(1,18,19,30,31),(1,21,23,25,28),(5,20,21,23,29),(3,20,23,25,27),(7,9,26,27,29),(4,20,21,25,26),(6,8,27,28,29),(3,17,19,28,31),(7,14,19,26,30)]
union=sorted(set().union(*map(set,base))); idxmap={tuple(map(int,q)):i for i,q in enumerate(combos)}
base_idx=[idxmap[b] for b in base]
base_blend=z(committee)+0.40*z(stat)
rows=[]
for q in itertools.combinations(union,5):
    if q in base or sum(q)>105: continue
    ovs=[len(set(q)&set(b)) for b in base]
    if max(ovs)!=3: continue
    parents=[i for i,v in enumerate(ovs) if v==3]
    parent_quality=max(base_blend[base_idx[i]] for i in parents)
    parent_rank=min(i+1 for i in parents)
    rows.append((idxmap[q],q,parents,parent_quality,parent_rank))
idxs=np.array([r[0] for r in rows],int)
# 3-core repair: preserve a strong parent core, but let Stat/Committee choose the 2 replacements.
score=0.45*z(stat[idxs])+0.30*z(committee[idxs])+0.25*z([r[3] for r in rows])
order=np.argsort(-score,kind='stable')
selected=[]; pc=collections.Counter(); nc=collections.Counter(); parent_use=collections.Counter()
for oi in order:
    idx=int(idxs[oi]); q=tuple(map(int,combos[idx])); prs=list(itertools.combinations(q,2)); parents=rows[oi][2]
    # spread across parent tickets; avoid too much pair/number reuse
    if all(parent_use[p]>=2 for p in parents): continue
    if any(pc[p]>=2 for p in prs): continue
    if any(nc[n]>=4 for n in q): continue
    selected.append((oi,idx))
    for p in prs: pc[p]+=1
    for n in q: nc[n]+=1
    parent_use[min(parents,key=lambda p:parent_use[p])]+=1
    if len(selected)>=10: break
winner=(1,4,20,25,29); widx=idxmap[winner]
raw_rank=None
if widx in set(map(int,idxs)):
    raw_rank=int(np.where(idxs[order]==widx)[0][0])+1
out=[]
for k,(oi,idx) in enumerate(selected,1):
    q=tuple(map(int,combos[idx])); parents=rows[oi][2]
    out.append({'no':k,'numbers':list(q),'sum':sum(q),'stat_rank':int(sr[idx]),'committee_rank':int(cr[idx]),'parent_tickets':[p+1 for p in parents],'max_overlap_initial':3,'winner_match':len(set(q)&set(winner))})
res={'draw':1402,'method':'exactly 3 numbers shared with at least one initial ticket; replace 2 numbers from initial union; sum<=105; score=0.45 zStat + 0.30 zCommittee + 0.25 parent-quality; portfolio pair/number diversification','history_used_through':t,'initial10':[list(x) for x in base],'initial_union':union,'candidate_count':len(rows),'winner':list(winner),'winner_eligible':widx in set(map(int,idxs)),'winner_raw_rank':raw_rank,'winner_selected_position':next((x['no'] for x in out if x['numbers']==list(winner)),None),'rebuilt10':out}
p=ROOT/'data'/'miniloto-1402-3core-rebuild.json'; p.write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
