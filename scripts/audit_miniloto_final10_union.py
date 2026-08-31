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

def anti_support(score,top5000,mode):
    if mode=='cap4':
        base=greedy(top5000[:1500],10,1,2,4,True); used_cap=4
    elif mode=='state234':
        base=[]; used_cap=None
        for cap in (2,3,4):
            trial=greedy(top5000[:1500],10,1,2,cap,False)
            if len(trial)>=10:
                base=trial[:10]; used_cap=cap; break
        if len(base)<10:
            base=greedy(top5000[:1500],10,1,2,4,True); used_cap=4
    else: raise ValueError(mode)
    support=inc[top5000[:500]].sum(0).astype(float)
    union=set()
    for idx in base: union.update(map(int,combos[idx]))
    allowed=np.zeros(32,dtype=bool); allowed[list(union)]=True
    pool=top5000[allowed[combos[top5000]].all(axis=1)]
    bset=set(base); pool=np.array([int(i) for i in pool if int(i) not in bset],dtype=int)
    if len(pool)==0: return base,used_cap
    supp=support[combos[pool]].mean(1); zz=(supp-supp.mean())/(supp.std()+1e-9)
    add=int(pool[np.argmax(score[pool]-0.10*zz)])
    best=None; bestkey=None
    for rem in base:
        keep=[x for x in base if x!=rem]+[add]
        nu=set(); pu=set()
        for idx in keep:
            nums=tuple(map(int,combos[idx])); nu.update(nums); pu.update(itertools.combinations(nums,2))
        key=(len(nu),len(pu),float(np.sum(score[keep])))
        if bestkey is None or key>bestkey: bestkey=key; best=keep
    return best,used_cap

def detailed_recent(rr,ids,cap):
    t=rr-1
    tickets=[tuple(map(int,combos[i])) for i in ids]
    win=tuple(map(int,draws[t])); wset=set(win)
    union=set().union(*map(set,tickets))
    hist=collections.Counter(n for ticket in tickets for n in ticket)
    ticket_pairs=set(); ticket_triples=set()
    for ticket in tickets:
        ticket_pairs.update(itertools.combinations(ticket,2))
        ticket_triples.update(itertools.combinations(ticket,3))
    win_pairs=list(itertools.combinations(win,2)); win_triples=list(itertools.combinations(win,3))
    pair_hits=[p for p in win_pairs if p in ticket_pairs]
    triple_hits=[q for q in win_triples if q in ticket_triples]
    all_counts=[hist.get(n,0) for n in range(1,32)]
    winner_hist=[]
    for n in win:
        c=hist.get(n,0)
        rank=1+sum(v>c for v in all_counts)
        winner_hist.append({'number':n,'count':c,'rank_by_ticket_frequency':rank})
    return {
        'draw':rr,
        'winner':list(win),
        'winner_sum':sum(win),
        'tickets':[list(x) for x in tickets],
        'union_numbers':sorted(union),
        'union_size':len(union),
        'number_recall':len(wset&union),
        'pair_recall_count':len(pair_hits),
        'pair_recall_total':len(win_pairs),
        'pair_recall_rate':len(pair_hits)/len(win_pairs),
        'triple_recall_count':len(triple_hits),
        'triple_recall_total':len(win_triples),
        'triple_recall_rate':len(triple_hits)/len(win_triples),
        'pair_hits':[list(x) for x in pair_hits],
        'triple_hits':[list(x) for x in triple_hits],
        'histogram':[hist.get(n,0) for n in range(1,32)],
        'winner_histogram_positions':winner_hist,
        'best_ticket':max(len(wset&set(ticket)) for ticket in tickets),
        'cap':cap,
    }

def audit(mode,start=501,end=1399):
    rows=[]; counts={str(k):0 for k in range(6)}; cap_counts=collections.Counter(); recent=[]
    for rr in range(start,end+1):
        t=rr-1; _,_,cs=committee_score(t); top=top_indices(cs,5000)
        ids,cap=anti_support(cs,top,mode); cap_counts[str(cap)]+=1
        union=set()
        for i in ids: union.update(map(int,combos[i]))
        win=set(map(int,draws[t])); hit=len(win&union); counts[str(hit)]+=1
        best=max(len(win&set(map(int,combos[i]))) for i in ids)
        rows.append({'draw':rr,'union_hit':hit,'best_ticket':best,'union_size':len(union),'cap':cap})
        if mode=='state234' and rr>=1380:
            recent.append(detailed_recent(rr,ids,cap))
        if rr%50==0: print(mode,'done',rr,flush=True)
    return {'mode':mode,'range':[start,end],'n':len(rows),'counts':counts,'union5_draws':[r['draw'] for r in rows if r['union_hit']==5],
            'union4_draws':[r['draw'] for r in rows if r['union_hit']==4], 'cap_counts':dict(cap_counts),'rows':rows,
            'recent20_diagnostics':recent if mode=='state234' else []}

out={'definition':'Winner Number Recall of union of final 10 tickets; pre-draw only model; no draw 1400/1401 used',
     'recall_definitions':{
        'number_recall':'Of the 5 winning main numbers, how many appear at least once anywhere in the 10-ticket union.',
        'pair_recall':'Of the 10 true winning pairs C(5,2), how many appear together in at least one of the 10 tickets.',
        'triple_recall':'Of the 10 true winning triples C(5,3), how many appear together in at least one of the 10 tickets.'
     },
     'canonical_cap4':audit('cap4'), 'state234':audit('state234')}
for key in ('canonical_cap4','state234'):
    rows=out[key]['rows']
    for a,b in [(501,799),(800,1099),(1100,1399),(1101,1399),(1200,1399)]:
        z=[r for r in rows if a<=r['draw']<=b]
        c={str(k):sum(r['union_hit']==k for r in z) for k in range(6)}
        out[key].setdefault('blocks',{})[f'{a}-{b}']={'n':len(z),'counts':c,'union5':c['5'],'union4plus':c['4']+c['5']}

p=ROOT/'data'/'miniloto-final10-union-audit.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',p)
print('CAP4',out['canonical_cap4']['counts'])
print('STATE234',out['state234']['counts'])
