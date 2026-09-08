import json, math, collections, itertools
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
# Reuse only definitions from the established hybrid research script; do not execute its backtest.
src=(ROOT/'scripts'/'research_miniloto_committee_flow_hybrid.py').read_text(encoding='utf-8')
prefix=src.split('def evaluate(period,lams):')[0]
ns={}
exec(prefix,ns)

combos=ns['combos']; draws=ns['draws']; sums=ns['sums']; inc=ns['inc']; combo_shapes=ns['combo_shapes']
a1_score=ns['a1_score']; a2_score=ns['a2_score']; z=ns['z']; canonical_final=ns['canonical_final']
rolling_layer_map=ns['rolling_layer_map']; flow_score_all=ns['flow_score_all']
N=len(combos)

def committee_scores(t):
    s1=a1_score(t); s2=a2_score(t); return z(s1)+0.15*z(s2)

def metrics(ids, win):
    w=set(map(int,win)); best=max(len(w & set(map(int,combos[i]))) for i in ids)
    u=set(); [u.update(map(int,combos[i])) for i in ids]
    return best, len(w&u), len(u)

def choose_keepers(canon, cm, k):
    # Keep the highest Committee-scoring canonical tickets, but use marginal union coverage as a tie-break.
    remain=list(map(int,canon)); keep=[]; covered=set()
    while len(keep)<k and remain:
        best=max(remain, key=lambda i:(float(cm[i]), len(set(map(int,combos[i]))-covered), -i))
        keep.append(best); covered.update(map(int,combos[best])); remain.remove(best)
    return keep

def partial_hybrid(t, cm, canon, keep_n, lam, coverage_weight):
    layer_map=rolling_layer_map(t,500)
    fs=flow_score_all(t,layer_map); h=cm+lam*z(fs)
    full_union=set(); [full_union.update(map(int,combos[i])) for i in canon]
    keep=choose_keepers(canon,cm,keep_n)
    selected=list(keep)
    covered=set(); [covered.update(map(int,combos[i])) for i in selected]
    missing=set(full_union)-covered
    ulist=sorted(full_union)
    pool=np.where(inc[:,ulist].sum(1)==5)[0]
    pool=[int(i) for i in pool if int(i) not in set(selected)]

    # Initialize diversity counters from kept Committee tickets.
    tc=collections.Counter(); pc=collections.Counter(); nc=collections.Counter(); lc=collections.Counter()
    for i in selected:
        nums=tuple(map(int,combos[i])); L=layer_map[combo_shapes[i]]; lc[L]+=1
        for x in itertools.combinations(nums,3): tc[x]+=1
        for x in itertools.combinations(nums,2): pc[x]+=1
        for x in nums: nc[x]+=1

    target_n=10-keep_n
    for _ in range(target_n):
        best=None; bestkey=None
        for i in pool:
            if i in selected: continue
            nums=tuple(map(int,combos[i])); L=layer_map[combo_shapes[i]]
            # Soft ABCD control; D should remain rare. Avoid hard failure if keepers already contain D.
            layer_pen=0.0
            if L=='D': layer_pen -= 1.2 + 0.8*max(0,lc['D'])
            elif L=='C': layer_pen -= 0.15*max(0,lc['C']-1)
            elif L=='A': layer_pen += 0.15 if lc['A']<5 else 0.0
            elif L=='B': layer_pen += 0.10 if lc['B']<3 else 0.0
            # Existing canonical diversity caps as penalties, not absolute filters.
            trs=list(itertools.combinations(nums,3)); prs=list(itertools.combinations(nums,2))
            div_pen=0.0
            div_pen -= 1.5*sum(tc[x]>=1 for x in trs)
            div_pen -= 0.5*sum(pc[x]>=2 for x in prs)
            div_pen -= 0.25*sum(nc[x]>=4 for x in nums)
            cov=len(set(nums)&missing)
            key=float(h[i])+coverage_weight*cov+layer_pen+div_pen
            tie=(key,cov,float(cm[i]),-i)
            if bestkey is None or tie>bestkey:
                bestkey=tie; best=i
        if best is None: break
        selected.append(best)
        nums=tuple(map(int,combos[best])); covered.update(nums); missing=set(full_union)-covered
        L=layer_map[combo_shapes[best]]; lc[L]+=1
        for x in itertools.combinations(nums,3): tc[x]+=1
        for x in itertools.combinations(nums,2): pc[x]+=1
        for x in nums: nc[x]+=1
    return selected[:10], dict(lc), len(full_union), len(missing)

def evaluate(period, settings):
    base={'n':0,'d3':0,'d4':0,'d5':0,'union_recall_sum':0,'union5':0,'union_size_sum':0}
    outs={name:{'n':0,'d3':0,'d4':0,'d5':0,'union_recall_sum':0,'union5':0,'union_size_sum':0,'missing_union_sum':0,'sum_in_80_110':0,'layer_counts':collections.Counter()} for name in settings}
    for rr in period:
        t=rr-1; cm=committee_scores(t); canon,_=canonical_final(cm); win=draws[t]
        bm,ur,us=metrics(canon,win); base['n']+=1; base['d3']+=bm>=3; base['d4']+=bm>=4; base['d5']+=bm>=5; base['union_recall_sum']+=ur; base['union5']+=ur==5; base['union_size_sum']+=us
        for name,(keep_n,lam,cw) in settings.items():
            ids,lc,orig_us,missing=partial_hybrid(t,cm,canon,keep_n,lam,cw)
            bm,ur,us=metrics(ids,win); d=outs[name]; d['n']+=1; d['d3']+=bm>=3; d['d4']+=bm>=4; d['d5']+=bm>=5; d['union_recall_sum']+=ur; d['union5']+=ur==5; d['union_size_sum']+=us; d['missing_union_sum']+=missing
            d['sum_in_80_110']+=sum(80<=int(sums[i])<=110 for i in ids)
            lm=rolling_layer_map(t,500)
            for i in ids: d['layer_counts'][lm[combo_shapes[i]]]+=1
        if rr%50==0: print('done',rr,flush=True)
    for d in [base,*outs.values()]:
        n=d['n']; d['mean_union_recall']=d['union_recall_sum']/n; d['mean_union_size']=d['union_size_sum']/n
        if 'missing_union_sum' in d: d['mean_missing_original_union']=d['missing_union_sum']/n; d['mean_sum_in_80_110']=d['sum_in_80_110']/n; d['layer_counts']=dict(d['layer_counts'])
    return base,outs

# Tune lambda/coverage only on development. Primary ordering: preserve d4/d5 and union5, then d3.
DEV=range(1000,1200); TEST=range(1200,1400)
grid={}
for keep in (6,8):
    for lam in (0.1,0.2,0.3,0.4):
        for cw in (0.5,1.0,1.5,2.0):
            grid[f'k{keep}_l{lam}_c{cw}']=(keep,lam,cw)
base_dev,dev_all=evaluate(DEV,grid)
selected={}
for keep in (6,8):
    names=[n for n,v in grid.items() if v[0]==keep]
    best=max(names,key=lambda n:(dev_all[n]['d5'],dev_all[n]['d4'],dev_all[n]['union5'],dev_all[n]['d3'],dev_all[n]['union_recall_sum']))
    selected[f'{keep}+{10-keep}']={'name':best,'params':grid[best],'development':dev_all[best]}
settings={k:tuple(v['params']) for k,v in selected.items()}
base_test,test=evaluate(TEST,settings)

# Blind reconstruction for draw 1403 using history only through 1402 (t=1402 zero-based).
t=1402; cm=committee_scores(t); canon,cap=canonical_final(cm)
current={}
for label,(keep,lam,cw) in settings.items():
    ids,lc,orig_us,missing=partial_hybrid(t,cm,canon,keep,lam,cw)
    lm=rolling_layer_map(t,500)
    current[label]={'params':{'keep':keep,'lambda':lam,'coverage_weight':cw},'canonical_cap':cap,'original_union_size':orig_us,'missing_original_union':missing,'layer_counts':lc,'tickets':[
        {'position':j+1,'numbers':list(map(int,combos[i])),'sum':int(sums[i]),'shape':str(combo_shapes[i]),'layer':lm[combo_shapes[i]],'committee_rank':1+int(np.sum(cm>cm[i])),'kept_from_canonical':i in set(choose_keepers(canon,cm,keep))}
        for j,i in enumerate(ids)]}

out={'protocol':{'development':[1000,1199],'fixed_test':[1200,1399],'target_blind':1403,'note':'1403 candidates reconstructed using only information available through draw 1402; draw 1403 result not used in parameter selection.'},
     'development_canonical':base_dev,'selected_on_development':selected,'fixed_test_canonical':base_test,'fixed_test_partial':test,'current_1403_blind':current}
path=ROOT/'data'/'miniloto-committee-flow-partial-hybrid-backtest.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print('WROTE',path)
print(json.dumps({'selected':selected,'test_canonical':base_test,'test_partial':test},ensure_ascii=False,indent=2))
