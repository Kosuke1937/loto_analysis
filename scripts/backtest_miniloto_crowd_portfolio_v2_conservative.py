from pathlib import Path
import itertools, json, runpy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'miniloto-crowd-portfolio-v2-conservative.json'

# Reuse the fully audited v1 machinery. run_path executes v1 and exposes its canonical
# Committee/Crowd functions and arrays; v2 then tests a much more conservative policy.
m=runpy.run_path(str(ROOT/'scripts'/'backtest_miniloto_crowd_portfolio_v1.py'))
combos=m['combos']; draws=m['draws']; counts=m['counts']; combo_index=m['combo_index']
a1_score=m['a1_score']; a2_score=m['a2_score']; z=m['z']; top_indices=m['top_indices']; finalize=m['finalize']; crowd_predict_all=m['crowd_predict_all']


def topology(ids):
    union=set(); pairs=set()
    for idx in ids:
        a=tuple(map(int,combos[idx])); union.update(a); pairs.update(itertools.combinations(a,2))
    return len(union),len(pairs)


def eval_ids(ids,t,cp):
    win=set(map(int,draws[t])); tickets=[set(map(int,combos[i])) for i in ids]
    best=max(len(win&s) for s in tickets)
    vals=np.array([cp[i] for i in ids],float)
    return {'best':best,'crowd_mean':float(vals.mean()),'crowd_median':float(np.median(vals))}


def conservative_replace(ids,cs,cp,top1500,delta,min_reduce):
    ids=list(map(int,ids)); base_u,base_p=topology(ids); selected=set(ids)
    best_ids=ids; best_key=None; best_meta=None
    for pos,rem in enumerate(ids):
        rem_c=float(cp[rem]); min_score=float(cs[rem]-delta); max_c=rem_c*(1-min_reduce)
        cand=[int(i) for i in top1500 if int(i) not in selected and cs[int(i)]>=min_score and cp[int(i)]<=max_c]
        # Check the lowest-crowd alternatives first; cap scan for runtime and to avoid deep arbitrary searches.
        cand=sorted(cand,key=lambda i:(cp[i],-cs[i],i))[:250]
        for add in cand:
            trial=ids.copy(); trial[pos]=add
            u,p=topology(trial)
            if u<base_u or p<base_p: continue
            reduction=rem_c-float(cp[add])
            # Prefer crowd reduction, then Committee quality, then deterministic lower index.
            key=(reduction,float(cs[add]-cs[rem]),-add)
            if best_key is None or key>best_key:
                best_key=key; best_ids=trial; best_meta={'removed':rem,'added':add,'crowd_reduction':reduction,'score_change':float(cs[add]-cs[rem]),'union':u,'pairs':p}
    return best_ids,best_meta

PARAMS=[
    {'delta':d,'min_reduce':r}
    for d in (0.02,0.05,0.10,0.20)
    for r in (0.05,0.10,0.15)
]
keys=['baseline']+[f"d{p['delta']}_r{p['min_reduce']}" for p in PARAMS]
res={k:[] for k in keys}

for rr in range(1000,1400):
    t=rr-1; s1=a1_score(t); s2=a2_score(t); cs=z(s1)+0.15*z(s2)
    top5k=top_indices(cs,5000); top500=top5k[:500]
    base_ids,_=finalize(cs,top5k,top500); cp=crowd_predict_all(t)
    bm=eval_ids(base_ids,t,cp); bm.update({'draw':rr,'replaced':False}); res['baseline'].append(bm)
    for p in PARAMS:
        k=f"d{p['delta']}_r{p['min_reduce']}"
        ids,meta=conservative_replace(base_ids,cs,cp,top5k[:1500],p['delta'],p['min_reduce'])
        mm=eval_ids(ids,t,cp); mm.update({'draw':rr,'replaced':meta is not None,'meta':meta})
        if rr==1395:
            wi=combo_index[tuple(map(int,draws[t]))]; mm['winner_selected']=wi in ids
        res[k].append(mm)
    if rr%25==0: print('v2 done',rr,flush=True)


def summary(xs,a,b):
    q=[x for x in xs if a<=x['draw']<=b]
    return {'n':len(q),'best3plus':sum(x['best']>=3 for x in q),'best4plus':sum(x['best']>=4 for x in q),'exact5':sum(x['best']==5 for x in q),
            'crowd_mean':float(np.mean([x['crowd_mean'] for x in q])),'crowd_median_mean':float(np.mean([x['crowd_median'] for x in q])),
            'replacement_draws':sum(bool(x.get('replaced')) for x in q)}

val={k:summary(v,1000,1199) for k,v in res.items()}
base=val['baseline']; eligible=[]
for p in PARAMS:
    k=f"d{p['delta']}_r{p['min_reduce']}"; s=val[k]
    if s['exact5']>=base['exact5'] and s['best4plus']>=base['best4plus'] and s['best3plus']>=base['best3plus']:
        eligible.append((base['crowd_mean']-s['crowd_mean'],s['best3plus'],s['best4plus'],-p['delta'],p['min_reduce'],k))
chosen=max(eligible)[-1] if eligible else 'baseline'

test_base=summary(res['baseline'],1200,1399); test_chosen=summary(res[chosen],1200,1399)
bmap={x['draw']:x for x in res['baseline'] if 1200<=x['draw']<=1399}; cmap={x['draw']:x for x in res[chosen] if 1200<=x['draw']<=1399}
gain3=[d for d in bmap if bmap[d]['best']<3 and cmap[d]['best']>=3]; loss3=[d for d in bmap if bmap[d]['best']>=3 and cmap[d]['best']<3]
gain4=[d for d in bmap if bmap[d]['best']<4 and cmap[d]['best']>=4]; loss4=[d for d in bmap if bmap[d]['best']>=4 and cmap[d]['best']<4]
all_test={k:summary(v,1200,1399) for k,v in res.items()}

out={
 'model':'MiniLoto Crowd-aware Portfolio v2 conservative one-ticket replacement',
 'design':{
   'baseline':'canonical final10 unchanged',
   'candidate_pool':'Committee Top1500 only',
   'replacement':'at most one ticket; candidate must lower predicted crowd by minimum ratio, stay within delta Committee score of removed ticket, and preserve or improve both number-union size and pair-union size',
   'parameter_selection':'1000-1199 only; require exact5, 4+, 3+ all no worse than baseline, then maximize predicted crowd reduction',
   'fixed_test':'1200-1399 not used for parameter selection'
 },
 'validation':val,'chosen':chosen,
 'fixed_test':{'baseline':test_base,'chosen':test_chosen,'crowd_reduction_pct':100*(1-test_chosen['crowd_mean']/test_base['crowd_mean']),
               'gain3_draws':gain3,'loss3_draws':loss3,'gain4_draws':gain4,'loss4_draws':loss4},
 'test_all_params_diagnostic_not_for_selection':all_test,
 'draw1395':{k:next((x for x in v if x['draw']==1395),None) for k,v in res.items() if k in ('baseline',chosen)}
}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'chosen':chosen,'validation':val[chosen],'fixed_test':out['fixed_test'],'draw1395':out['draw1395']},ensure_ascii=False,indent=2))
