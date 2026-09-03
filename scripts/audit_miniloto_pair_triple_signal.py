import itertools,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8');m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
    if m: rows.extend(json.loads(m.group(1)))
rows=sorted(rows,key=lambda r:int(r[0]))
draws=[tuple(int(x) for x in r[2:7]) for r in rows]
PAIR=list(itertools.combinations(range(1,32),2));TRI=list(itertools.combinations(range(1,32),3))
pidx={p:i for i,p in enumerate(PAIR)};tidx={q:i for i,q in enumerate(TRI)}
P=len(PAIR);Q=len(TRI);T=len(draws)
# prefix counts: 465 and 4495 only, so this is cheap.
pp=np.zeros((T+1,P),np.int16);tp=np.zeros((T+1,Q),np.int16);npref=np.zeros((T+1,32),np.int16)
for i,d in enumerate(draws):
    pp[i+1]=pp[i];tp[i+1]=tp[i];npref[i+1]=npref[i]
    for n in d: npref[i+1,n]+=1
    for p in itertools.combinations(d,2): pp[i+1,pidx[tuple(sorted(p))]]+=1
    for q in itertools.combinations(d,3): tp[i+1,tidx[tuple(sorted(q))]]+=1

def z(x):
    x=np.asarray(x,float);s=x.std();return (x-x.mean())/(s+1e-12)
def rank_order(s): return np.argsort(-np.asarray(s),kind='stable')
def score_pair(t,w,kind='raw'):
    lo=max(0,t-w);c=(pp[t]-pp[lo]).astype(float)
    if kind=='raw': return c
    n=(npref[t]-npref[lo]).astype(float)
    # normalize pair support by geometric mean of marginal number counts
    den=np.array([np.sqrt(max(n[a],1)*max(n[b],1)) for a,b in PAIR])
    return c/den
def score_tri(t,w,kind='raw'):
    lo=max(0,t-w);c=(tp[t]-tp[lo]).astype(float)
    if kind=='raw': return c
    n=(npref[t]-npref[lo]).astype(float)
    den=np.array([(max(n[a],1)*max(n[b],1)*max(n[c0],1))**(1/3) for a,b,c0 in TRI])
    return c/den
PAIR_METHODS=[];TRI_METHODS=[]
for w in (50,100,300,500):
    PAIR_METHODS += [(f'pair_raw{w}',lambda t,w=w:score_pair(t,w,'raw')),(f'pair_resid{w}',lambda t,w=w:score_pair(t,w,'resid'))]
    TRI_METHODS += [(f'tri_raw{w}',lambda t,w=w:score_tri(t,w,'raw')),(f'tri_resid{w}',lambda t,w=w:score_tri(t,w,'resid'))]
# fixed short/long contrasts, still pre-draw only
PAIR_METHODS += [('pair_shortlong',lambda t:z(score_pair(t,100,'resid'))-0.35*z(score_pair(t,500,'resid')))]
TRI_METHODS += [('tri_shortlong',lambda t:z(score_tri(t,100,'resid'))-0.35*z(score_tri(t,500,'resid')))]
PK=(25,50,100,200);TK=(100,300,500,1000)

def eval_range(a,b):
    acc={'pair':{},'triple':{}};detail=[]
    for name,_ in PAIR_METHODS:
        acc['pair'][name]={k:[] for k in PK}
    for name,_ in TRI_METHODS:
        acc['triple'][name]={k:[] for k in TK}
    for rr in range(a,b+1):
        t=rr-1;win=draws[t];wp={pidx[tuple(sorted(p))] for p in itertools.combinations(win,2)};wt={tidx[tuple(sorted(q))] for q in itertools.combinations(win,3)}
        drow={'draw':rr,'pair':{},'triple':{}}
        for name,fn in PAIR_METHODS:
            order=rank_order(fn(t));vals={}
            for k in PK:
                hit=len(wp & set(map(int,order[:k])));acc['pair'][name][k].append(hit);vals[str(k)]=hit
            drow['pair'][name]=vals
        for name,fn in TRI_METHODS:
            order=rank_order(fn(t));vals={}
            for k in TK:
                hit=len(wt & set(map(int,order[:k])));acc['triple'][name][k].append(hit);vals[str(k)]=hit
            drow['triple'][name]=vals
        detail.append(drow)
        if rr%50==0: print('done',rr,flush=True)
    def summarize(group,U,Ks):
        out={}
        for name,byk in acc[group].items():
            out[name]={}
            for k in Ks:
                x=np.array(byk[k],float);rand=10*k/U
                out[name][str(k)]={'mean_hits':float(x.mean()),'random_mean':float(rand),'enrichment':float(x.mean()/rand),'draws_ge1':int((x>=1).sum()),'draws_ge2':int((x>=2).sum()),'draws_ge3':int((x>=3).sum()),'max_hits':int(x.max())}
        return out
    return {'range':[a,b],'pair':summarize('pair',P,PK),'triple':summarize('triple',Q,TK),'rows':detail}

dev=eval_range(1000,1199);test=eval_range(1200,1399)
# Select one pair and one triple operating point on dev only, favor enrichment but require >=1 random expected hit to avoid tiny-K instability.
def choose(summary,kind):
    cand=[]
    for m,byk in summary[kind].items():
        for k,v in byk.items():
            if v['random_mean']>=1.0:
                cand.append((v['enrichment'],v['mean_hits'],m,k))
    return max(cand) if cand else None
psel=choose(dev,'pair');tsel=choose(dev,'triple')
sel={}
if psel:
    _,_,m,k=psel;sel['pair']={'method':m,'k':int(k),'development':dev['pair'][m][k],'test':test['pair'][m][k]}
if tsel:
    _,_,m,k=tsel;sel['triple']={'method':m,'k':int(k),'development':dev['triple'][m][k],'test':test['triple'][m][k]}
# draw 1402 diagnostic, never used for selection
rr=1402;diag=None
if rr<=T:
    t=rr-1;win=draws[t];wp={pidx[tuple(sorted(p))] for p in itertools.combinations(win,2)};wt={tidx[tuple(sorted(q))] for q in itertools.combinations(win,3)};diag={'draw':rr,'winner':list(win)}
    if 'pair' in sel:
        m=sel['pair']['method'];k=sel['pair']['k'];fn=dict(PAIR_METHODS)[m];order=rank_order(fn(t));diag['pair_selected_hits']=len(wp&set(map(int,order[:k])))
    if 'triple' in sel:
        m=sel['triple']['method'];k=sel['triple']['k'];fn=dict(TRI_METHODS)[m];order=rank_order(fn(t));diag['triple_selected_hits']=len(wt&set(map(int,order[:k])))
out={'definition':'Pre-draw pair/triple signal audit. Scores use only history before each draw. Development 1000-1199 selects one operating point; fixed test 1200-1399 is untouched. 1402 is diagnostic only.','universes':{'pairs':P,'triples':Q,'winner_pairs_per_draw':10,'winner_triples_per_draw':10},'pair_k':PK,'triple_k':TK,'development':dev,'test':test,'selected_dev_only':sel,'draw1402_diagnostic':diag,'warning':'Many methods/K are displayed. The selected operating points are exploratory because multiple comparisons are present; only stability on the untouched test can justify keeping a signal.'}
p=ROOT/'data'/'miniloto-pair-triple-signal-audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
# compact summary for easy app/report use
compact={'selected_dev_only':sel,'draw1402_diagnostic':diag,'test_pair_all':test['pair'],'test_triple_all':test['triple']}
(ROOT/'data'/'miniloto-pair-triple-signal-summary.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(compact,ensure_ascii=False,indent=2),flush=True)
