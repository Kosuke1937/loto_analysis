import runpy, json, itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'research_miniloto_a_dynamic_promotion.py'))
combos=ns['combos']; draws=ns['draws']; A_SHAPES=ns['A_SHAPES']; draw_shapes=ns['draw_shapes']; A_idx=ns['A_idx']; maskA=ns['maskA']
a_stat_score=ns['a_stat_score']; a_dynamic_score=ns['a_dynamic_score']; combo_index=ns['combo_index']; z_role=ns['z_role']

def order(sc): return A_idx[np.lexsort((A_idx,-sc[A_idx]))]
def overlap(a,b): return len(set(map(int,a))&set(map(int,b)))
def eval_sel(sel,winner,wi):
    m=[overlap(combos[i],winner) for i in sel]; uni=set(combos[sel].ravel().tolist()); w=set(map(int,winner)); rec=len(w&uni)
    return {'exact':int(wi in set(map(int,sel))),'best':int(max(m)),'recall':rec,'union_size':len(uni)}
def caps_ok(cand,sel,numcap):
    c=set(map(int,combos[cand])); nums={}; pairs={}; triples={}
    for i in sel:
        s=set(map(int,combos[i]));
        for n in s: nums[n]=nums.get(n,0)+1
        for p in itertools.combinations(sorted(s),2): pairs[p]=pairs.get(p,0)+1
        for tr in itertools.combinations(sorted(s),3): triples[tr]=triples.get(tr,0)+1
    if any(nums.get(n,0)>=numcap for n in c): return False
    if any(pairs.get(p,0)>=2 for p in itertools.combinations(sorted(c),2)): return False
    if any(triples.get(tr,0)>=1 for tr in itertools.combinations(sorted(c),3)): return False
    return True

def diversified_astat(ast,n=10,scan=1500):
    od=order(ast)[:scan]
    for cap in (2,3,4):
        sel=[]
        for i in od:
            if caps_ok(int(i),sel,cap): sel.append(int(i))
            if len(sel)>=n: break
        if len(sel)>=n: return np.array(sel[:n],int),cap
    return np.array(list(map(int,od[:n])),int),4

def rescue(base,ast,dyn,slots=1,K=1000):
    sel=list(map(int,base)); zd=z_role(dyn); za=z_role(ast)
    od=order(dyn); kth=dyn[od[min(K-1,len(od)-1)]]; pool=A_idx[dyn[A_idx]>=kth]
    for _ in range(slots):
        usednums=set(combos[np.array(sel)].ravel().tolist()) if sel else set()
        usedpairs=set()
        for i in sel:
            usedpairs.update(itertools.combinations(sorted(map(int,combos[i])),2))
        best=None; bestkey=None
        for i in pool:
            i=int(i)
            if i in sel: continue
            c=set(map(int,combos[i])); newn=len(c-usednums)
            cp=set(itertools.combinations(sorted(c),2)); newp=len(cp-usedpairs)
            key=(newn,newp,float(zd[i]),float(za[i]),-i)
            if bestkey is None or key>bestkey: bestkey=key; best=i
        if best is not None: sel.append(best)
    return np.array(sel,int)

def summarize(xs):
    return {'n':len(xs),'exact5':sum(x['exact'] for x in xs),'best3plus':sum(x['best']>=3 for x in xs),'best4plus':sum(x['best']>=4 for x in xs),
            'union5':sum(x['recall']==5 for x in xs),'union4plus':sum(x['recall']>=4 for x in xs),'avg_best':float(np.mean([x['best'] for x in xs])),
            'avg_recall':float(np.mean([x['recall'] for x in xs])),'avg_union_size':float(np.mean([x['union_size'] for x in xs]))}

methods=['astat_raw10','astat_div10']
for slots in (1,2):
  for K in (500,1000,3000): methods.append(f'astat_div{10-slots}_dyn{slots}_K{K}')
records=[]
for rr in range(800,1400):
    t=rr-1
    if draw_shapes[t] not in A_SHAPES: continue
    winner=draws[t]; wi=combo_index[tuple(map(int,winner))]; ast=a_stat_score(t); dyn=a_dynamic_score(t)
    od=order(ast); r={'draw':rr,'methods':{}}
    r['methods']['astat_raw10']=eval_sel(od[:10],winner,wi)
    d10,cap=diversified_astat(ast,10); r['methods']['astat_div10']=eval_sel(d10,winner,wi); r['cap']=cap
    for slots in (1,2):
      base,_=diversified_astat(ast,10-slots)
      for K in (500,1000,3000):
        sel=rescue(base,ast,dyn,slots,K); r['methods'][f'astat_div{10-slots}_dyn{slots}_K{K}']=eval_sel(sel,winner,wi)
    records.append(r)

def block(lo,hi):
    rr=[r for r in records if lo<=r['draw']<=hi]
    return {'A_draws':len(rr),'methods':{m:summarize([r['methods'][m] for r in rr]) for m in methods},
            'cap_usage':{str(c):sum(r['cap']==c for r in rr) for c in (2,3,4)}}
blocks={'development_800_999':block(800,999),'validation_1000_1199':block(1000,1199),'diagnostic_1200_1399':block(1200,1399)}
# choose on development: exact, 4+,3+,union5,union4+,avg recall
D=blocks['development_800_999']['methods']
def k(m):
 x=D[m]; return (x['exact5'],x['best4plus'],x['best3plus'],x['union5'],x['union4plus'],x['avg_recall'])
selected=max(methods,key=k)
out={'protocol':{'A_layer_only':True,'development':[800,999],'validation':[1000,1199],'diagnostic':[1200,1399],'excluded':[1400,1401],
 'base':'A-Stat; diversification triple cap1, pair cap2, number cap 2->3->4; Dynamic rescue maximizes new number union, then new pair union, then Dynamic/A-Stat score.',
 'selection':'development only'},'selected_on_development':selected,'blocks':blocks}
path=ROOT/'data'/'miniloto-a-stat-dynamic-assembly-summary.json'; path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(out,ensure_ascii=False,indent=2))
