import itertools, json, runpy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'research_miniloto_a_nearhit_ranks.py'))
draws=ns['draws']; draw_shapes=ns['draw_shapes']; A_SHAPES=ns['A_SHAPES']; score_A=ns['score_A']; Aidx=ns['Aidx']; combos=ns['combos']

def summarize(start,end):
    maxfreq=[]; distinct=[]; totals=[]; rows=[]
    for rr in range(start,end+1):
        t=rr-1
        if draw_shapes[t] not in A_SHAPES: continue
        sc=score_A(t,False)
        order=Aidx[np.lexsort((Aidx,-sc[Aidx]))][:100]
        top=[set(map(int,combos[i])) for i in order]
        winner=tuple(map(int,draws[t]))
        fs=[]
        for tri in itertools.combinations(winner,3):
            c=sum(set(tri).issubset(x) for x in top)
            fs.append((tri,int(c)))
        fs.sort(key=lambda x:(-x[1],x[0]))
        vals=[x[1] for x in fs]
        maxfreq.append(max(vals)); distinct.append(sum(v>0 for v in vals)); totals.append(sum(vals))
        rows.append({'draw':rr,'winner':list(winner),'winner_triples':[{'triple':list(tri),'count':c} for tri,c in fs]})
    n=len(maxfreq)
    return {
      'n_A_draws':n,
      'max_same_winner_triple_frequency_mean':float(np.mean(maxfreq)),
      'max_same_winner_triple_frequency_median':float(np.median(maxfreq)),
      'draws_maxfreq_ge_2':sum(v>=2 for v in maxfreq),
      'draws_maxfreq_ge_3':sum(v>=3 for v in maxfreq),
      'draws_maxfreq_ge_5':sum(v>=5 for v in maxfreq),
      'avg_distinct_winner_triples_present':float(np.mean(distinct)),
      'avg_total_winner_triple_occurrences':float(np.mean(totals)),
      'max_frequency_histogram':{str(k):sum(v==k for v in maxfreq) for k in range(max(maxfreq)+1)},
      'rows':rows
    }
out={'definition':'A-Stat Top100 only. For each actual winning 5-number set, count frequency of each of its 10 possible 3-number subsets among the Top100 tickets.',
     'development_800_999':summarize(800,999),'validation_1000_1199':summarize(1000,1199),'diagnostic_1200_1399':summarize(1200,1399),'excluded':[1400,1401]}
p=ROOT/'data'/'miniloto-astat-top100-winner-triple-frequency-fast.json'; p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for k,v in out.items():
    if isinstance(v,dict) and 'n_A_draws' in v: print(k,{kk:vv for kk,vv in v.items() if kk!='rows'},flush=True)
print('WROTE',p)
