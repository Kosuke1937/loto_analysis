import itertools, json, runpy
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ns=runpy.run_path(str(ROOT/'scripts'/'research_miniloto_committee_v2.py'))
draws=ns['draws']; combos=ns['combos']; combo_shapes=ns['combo_shapes']; A_SHAPES=ns['A_SHAPES']; stat_score=ns['stat_score']; topk_idx=ns['topk_idx']

def summarize(start,end):
    rows=[]
    maxfreq=[]
    any2=any3=any5=0
    total_positive_triples=0
    total_occurrences=0
    for rr in range(start,end+1):
        t=rr-1
        if combo_shapes[ns['combo_index'][tuple(map(int,draws[t]))]] not in A_SHAPES:
            continue
        s=stat_score(t,role='A')
        top=topk_idx(s,100)
        top_sets=[set(map(int,combos[i])) for i in top]
        winner=tuple(map(int,draws[t])); triples=list(itertools.combinations(winner,3))
        freqs=[]
        for tri in triples:
            st=set(tri)
            c=sum(st.issubset(x) for x in top_sets)
            freqs.append({'triple':list(tri),'count':int(c)})
        freqs.sort(key=lambda x:(-x['count'],x['triple']))
        mf=freqs[0]['count']
        maxfreq.append(mf)
        any2 += mf>=2; any3 += mf>=3; any5 += mf>=5
        total_positive_triples += sum(x['count']>0 for x in freqs)
        total_occurrences += sum(x['count'] for x in freqs)
        rows.append({'draw':rr,'winner':list(winner),'max_winner_triple_frequency':int(mf),'winner_triples':freqs})
    n=len(rows)
    hist={str(k):sum(x==k for x in maxfreq) for k in range(0,max(maxfreq+[0])+1)}
    return {
      'n_A_draws':n,
      'max_frequency_mean':float(np.mean(maxfreq)) if n else 0,
      'max_frequency_median':float(np.median(maxfreq)) if n else 0,
      'draws_with_winner_triple_count_ge_2':int(any2),
      'draws_with_winner_triple_count_ge_3':int(any3),
      'draws_with_winner_triple_count_ge_5':int(any5),
      'avg_distinct_winner_triples_present_in_top100':float(total_positive_triples/n) if n else 0,
      'avg_total_winner_triple_occurrences_in_top100':float(total_occurrences/n) if n else 0,
      'max_frequency_histogram':hist,
      'rows':rows
    }

out={
 'definition':'For each A-layer draw, take A-Stat Top100. For each of the 10 three-number subsets of the actual winning 5 numbers, count how many Top100 tickets contain that exact triple.',
 'development_800_999':summarize(800,999),
 'validation_1000_1199':summarize(1000,1199),
 'diagnostic_1200_1399':summarize(1200,1399),
 'excluded':[1400,1401]
}
path=ROOT/'data'/'miniloto-astat-top100-winner-triple-frequency.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for k,v in out.items():
    if isinstance(v,dict) and 'n_A_draws' in v:
        print(k,{kk:vv for kk,vv in v.items() if kk!='rows'})
print('WROTE',path)
