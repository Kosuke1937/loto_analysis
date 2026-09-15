from pathlib import Path
import itertools, json, re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'miniloto-gap-tendencies.json'


def load_rows():
    rows=[]
    for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
        txt=p.read_text(encoding='utf-8')
        m=re.search(r'\.push\((\[.*\])\);?\s*$',txt,re.S)
        if not m:
            raise RuntimeError(f'cannot parse {p}')
        rows.extend(json.loads(m.group(1)))
    rows.sort(key=lambda r:int(r[0]))
    return rows


def features(nums):
    a=np.array(sorted(map(int,nums)),dtype=int)
    gaps=np.diff(a)
    return {
        'gaps':gaps,
        'range':int(a[-1]-a[0]),
        'min_gap':int(gaps.min()),
        'max_gap':int(gaps.max()),
        'gap_std':float(np.std(gaps)),
        'consecutive_pairs':int(np.sum(gaps==1)),
    }


def qstats(x):
    x=np.asarray(x,dtype=float)
    return {
        'mean':float(np.mean(x)),
        'median':float(np.median(x)),
        'q25':float(np.quantile(x,.25)),
        'q75':float(np.quantile(x,.75)),
    }


def summarize(draw_rows):
    feats=[features(r[2:7]) for r in draw_rows]
    gaps=np.vstack([f['gaps'] for f in feats])
    allg=gaps.ravel()
    gc=Counter(map(int,allg))
    n=len(feats)
    patterns=Counter('-'.join(map(str,map(int,g))) for g in gaps)
    consec=Counter(f['consecutive_pairs'] for f in feats)
    minc=Counter(f['min_gap'] for f in feats)
    maxc=Counter(f['max_gap'] for f in feats)
    return {
        'n_draws':n,
        'draw_range':[int(draw_rows[0][0]),int(draw_rows[-1][0])] if n else None,
        'adjacent_gap_counts':{str(k):int(gc[k]) for k in sorted(gc)},
        'adjacent_gap_share':{str(k):float(gc[k]/(4*n)) for k in sorted(gc)},
        'gap_position_stats':[
            {'position':i+1,**qstats(gaps[:,i]),'mode':int(Counter(map(int,gaps[:,i])).most_common(1)[0][0])}
            for i in range(4)
        ],
        'range_stats':qstats([f['range'] for f in feats]),
        'min_gap_stats':qstats([f['min_gap'] for f in feats]),
        'max_gap_stats':qstats([f['max_gap'] for f in feats]),
        'gap_std_stats':qstats([f['gap_std'] for f in feats]),
        'consecutive_pair_count_distribution':{str(k):int(consec[k]) for k in range(5)},
        'share_with_any_consecutive':float(sum(v for k,v in consec.items() if k>=1)/n),
        'share_with_2plus_consecutive_pairs':float(sum(v for k,v in consec.items() if k>=2)/n),
        'min_gap_distribution':{str(k):int(minc[k]) for k in sorted(minc)},
        'max_gap_distribution':{str(k):int(maxc[k]) for k in sorted(maxc)},
        'top_gap_patterns':[{'pattern':p,'count':int(c),'share':float(c/n)} for p,c in patterns.most_common(20)],
    }


def summarize_theory():
    combos=list(itertools.combinations(range(1,32),5))
    rows=[[i+1,'']+list(c)+[0,None] for i,c in enumerate(combos)]
    s=summarize(rows)
    s['note']='All 31C5=169,911 combinations equally weighted; this is the exact combinatorial baseline, not historical draws.'
    return s


def count_layer(cnt):
    if cnt is None: return None
    cnt=int(cnt)
    if cnt<=1: return '0-1'
    if cnt<=5: return '2-5'
    if cnt<=15: return '6-15'
    if cnt<=49: return '16-49'
    return '50+'


def crowd_gap_analysis(rows):
    rec=[]
    for r in rows:
        cnt=r[8] if len(r)>8 else None
        if cnt is None: continue
        f=features(r[2:7])
        rec.append({'draw':int(r[0]),'count':int(cnt),**{k:v for k,v in f.items() if k!='gaps'}})
    if not rec: return {}
    corr={}
    for key in ['range','min_gap','max_gap','gap_std','consecutive_pairs']:
        rho,p=spearmanr([x[key] for x in rec],[x['count'] for x in rec])
        corr[key]={'rho':float(rho),'p':float(p)}
    by=defaultdict(list)
    for x in rec: by[count_layer(x['count'])].append(x)
    order=['0-1','2-5','6-15','16-49','50+']
    groups={}
    for layer in order:
        xs=by.get(layer,[])
        if not xs: continue
        groups[layer]={
            'n':len(xs),
            'count_median':float(np.median([x['count'] for x in xs])),
            'range_median':float(np.median([x['range'] for x in xs])),
            'min_gap_median':float(np.median([x['min_gap'] for x in xs])),
            'max_gap_median':float(np.median([x['max_gap'] for x in xs])),
            'gap_std_median':float(np.median([x['gap_std'] for x in xs])),
            'consecutive_pairs_mean':float(np.mean([x['consecutive_pairs'] for x in xs])),
            'share_any_consecutive':float(np.mean([x['consecutive_pairs']>=1 for x in xs])),
        }
    return {'n':len(rec),'spearman_count_vs_gap_features':corr,'by_first_prize_count_layer':groups}


def main():
    rows=load_rows()
    latest=int(rows[-1][0])
    theory=summarize_theory()
    windows={}
    for w in [20,50,100,200,500]:
        windows[f'last{w}']=summarize(rows[-w:])
    windows['all']=summarize(rows)

    blocks={}
    for a,b in [(1,500),(501,1000),(1001,latest)]:
        z=[r for r in rows if a<=int(r[0])<=b]
        if z: blocks[f'{a}-{b}']=summarize(z)

    # Lift of observed adjacent-gap frequency vs exact combinatorial baseline.
    tshare={int(k):v for k,v in theory['adjacent_gap_share'].items()}
    lifts={}
    for name,s in windows.items():
        obs={int(k):v for k,v in s['adjacent_gap_share'].items()}
        lifts[name]={str(k):float(obs.get(k,0)/tshare[k]) for k in sorted(tshare) if tshare[k]>0}

    out={
        'title':'Mini Loto winning-number adjacent-gap tendencies',
        'latest_draw':latest,
        'definition':'For sorted winning numbers n1<n2<n3<n4<n5, gaps are g1=n2-n1, g2=n3-n2, g3=n4-n3, g4=n5-n4.',
        'theoretical_all_combinations':theory,
        'historical_windows':windows,
        'historical_blocks':blocks,
        'gap_frequency_lift_vs_theory':lifts,
        'first_prize_count_relation':crowd_gap_analysis(rows),
    }
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({
        'latest_draw':latest,
        'all':{k:windows['all'][k] for k in ['share_with_any_consecutive','share_with_2plus_consecutive_pairs','range_stats','min_gap_stats','max_gap_stats']},
        'last100':{k:windows['last100'][k] for k in ['share_with_any_consecutive','share_with_2plus_consecutive_pairs','range_stats','min_gap_stats','max_gap_stats']},
        'last20':{k:windows['last20'][k] for k in ['share_with_any_consecutive','share_with_2plus_consecutive_pairs','range_stats','min_gap_stats','max_gap_stats']},
        'crowd':out['first_prize_count_relation'],
    },ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
