import json,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if not m: raise SystemExit(f'cannot parse {p}')
    DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])
by=collections.defaultdict(list)
for r in DATA:
    key=tuple(sorted(r[2:7]))
    by[key].append({'draw':r[0],'date':r[1],'bonus':r[7]})
dups=[{'numbers':list(k),'occurrences':v,'times':len(v)} for k,v in sorted(by.items()) if len(v)>=2]
out={'n_draws':len(DATA),'n_unique_combinations':len(by),'n_duplicate_combination_types':len(dups),'n_repeat_draws_beyond_first':sum(x['times']-1 for x in dups),'duplicates':dups}
p=ROOT/'data'/'miniloto-history-duplicate-audit.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
