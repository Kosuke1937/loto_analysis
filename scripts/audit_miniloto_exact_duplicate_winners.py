import json,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=[]
for p in sorted((ROOT/'data').glob('miniloto-chunk-*.js')):
    txt=p.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?$',txt,re.S)
    if not m:
        raise SystemExit(f'cannot parse {p}')
    DATA.extend(json.loads(m.group(1)))
DATA.sort(key=lambda r:r[0])
by=collections.defaultdict(list)
for r in DATA:
    key=tuple(sorted(map(int,r[2:7])))
    by[key].append({'draw':int(r[0]),'date':r[1],'bonus':None if r[7] is None else int(r[7])})
dups=[]
for nums,rows in by.items():
    if len(rows)>=2:
        dups.append({'numbers':list(nums),'count':len(rows),'draws':rows})
dups.sort(key=lambda x:(-x['count'],x['numbers']))
out={
 'draw_count':len(DATA),
 'unique_main_combinations':len(by),
 'duplicate_combination_count':len(dups),
 'draws_in_duplicate_groups':sum(x['count'] for x in dups),
 'duplicate_groups':dups
}
p=ROOT/'data'/'miniloto-exact-duplicate-winners.json'
p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
