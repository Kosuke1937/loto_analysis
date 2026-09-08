import json
import re
from pathlib import Path

DATA_DIR = Path('data')
OUT = DATA_DIR / 'loto6-history.json'


def chunk_no(path: Path) -> int:
    m = re.search(r'loto6-chunk-(\d+)\.js$', path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


rows = []
for path in sorted(DATA_DIR.glob('loto6-chunk-*.js'), key=chunk_no):
    text = path.read_text(encoding='utf-8')
    m = re.search(r'\.push\((\[.*\])\);\s*$', text, re.S)
    if not m:
        raise SystemExit(f'chunk payload not found: {path}')
    payload = json.loads(m.group(1))
    rows.extend(payload)

out = []
last_draw = 0
for r in rows:
    if len(r) < 9:
        raise SystemExit(f'invalid row: {r}')
    draw = int(r[0])
    nums = [int(x) for x in r[2:8]]
    bonus = int(r[8])
    if draw <= last_draw:
        raise SystemExit(f'draw order error: {last_draw} -> {draw}')
    if len(nums) != 6 or len(set(nums)) != 6 or any(n < 1 or n > 43 for n in nums):
        raise SystemExit(f'invalid main numbers at draw {draw}: {nums}')
    if bonus < 1 or bonus > 43:
        raise SystemExit(f'invalid bonus at draw {draw}: {bonus}')
    out.append({
        'draw': draw,
        'date': r[1],
        'nums': nums,
        'bonus': bonus,
        'jackpot_count': r[9] if len(r) > 9 else None,
    })
    last_draw = draw

OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(f'wrote {len(out)} records to {OUT} (latest draw {out[-1]["draw"] if out else "-"})')
