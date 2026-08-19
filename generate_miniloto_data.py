import re, base64, gzip, json
from pathlib import Path

src = Path('miniloto-history.html').read_text(encoding='utf-8')
m = re.search(r"const Z='([^']+)'", src, re.S)
if not m:
    raise SystemExit('compressed data not found')

z = re.sub(r'[^A-Za-z0-9+/=]', '', m.group(1))
z += '=' * (-len(z) % 4)
raw_bytes = base64.b64decode(z)
raw = json.loads(gzip.decompress(raw_bytes).decode('utf-8'))

out=[]
for r in raw:
    out.append({
        'draw': r[0],
        'date': r[1],
        'nums': r[2:7],
        'bonus': r[7],
        'jackpot_count': r[8] if len(r) > 8 else None
    })

if len(out) != 1400:
    raise SystemExit(f'unexpected record count: {len(out)}')

Path('miniloto-data.json').write_text(
    json.dumps(out, ensure_ascii=False, separators=(',', ':')),
    encoding='utf-8'
)
print('wrote', len(out), 'records')
