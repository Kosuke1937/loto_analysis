import re, base64, gzip, json
from pathlib import Path

src = Path('miniloto-history.html').read_text(encoding='utf-8')
m = re.search(r"const Z='([^']+)'", src, re.S)
if not m:
    raise SystemExit('compressed data not found')
z = re.sub(r'[^A-Za-z0-9+/=]', '', m.group(1))
raw = json.loads(gzip.decompress(base64.b64decode(z)).decode('utf-8'))
# raw row format: [draw,date,n1,n2,n3,n4,n5,bonus,jackpot_count,...]
out=[]
for r in raw:
    out.append({
        'draw': r[0], 'date': r[1], 'nums': r[2:7], 'bonus': r[7],
        'jackpot_count': r[8] if len(r)>8 else None
    })
Path('miniloto-data.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
print('wrote', len(out), 'records')
# touch 2026-08-19 to trigger workflow
