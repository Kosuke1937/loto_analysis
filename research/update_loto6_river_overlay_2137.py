from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "20260914-2137-overlay"
VERSION = "0.14.4"

river = ROOT / "loto6-river.html"
r = river.read_text(encoding="utf-8")

r = re.sub(r'App v0\.14\.\d+', f'App v{VERSION}', r)
r = r.replace('?v=20260914-2137', f'?v={TOKEN}')
r = r.replace(
    '横＝数字1〜43、縦＝抽せん回。最新回は赤。「前回予測」では最新確定回の抽せん前に成立していた条件を1〜43すべてに表示し、「次回」では最新回から見た条件を表示します。数字見出しまたは丸をタップすると、その数字の列を強調します。',
    '横＝数字1〜43、縦＝抽せん回。最新回は赤。最新確定回の行には、その回の抽せん前に成立していた条件リングを1〜43すべて重ねて表示します。「次回」行は最新回から見た条件です。数字見出しまたは丸をタップすると、その数字の列を強調します。'
)

r = r.replace(
    'const W=1280,L=62,R=14,T=44,rowH=27,forecastH=30,nextH=32,H=T+rowH*rs.length+forecastH+nextH+16,plotW=W-L-R,x=n=>L+(n-1)*plotW/42;',
    'const W=1280,L=62,R=14,T=44,rowH=27,nextH=32,H=T+rowH*rs.length+nextH+16,plotW=W-L-R,x=n=>L+(n-1)*plotW/42;'
)

pat = re.compile(
    r'\n  const py=T\+rs\.length\*rowH\+forecastH/2;.*?\n  const ny=T\+rs\.length\*rowH\+forecastH\+nextH/2;',
    re.S,
)
overlay = r'''
  // Overlay the previous forecast directly on the latest confirmed draw row.
  // This shows where all 1..43 numbers sat BEFORE that draw, behind/around the actual winners.
  if(rs.length){
    const latestTarget=rs.at(-1),ly=T+(rs.length-1)*rowH+rowH/2;
    for(let n=1;n<=43;n++){const xx=x(n),rel=relationMatches(latestTarget,n);
      if(relationOn.outside&&rel.outside)a.push(`<circle cx="${xx}" cy="${ly}" r="17" class="relationHalo outside"/>`);
      if(relationOn.prev35Same&&rel.prev35Same)a.push(`<circle cx="${xx}" cy="${ly}" r="15.5" class="relationHalo prev35Same"/>`);
      if(relationOn.prev2Near&&rel.prev2Near)a.push(`<circle cx="${xx}" cy="${ly}" r="14" class="relationHalo prev2Near"/>`);
      if(relationOn.prevNear&&rel.prevNear)a.push(`<circle cx="${xx}" cy="${ly}" r="11.5" class="relationHalo prevNear"/>`);
      if(relationOn.prevSame&&rel.prevSame)a.push(`<circle cx="${xx}" cy="${ly}" r="9.8" class="relationHalo prevSame"/>`);
    }
  }
  const ny=T+rs.length*rowH+nextH/2;'''
r2, n = pat.subn(overlay, r, count=1)
if n != 1:
    raise RuntimeError(f'previous forecast row block not found or ambiguous: {n}')
r = r2

# Keep the explanatory box, but make its wording match the overlay behavior.
r = r.replace(
    '前回予測（第${target[0]}回の抽せん前）4条件Union',
    '最新回に重ねた抽せん前条件（第${target[0]}回）4条件Union'
)

river.write_text(r, encoding="utf-8")

# Bump displayed app version/cache token consistently across top-level pages.
for p in ROOT.glob('*.html'):
    t = p.read_text(encoding='utf-8')
    t2 = re.sub(r'App v0\.14\.\d+', f'App v{VERSION}', t)
    t2 = t2.replace('?v=20260914-2137', f'?v={TOKEN}')
    t2 = re.sub(r'(loto6-chunk-\d+\.js\?v=)[^"\']+', rf'\g<1>{TOKEN}', t2)
    if t2 != t:
        p.write_text(t2, encoding='utf-8')

(ROOT / 'VERSION').write_text(VERSION + '\n', encoding='utf-8')
print('updated river overlay / app', VERSION)
