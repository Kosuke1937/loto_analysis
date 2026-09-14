from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "20260914-2137"
VERSION = "0.14.3"

# 1) append draw 2137 to the latest data chunk
chunk = ROOT / "data" / "loto6-chunk-12.js"
s = chunk.read_text(encoding="utf-8")
row = '[2137,"2026/09/14",4,8,10,25,28,33,9,2]'
if row not in s:
    if not s.rstrip().endswith(']]);'):
        raise RuntimeError("unexpected loto6-chunk-12.js ending")
    s = s.rstrip()[:-4] + ',' + row + ']]);\n'
    chunk.write_text(s, encoding="utf-8")

# 2) update river page: latest draw + previous-forecast condition row
river = ROOT / "loto6-river.html"
r = river.read_text(encoding="utf-8")
r = re.sub(r'App v0\.14\.\d+', f'App v{VERSION}', r)
r = r.replace('?v=20260908-flow', f'?v={TOKEN}').replace('?v=20260910-2136', f'?v={TOKEN}')
r = r.replace(
    '横＝数字1〜43、縦＝抽せん回。最新回は赤。「次回」行にもON中の条件リングを表示します。数字見出しまたは丸をタップすると、その数字の列を強調します。',
    '横＝数字1〜43、縦＝抽せん回。最新回は赤。「前回予測」では最新確定回の抽せん前に成立していた条件を1〜43すべてに表示し、「次回」では最新回から見た条件を表示します。数字見出しまたは丸をタップすると、その数字の列を強調します。'
)
r = r.replace(
    '<div id="selectedBox" class="selectedBox"></div>\n  <div id="nextConditionBox" class="selectedBox"></div>',
    '<div id="selectedBox" class="selectedBox"></div>\n  <div id="prevConditionBox" class="selectedBox"></div>\n  <div id="nextConditionBox" class="selectedBox"></div>'
)
r = r.replace(
    'const W=1280,L=62,R=14,T=44,rowH=27,nextH=32,H=T+rowH*rs.length+nextH+16,plotW=W-L-R,x=n=>L+(n-1)*plotW/42;',
    'const W=1280,L=62,R=14,T=44,rowH=27,forecastH=30,nextH=32,H=T+rowH*rs.length+forecastH+nextH+16,plotW=W-L-R,x=n=>L+(n-1)*plotW/42;'
)
old_ny = '  const ny=T+rs.length*rowH+nextH/2;'
if old_ny not in r:
    raise RuntimeError('renderRiver next-row anchor not found')
prev_block = '''  const py=T+rs.length*rowH+forecastH/2;
  const prevTarget=DATA.length?DATA.at(-1):null;
  a.push(`<line class="rowline" x1="${L-6}" y1="${T+rs.length*rowH}" x2="${W-R}" y2="${T+rs.length*rowH}"/>`);
  a.push(`<text class="drawLabel" x="${L-9}" y="${py+4}" text-anchor="end">前回予測</text>`);
  for(let n=1;n<=43;n++){const xx=x(n),rel=prevTarget?relationMatches(prevTarget,n):{prevSame:false,prevNear:false,prev2Near:false,prev35Same:false,outside:false};
    if(relationOn.outside&&rel.outside)a.push(`<circle cx="${xx}" cy="${py}" r="16" class="relationHalo outside"/>`);
    if(relationOn.prev35Same&&rel.prev35Same)a.push(`<circle cx="${xx}" cy="${py}" r="14.5" class="relationHalo prev35Same"/>`);
    if(relationOn.prev2Near&&rel.prev2Near)a.push(`<circle cx="${xx}" cy="${py}" r="13" class="relationHalo prev2Near"/>`);
    if(relationOn.prevNear&&rel.prevNear)a.push(`<circle cx="${xx}" cy="${py}" r="10.5" class="relationHalo prevNear"/>`);
    if(relationOn.prevSame&&rel.prevSame)a.push(`<circle cx="${xx}" cy="${py}" r="9" class="relationHalo prevSame"/>`);
    a.push(`<circle data-n="${n}" cx="${xx}" cy="${py}" r="7" class="nextDot" style="cursor:pointer"/>`);if(n%5===0||n===1||n===43)a.push(`<text class="nextTxt" x="${xx}" y="${py+3}" text-anchor="middle">${n}</text>`)
  }
  a.push(`<line class="rowline" x1="${L-6}" y1="${T+rs.length*rowH+forecastH}" x2="${W-R}" y2="${T+rs.length*rowH+forecastH}"/>`);
  const ny=T+rs.length*rowH+forecastH+nextH/2;'''
r = r.replace(old_ny, prev_block)

old_box = '''function renderNextConditionBox(){
  const inside=[],outside=[];for(let n=1;n<=43;n++){const m=nextRelationMatches(n);(m.outside?outside:inside).push(n)}
  $('nextConditionBox').innerHTML=`<b>次回の4条件Union ${inside.length}数字</b><br>${inside.map(pad).join('・')}<br><b>4条件外 ${outside.length}数字</b><br>${outside.map(pad).join('・')}<br><span class="label">※ 条件外は除外候補ではなく、現在の4つの流れ条件では説明できない数字として別管理します。</span>`;
}'''
if old_box not in r:
    raise RuntimeError('renderNextConditionBox block not found')
new_box = '''function renderPreviousConditionBox(){
  const target=DATA.length?DATA.at(-1):null,inside=[],outside=[];
  if(!target){$('prevConditionBox').innerHTML='';return}
  for(let n=1;n<=43;n++){const m=relationMatches(target,n);(m.outside?outside:inside).push(n)}
  $('prevConditionBox').innerHTML=`<b>前回予測（第${target[0]}回の抽せん前）4条件Union ${inside.length}数字</b><br>${inside.map(pad).join('・')}<br><b>4条件外 ${outside.length}数字</b><br>${outside.map(pad).join('・')}`;
}
function renderNextConditionBox(){
  const inside=[],outside=[];for(let n=1;n<=43;n++){const m=nextRelationMatches(n);(m.outside?outside:inside).push(n)}
  $('nextConditionBox').innerHTML=`<b>次回の4条件Union ${inside.length}数字</b><br>${inside.map(pad).join('・')}<br><b>4条件外 ${outside.length}数字</b><br>${outside.map(pad).join('・')}<br><span class="label">※ 条件外は除外候補ではなく、現在の4つの流れ条件では説明できない数字として別管理します。</span>`;
}'''
r = r.replace(old_box, new_box)
r = r.replace(
    'function render(){const rs=rows();renderKpis(rs);renderRiver(rs);renderLists(rs);renderNextConditionBox();renderMeta(rs)}',
    'function render(){const rs=rows();renderKpis(rs);renderRiver(rs);renderLists(rs);renderPreviousConditionBox();renderNextConditionBox();renderMeta(rs)}'
)
river.write_text(r, encoding="utf-8")

# 3) update dashboard/latest result and Loto6 landing page
index = ROOT / "index.html"
i = index.read_text(encoding="utf-8")
i = re.sub(r'App v0\.14\.\d+', f'App v{VERSION}', i)
i = i.replace('更新 2026/09/10', '更新 2026/09/14')
i = i.replace('最新確定 第2136回｜2026/09/10', '最新確定 第2137回｜2026/09/14')
i = i.replace(
    '<span class="ball">06</span><span class="ball">07</span><span class="ball">33</span><span class="ball">37</span><span class="ball">41</span><span class="ball">43</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">42</span>',
    '<span class="ball">04</span><span class="ball">08</span><span class="ball">10</span><span class="ball">25</span><span class="ball">28</span><span class="ball">33</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">09</span>'
)
i = i.replace('href="loto6.html?v=20260910-2136"', f'href="loto6.html?v={TOKEN}"')
i = i.replace(
    '第2136回：06・07・33・37・41・43、B42。合計167、帯構成2-0-0-2-2。15口購入、最高1個一致、当選なし。',
    '第2137回：04・08・10・25・28・33、B09。合計108、帯構成2-1-2-1-0。Flow候補10口は最高2個一致、候補Unionは5/6（25のみ欠落）。購入実績は未反映。'
)
index.write_text(i, encoding="utf-8")

landing = ROOT / "loto6.html"
l = landing.read_text(encoding="utf-8")
l = re.sub(r'App v0\.14\.\d+', f'App v{VERSION}', l)
l = l.replace('最新確定：第2136回', '最新確定：第2137回')
l = l.replace('06・07・33・37・41・43 ／ BO 42', '04・08・10・25・28・33 ／ BO 09')
l = l.replace('抽せん日 2026/09/10｜合計167｜帯構成2-0-0-2-2｜前回本数字再登場2（07,37）｜購入15口：最高1個一致・当選なし', '抽せん日 2026/09/14｜合計108｜帯構成2-1-2-1-0｜前回本数字再登場1（33）｜Flow候補10口：最高2個一致・Union5/6（購入実績未反映）')
l = l.replace('第1回〜第2136回。', '第1回〜第2137回。')
l = l.replace('?v=20260910-2136', f'?v={TOKEN}').replace('?v=20260908-flow', f'?v={TOKEN}')
landing.write_text(l, encoding="utf-8")

# 4) cache-bust Loto6 data scripts/links across root HTML pages
for p in ROOT.glob('*.html'):
    t = p.read_text(encoding='utf-8')
    t2 = re.sub(r'(loto6-chunk-\d+\.js\?v=)[^"\']+', rf'\g<1>{TOKEN}', t)
    t2 = t2.replace('?v=20260910-2136', f'?v={TOKEN}').replace('?v=20260908-flow', f'?v={TOKEN}')
    if t2 != t:
        p.write_text(t2, encoding='utf-8')

(ROOT / 'VERSION').write_text(VERSION + '\n', encoding='utf-8')
print('updated Loto6 app to draw 2137 / app', VERSION)
