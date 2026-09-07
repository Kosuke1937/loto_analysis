from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Restore Exact-Core v1 summary + winner score display in development page.
p = ROOT / 'miniloto-development.html'
s = p.read_text(encoding='utf-8')

exact = '''
<!-- EXACT_CORE_RESTORE_START -->
<div class="section">Exact-Core v1：過去検証</div>
<div class="grid">
  <div class="kpi"><div class="label">評価区間</div><div class="n">1000–1399</div><div class="delta">400回</div></div>
  <div class="kpi"><div class="label">平均購入口数</div><div class="n">約19.92口</div></div>
  <div class="kpi"><div class="label">3個以上一致</div><div class="n">123 / 400</div></div>
  <div class="kpi"><div class="label">4個以上一致</div><div class="n">8 / 400</div></div>
  <div class="kpi"><div class="label">5個完全一致</div><div class="n">3 / 400</div><div class="delta">1044・1378・1395</div></div>
  <div class="kpi"><div class="label">構成</div><div class="n" style="font-size:16px">Original + Cluster + Strong-Diversity</div></div>
</div>
<div class="note"><b>Exact-Core v1</b> は、Original Committee先頭8口、Cluster-Leader先頭8口、Strong-Diversity先頭8口をunionして重複除去する複数経路方式。完全一致は第1044回＝Strong-Diversity、第1378回＝Cluster-Leader、第1395回＝Original Committeeで確認。</div>
<div class="note warn">この3回は研究上の重要な再現結果だが、Cluster/Exactルート開発に後期データを使っているため、将来性能の証明ではない。正本Committeeとは分離して研究枝として扱う。</div>
<div class="card"><div class="head"><div class="name">実当選組スコア時系列</div><span class="status research">復元</span></div><div class="txt">第1200〜1399回の実当選5数字について、A1、4core、Committee、A-Stat/A-Dynamic/B-Statなどのスコア推移を保存済みデータから確認できる。</div><div class="txt"><a href="miniloto-score-traces.html?v=restore-20260907" style="color:#9fc0ff;font-weight:800">スコア時系列ページを開く →</a></div></div>
<!-- EXACT_CORE_RESTORE_END -->
'''

if '<!-- EXACT_CORE_RESTORE_START -->' not in s:
    marker = '<div class="section">現在の正本</div>'
    if marker not in s:
        raise SystemExit('current canonical section marker not found')
    s = s.replace(marker, exact + marker, 1)

old = "if(d)$('sel').innerHTML=`<b>第${d.draw}回</b><br>${d.nums.map(n=>String(n).padStart(2,'0')).join('・')}<br>Stat ${d.statRank.toLocaleString()}位 / 4core ${d.coreRank.toLocaleString()}位 / Committee ${d.committeeRank.toLocaleString()}位`"
new = "if(d)$('sel').innerHTML=`<b>第${d.draw}回</b><br>${d.nums.map(n=>String(n).padStart(2,'0')).join('・')}<br>Stat score <b>${Number(d.statScore).toFixed(6)}</b> / ${d.statRank.toLocaleString()}位<br>4core score <b>${Number(d.coreScore).toFixed(3)}</b> / ${d.coreRank.toLocaleString()}位<br>Committee score <b>${Number(d.committeeScore).toFixed(6)}</b> / ${d.committeeRank.toLocaleString()}位`"
if old in s:
    s = s.replace(old, new, 1)
elif 'Stat score <b>${Number(d.statScore)' not in s:
    raise SystemExit('winner selected display pattern not found')

p.write_text(s, encoding='utf-8')

# Restore navigation to the score-trace page from Mini Loto menu.
p = ROOT / 'miniloto.html'
s = p.read_text(encoding='utf-8')
score_card = '<a class="card" href="miniloto-score-traces.html?v=restore-20260907"><b>実当選組スコア履歴</b><span>第1200〜1399回の当選5数字に対するA1・4core・Committee・各Specialistのスコア時系列</span></a>'
if 'miniloto-score-traces.html' not in s:
    needle = '</div></div></body></html>'
    if needle not in s:
        raise SystemExit('menu closing marker not found')
    s = s.replace(needle, score_card + needle, 1)
p.write_text(s, encoding='utf-8')

# Validation.
dev = (ROOT / 'miniloto-development.html').read_text(encoding='utf-8')
menu = (ROOT / 'miniloto.html').read_text(encoding='utf-8')
assert 'Exact-Core v1：過去検証' in dev
assert '1044・1378・1395' in dev
assert 'Stat score <b>${Number(d.statScore).toFixed(6)}</b>' in dev
assert 'Committee score <b>${Number(d.committeeScore).toFixed(6)}</b>' in dev
assert 'miniloto-score-traces.html' in dev
assert '実当選組スコア履歴' in menu
print('restored Exact-Core v1 and winner score UI')
