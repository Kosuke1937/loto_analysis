from pathlib import Path
p=Path('miniloto-development.html')
s=p.read_text(encoding='utf-8')
start='<!-- DIRECT_RANK_SUMMARY_START -->'
end='<!-- DIRECT_RANK_SUMMARY_END -->'
block='''<!-- DIRECT_RANK_SUMMARY_START -->
<div class="section">実当選組の直接ランキング到達度</div>
<div class="grid">
  <div class="kpi"><div class="label">評価区間</div><div class="n">1200–1399</div><div class="delta">200回</div></div>
  <div class="kpi"><div class="label">Committee Top500</div><div class="n">2 / 200</div></div>
  <div class="kpi"><div class="label">Committee Top1000</div><div class="n">3 / 200</div><div class="delta">厳しめの「直接上位化」基準</div></div>
  <div class="kpi"><div class="label">Committee Top3000</div><div class="n">8 / 200</div></div>
  <div class="kpi"><div class="label">Committee Top5000</div><div class="n">12 / 200</div></div>
  <div class="kpi"><div class="label">Committee Top10000</div><div class="n">20 / 200</div></div>
  <div class="kpi"><div class="label">代表例</div><div class="n">第1395回</div><div class="delta">Stat 106位 → Committee 127位 → 最終10口で完全一致</div></div>
</div>
<div class="note"><b>解釈：</b>実当選5数字そのものをCommitteeの極上位へ直接押し上げられた回は多くない。Top1000までを「そこそこ上位」と置くと3/200回。一方、Exact-Core v1は直接ランキングだけでなくOriginal / Cluster / Strong-Diversityの複数経路を使い、1000–1399の400回で完全一致3回（1044・1378・1395）を再現した。</div>
<div class="note warn"><b>研究方針：</b>今後は「当選組そのものを直接上位化する経路」を主評価し、Assembly/Repairは補助層として比較する。Top500/1000/3000/5000/10000の実当選組到達数を主要KPIとして継続記録する。</div>
<!-- DIRECT_RANK_SUMMARY_END -->'''
if start in s and end in s:
    a=s.index(start); b=s.index(end)+len(end); s=s[:a]+block+s[b:]
else:
    marker='<!-- EXACT_CORE_RESTORE_END -->'
    if marker in s:
        s=s.replace(marker, marker+'\n'+block, 1)
    else:
        s=s.replace('<div class="section">現在の正本</div>', block+'\n<div class="section">現在の正本</div>', 1)
p.write_text(s,encoding='utf-8')
print('updated',p)
