from pathlib import Path
p=Path('miniloto-development.html')
s=p.read_text(encoding='utf-8')
start='<!-- ASSEMBLY_VNEXT_FIXED_START -->'
end='<!-- ASSEMBLY_VNEXT_FIXED_END -->'
block='''<!-- ASSEMBLY_VNEXT_FIXED_START -->
<div class="section">Assembly-vNext 固定検証：Core 4 / Pair 3 / Novel 3</div>
<div class="grid">
  <div class="kpi"><div class="label">固定テスト 1200–1399</div><div class="n">3+ 41→39</div><div class="red">全面置換は悪化</div></div>
  <div class="kpi"><div class="label">4個以上一致</div><div class="n">3→3</div><div class="delta">増減なし</div></div>
  <div class="kpi"><div class="label">5個一致</div><div class="n">1→0</div><div class="red">第1395回完全一致を喪失</div></div>
  <div class="kpi"><div class="label">Winner Number Recall 5/5</div><div class="n">73→69</div><div class="red">Coverageも低下</div></div>
</div>
<div class="note danger"><b>結論：</b>Core Preserve 4口＋Pair Cross 3口＋Novel 3口で最終10口を全面置換する方式は不採用。固定テスト200回で3個以上一致41→39、4個以上3→3、完全一致1→0。paired gain/lossは3+で21/23、4+で2/2、5一致で0/1。</div>
<div class="note warn"><b>第1402回直接診断：</b>実際に提示した初期10口から「親口とちょうど3数字共有し、残り2数字を初期Unionから交換」する3-core再構築では、当選組01・04・20・25・29は候補には入ったがraw順位7,823位で、最終10口には入らなかった。生成10口の最大一致は3個。したがって「3-coreを残せば自動的に1等へ届く」わけではない。</div>
<div class="note ok"><b>次の研究枝：</b>正本10口を捨てず、9+1または8+2でCore Repairだけを追加・条件付き置換する。全面再構築ではなく、正本の5一致・4一致を保護したままAssembly失敗を救う補助レーンとして検証する。</div>
<!-- ASSEMBLY_VNEXT_FIXED_END -->'''
if start in s and end in s:
    a=s.index(start);b=s.index(end)+len(end);s=s[:a]+block+s[b:]
else:
    marker='<div class="section">次の判断基準</div>'
    if marker in s:s=s.replace(marker,block+'\n'+marker,1)
    else:s=s.replace('</div><script',block+'</div><script',1)
# normalize any stale 0.11.7 labels introduced by old dev page
s=s.replace('App v0.11.7','App v0.12.1').replace('Loto Analysis App v0.11.7','Loto Analysis App v0.12.1').replace('?v=0.11.7','?v=0.12.1-1402').replace('.js?v=0.11.7','.js?v=0.12.1-1402')
p.write_text(s,encoding='utf-8')
print('updated',p)
