from pathlib import Path
p=Path('miniloto-development.html');s=p.read_text(encoding='utf-8')
start='<!-- PAIR_TRIPLE_SIGNAL_AUDIT_START -->';end='<!-- PAIR_TRIPLE_SIGNAL_AUDIT_END -->'
block='''<!-- PAIR_TRIPLE_SIGNAL_AUDIT_START -->
<div class="section">Pair / Triple 独立予測信号監査</div>
<div class="grid">
  <div class="kpi"><div class="label">Pair raw300 Top50</div><div class="n">1.01〜1.13×</div><div class="delta">600–1399の区間別ランダム期待比</div></div>
  <div class="kpi"><div class="label">Triple raw100 Top300</div><div class="n">0.99〜1.20×</div><div class="delta">後半1000–1399だけやや強い</div></div>
  <div class="kpi"><div class="label">1200–1399 Pair</div><div class="n">1.210 / 10</div><div class="delta">ランダム1.075、約1.125倍</div></div>
  <div class="kpi"><div class="label">1200–1399 Triple</div><div class="n">0.775 / 10</div><div class="delta">ランダム0.667、約1.161倍</div></div>
</div>
<div class="note warn"><b>固定ストレス結果：</b>Pair raw300 Top50は600–799=1.060倍、800–999=1.009倍、1000–1199=1.093倍、1200–1399=1.125倍。Triple raw100 Top300は0.989倍、1.019倍、1.199倍、1.161倍。古い区間ではほぼランダムで、後半だけ強く見えるため恒常的予測信号とはみなさない。</div>
<div class="note danger"><b>直近1400–1402：</b>Pair raw300 Top50は3回合計1 hit（期待約3.23）、Triple raw100 Top300は0 hit（期待約2.00）。直近では両方とも崩れており、単純頻度を主Assembly条件へ昇格しない。</div>
<div class="note ok"><b>次の研究枝：</b>Pair/Triple頻度を直接採用するのではなく、なぜ1000–1399でのみ強く見えたかを状態依存で監査する。候補として窓間ドリフト、pair/triple分布entropy、集中度、Committee構造状態との関係を調べ、事前に効く/効かない状態を識別できるか検証する。</div>
<!-- PAIR_TRIPLE_SIGNAL_AUDIT_END -->'''
if start in s and end in s:
 a=s.index(start);b=s.index(end)+len(end);s=s[:a]+block+s[b:]
else:
 marker='<div class="section">現在の正本</div>'
 s=s.replace(marker,block+'\n'+marker,1) if marker in s else s.replace('</div><script',block+'</div><script',1)
p.write_text(s,encoding='utf-8');print('updated')
