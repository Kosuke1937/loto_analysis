from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# Root dashboard: refresh Mini Loto latest draw and provisional payout display.
p = ROOT / 'index.html'
t = p.read_text(encoding='utf-8')
t = t.replace('App v0.14.0', 'App v0.14.1')
t = t.replace('更新 2026/09/07', '更新 2026/09/08')
t = t.replace('最新確定 第1402回｜2026/09/01', '最新確定 第1403回｜2026/09/08')
t = t.replace('<span class="ball">01</span><span class="ball">04</span><span class="ball">20</span><span class="ball">25</span><span class="ball">29</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">22</span>', '<span class="ball">08</span><span class="ball">11</span><span class="ball">14</span><span class="ball">17</span><span class="ball">19</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">24</span>', 1)
t = t.replace('<div class="n">¥7,600</div><div class="t">累積投資額</div>', '<div class="n">¥11,600</div><div class="t">累積投資額</div>', 1)
t = t.replace('<div class="n positive">¥1,100</div><div class="t">累積当選額</div>', '<div class="n positive">¥2,100*</div><div class="t">累積当選額</div>', 1)
t = t.replace('<div class="n">14.5%</div><div class="t">回収率</div>', '<div class="n">18.1%*</div><div class="t">回収率</div>', 1)
t = t.replace('href="miniloto.html?v=0.12.2"', 'href="miniloto.html?v=0.12.3-1403"')
t = t.replace('第1402回：01・04・20・25・29、B22。1等7口。再構築10口を実購入、最大2個一致・当選なし。', '第1403回：08・11・14・17・19、B24。20口購入で4等1口。*第1403回4等は1,000円を暫定反映（公式理論値）。回別実額公開後に更新。')
p.write_text(t, encoding='utf-8')

# Mini Loto records: show the same provisional cumulative payout/recovery.
p = ROOT / 'miniloto-records.html'
t = p.read_text(encoding='utf-8')
t = t.replace('累積当選額</div><div class="n">¥1,100＋4等確定', '累積当選額</div><div class="n">¥2,100*')
t = t.replace('回収率</div><div class="n">当せん金反映待ち', '回収率</div><div class="n">18.1%*')
t = t.replace('当せん金は公式額確認後に累積へ確定反映。', '4等1口は1,000円を暫定反映（公式理論値）。回別実額公開後に確定額へ置換。')
# If the prior updater was not reflected for any reason, normalize those stats too.
t = t.replace('累積投資額</div><div class="n">¥7,600', '累積投資額</div><div class="n">¥11,600')
t = t.replace('登録購入口</div><div class="n">38口', '登録購入口</div><div class="n">58口')
p.write_text(t, encoding='utf-8')

print('updated index.html and miniloto-records.html for draw 1403 provisional payout')
