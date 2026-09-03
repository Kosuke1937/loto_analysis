#!/usr/bin/env python3
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
VER='0.13.0'
CACHE='20260903-2134'
WIN=[5,9,10,19,26,35]; BO=18
GROUP1=[[2,7,16,18,37,42],[5,8,16,18,35,41],[2,5,26,34,35,37],[2,7,28,30,35,36],[5,7,26,32,33,36],[8,16,26,28,29,42],[7,8,29,32,35,37],[5,8,28,34,36,38],[2,16,26,30,41,43],[5,18,28,30,40,42]]
GROUP2=[[5,7,18,29,36,42],[5,16,18,28,35,41],[5,7,26,28,35,42],[2,8,16,28,35,36],[5,8,18,26,33,36],[5,8,16,26,29,42],[7,8,18,28,32,35],[5,8,16,28,34,36],[2,5,16,26,30,43],[5,7,18,28,36,40]]

def write(p,s): (ROOT/p).write_text(s,encoding='utf-8')
def read(p): return (ROOT/p).read_text(encoding='utf-8')
def padset(a): return ' '.join(f'{x:02d}' for x in a)
def evalrow(a):
 m=sorted(set(a)&set(WIN)); b=BO in a
 prize='5等' if len(m)==3 else ('4等' if len(m)==4 else ('3等' if len(m)==5 and not b else ('2等' if len(m)==5 and b else ('1等' if len(m)==6 else 'なし'))))
 return len(m),m,b,prize

# 1) history data
p='data/loto6-chunk-12.js'; s=read(p)
if '[2134,' not in s:
 s=s.replace(']]);',',[2134,"2026/09/03",5,9,10,19,26,35,18,null]]);')
write(p,s)
meta=json.loads(read('data/loto6-meta.json')); meta['count']=2134;meta['last']=2134;meta['latest']=[2134,'2026/09/03',5,9,10,19,26,35,18,None]
write('data/loto6-meta.json',json.dumps(meta,ensure_ascii=False,indent=2)+'\n')

# 2) reproducible purchase/result data
purchase={
 'draw':2134,'date':'2026-09-03','result':{'main':WIN,'bonus':BO,'sum':104,'band':'2-2-1-1-0'},
 'purchase_count':20,'investment_yen':4000,'winning_amount_yen':2000,
 'groups':[
  {'name':'21数字からの再構築','tickets':[{'no':i+1,'numbers':a,'main_match':evalrow(a)[0],'matched_numbers':evalrow(a)[1],'bonus_in_ticket':evalrow(a)[2],'prize':evalrow(a)[3]} for i,a in enumerate(GROUP1)]},
  {'name':'修正版Assembly','tickets':[{'no':i+1,'numbers':a,'main_match':evalrow(a)[0],'matched_numbers':evalrow(a)[1],'bonus_in_ticket':evalrow(a)[2],'prize':evalrow(a)[3]} for i,a in enumerate(GROUP2)]}
 ]
}
write('data/loto6-purchase-2134.json',json.dumps(purchase,ensure_ascii=False,indent=2)+'\n')
analysis={
 'draw':2134,'winner':WIN,'bonus':BO,'sum':104,'band':'2-2-1-1-0','odd_even':'4:2','previous_main_overlap':0,
 'band_state':{'prior20_count':0,'prior21_50_count':0,'prior50_count':0,'classification':'50回新規型'},
 'committee_exact':{'formula':'Z(Stat500)+0.20Z(5core300)+0.15Z(4core300)','stat_rank':4439322,'core5_rank':1079751,'core4_rank':689706,'committee_rank':3535934,'space':6096454,'committee_score':-0.052,'top_percent':58.0},
 'purchased':{'tickets':20,'best_main_match':3,'five_prize_tickets':2,'winner_number_recall_group1':'3/6','winner_number_recall_group2':'3/6'},
 'postmortem':['帯構成2-2-1-1-0は直前50回未出で、50回新規型シナリオは通過','直近20回に出た数字だけという条件は当選6数字すべて通過','09・10・19をハード除外したため、最終21数字プールでは完全一致が構造的に不可能','Committee実順位は3,535,934位で、今回はCommittee低位型','購入20口では05・26・35の3個一致が2口あり、5等2口']
}
write('data/loto6-analysis-2134.json',json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')

# 3) root dashboard
p='index.html';s=read(p)
s=s.replace('App v0.12.2','App v'+VER)
s=s.replace('最新確定 第2133回｜2026/08/31','最新確定 第2134回｜2026/09/03')
old='<span class="ball">01</span><span class="ball">11</span><span class="ball">14</span><span class="ball">20</span><span class="ball">29</span><span class="ball">38</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">27</span>'
new='<span class="ball">05</span><span class="ball">09</span><span class="ball">10</span><span class="ball">19</span><span class="ball">26</span><span class="ball">35</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">18</span>'
s=s.replace(old,new)
s=s.replace('<div class="n">¥4,000</div><div class="t">累積投資額</div>','<div class="n">¥8,000</div><div class="t">累積投資額</div>',1)
s=s.replace('<div class="n">¥0</div><div class="t">累積当選額</div>','<div class="n positive">¥2,000</div><div class="t">累積当選額</div>',1)
s=s.replace('<div class="n">0.0%</div><div class="t">回収率</div>','<div class="n">25.0%</div><div class="t">回収率</div>',1)
s=s.replace('loto6.html?v=0.12.2-2133',f'loto6.html?v={CACHE}')
s=s.replace('第2133回：01・11・14・20・29・38、B27。合計113、帯構成1-2-2-1-0、連番なし。1等1口、次回CO 27,341,485円。','第2134回：05・09・10・19・26・35、B18。合計104、帯構成2-2-1-1-0、09-10連番。20口購入で5等2口（計¥2,000）。')
write(p,s)

# 4) loto6 landing
p='loto6.html';s=read(p)
s=s.replace('App v0.12.2','App v'+VER).replace('./?v=0.12.2',f'./?v={CACHE}')
start=s.index('<div class="latest">');end=s.index('</div><div class="menu">',start)
latest='<div class="latest"><b>最新確定：第2134回</b><br>05・09・10・19・26・35 ／ BO 18<br><span style="color:var(--muted)">抽せん日 2026/09/03｜合計104｜帯構成2-2-1-1-0｜09-10連番｜購入20口：5等2口｜Committee実順位3,535,934位</span></div>'
s=s[:start]+latest+s[end+6:]
s=s.replace('loto6-history.html?v=20260831-2133',f'loto6-history.html?v={CACHE}').replace('第1回〜第2133回。', '第1回〜第2134回。')
s=s.replace('loto6-trends.html?v=20260831-2133',f'loto6-trends.html?v={CACHE}')
s=s.replace('loto6-roadmap.html?v=0.12.2',f'loto6-roadmap.html?v={CACHE}').replace('loto6-development.html?v=0.12.2',f'loto6-development.html?v={CACHE}').replace('loto6-records.html?v=0.12.2',f'loto6-records.html?v={CACHE}')
write(p,s)

# 5) history page: fix stale guard/text/cache
p='loto6-history.html';s=read(p)
s=s.replace('第1回〜第2131回','第1回〜第2134回').replace('全2131回','全2134回').replace('DATA.length!==2131','DATA.length!==2134')
s=s.replace('v=20260824-1952',f'v={CACHE}').replace('?v=1952',f'?v={CACHE}')
write(p,s)

# 6) trends page: dynamic data already works, fix stale labels/cache/version
p='loto6-trends.html';s=read(p)
s=s.replace('App v0.11.7','App v'+VER).replace('Loto Analysis App v0.11.7','Loto Analysis App v'+VER).replace('第1〜2131回','第1〜2134回').replace('v=0.11.7',f'v={CACHE}')
write(p,s)

# 7) records page: add exact purchased 20 lines and totals
p='loto6-records.html';s=read(p)
s=s.replace('App v0.11.7','App v'+VER).replace('Loto Analysis App v0.11.7','Loto Analysis App v'+VER).replace('v=0.11.7',f'v={CACHE}')
s=s.replace('.none{color:#7f8a9b}', '.none{color:#7f8a9b}.win{color:#7ce0a7;font-weight:800}')
s=s.replace('<div class="n">¥4,000</div></div><div class="card"><div class="label">累積当選額</div><div class="n">¥0</div></div><div class="card"><div class="label">回収率</div><div class="n">0.0%</div>', '<div class="n">¥8,000</div></div><div class="card"><div class="label">累積当選額</div><div class="n">¥2,000</div></div><div class="card"><div class="label">回収率</div><div class="n">25.0%</div>')
rows=[]
for prefix,arr in [('R',GROUP1),('A',GROUP2)]:
 for i,a in enumerate(arr,1):
  n,ms,b,pr=evalrow(a);hitcls='win' if n>=3 else ('hit2' if n==2 else ('hit1' if n==1 else 'none')); hit='0' if n==0 else f'{n}（'+','.join(f'{x:02d}' for x in ms)+'）'; bo=f'{BO:02d}' if b else '—'; pcl='win' if pr!='なし' else 'none'
  rows.append(f'<tr><td>{prefix}{i}</td><td class="set">{padset(a)}</td><td class="{hitcls}">{hit}</td><td>{bo}</td><td class="{pcl}">{pr}</td></tr>')
record='<div class="record"><div class="recordHead"><div><div class="draw">第2134回</div><div class="date">2026-09-03　20口購入（R=21数字再構築 / A=修正版Assembly）</div></div><div class="result">本数字 <span class="numbers">05 09 10 19 26 35</span><br>BO <span class="numbers">18</span></div></div><div class="tableWrap"><table><thead><tr><th>No.</th><th>購入口</th><th>本数字一致</th><th>BO</th><th>当選</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><div class="summary"><span>投資額 <b>¥4,000</b></span><span>当選額 <b>¥2,000</b></span><span>最高一致 <b>3個</b></span><span>当選口数 <b>2口（5等）</b></span><span>回収率 <b>50.0%</b></span></div></div>'
marker='<div class="record"><div class="recordHead"><div><div class="draw">第2131回</div>'
if '第2134回</div><div class="date">2026-09-03' not in s: s=s.replace(marker,record+marker)
write(p,s)

# 8) development page: add postmortem and update version/cache
p='loto6-development.html';s=read(p)
s=s.replace('App v0.11.7','App v'+VER).replace('Loto Analysis App v0.11.7','Loto Analysis App v'+VER).replace('v=0.11.7',f'v={CACHE}')
post='<div class="section">第2134回 Postmortem</div><div class="grid"><div class="card"><div class="label">実当選</div><div class="n">05 09 10 19 26 35</div><div class="txt">BO18、合計104、帯構成2-2-1-1-0。直前20回0回・21〜50回0回で、直近50回新規型シナリオに該当。</div></div><div class="card"><div class="label">Committee実順位</div><div class="n">3,535,934位</div><div class="txt">全6,096,454通り。Score -0.052、上位約58%。今回はCommittee低位型で、Committee単独では拾えない。</div></div><div class="card"><div class="label">購入実績</div><div class="n">5等 × 2口</div><div class="txt">21数字再構築R3と修正版Assembly A3が05・26・35の3個一致。20口¥4,000に対し¥2,000回収。</div></div><div class="card"><div class="label">最大の失敗モード</div><div class="n">09・10・19を除外</div><div class="txt">直近20回出現済み条件はWinner 6/6通過したが、その後の低頻度ハード除外で当選数字3個を落とした。</div></div></div><div class="note"><b>次の検証：</b> 帯構成の「直近20未出＋21〜50で1回または50回新規」は維持候補。一方、低頻度数字の完全除外はsoft penaltyへ変更をバックテストする。数字プールはCore＋Satelliteに分け、同一帯構成の10口内集中を抑えつつAssemblyを改善する。</div>'
marker='<div class="section">直近2130〜2131回からの更新</div>'
if '第2134回 Postmortem' not in s: s=s.replace(marker,post+'<div class="section">直近2130〜2134回からの更新</div>')
else: s=s.replace(marker,'<div class="section">直近2130〜2134回からの更新</div>')
write(p,s)

# 9) roadmap page: add latest lesson
p='loto6-roadmap.html';s=read(p)
s=s.replace('App v0.12.0','App v'+VER).replace('Loto Analysis App v0.12.0','Loto Analysis App v'+VER).replace('v=0.12.0',f'v={CACHE}')
latest='<div class="section">最新：第2134回で分かったこと</div><div class="grid"><div class="card"><b>帯構成シナリオは残した</b><p>実当選2-2-1-1-0は直前50回にない新規型。時系列帯構成の考え方はWinnerを除外しなかった。</p></div><div class="card"><b>数字ハード除外がWinnerを壊した</b><p>09・10・19を完全除外したため、最終21数字から1等組を再構成できなかった。Core＋Satelliteへ見直す。</p></div><div class="card"><b>Committeeは低位型</b><p>実当選は3,535,934位 / 6,096,454。Committee高順位だけで最終口を作る設計は不十分。</p></div><div class="card"><b>Assemblyは3個一致を2口生成</b><p>購入20口のうち2口で05・26・35が3個一致。候補数字抽出と同一口への集約を分けて評価する。</p></div></div>'
marker='<div class="section">これまでの流れ</div>'
if '最新：第2134回で分かったこと' not in s: s=s.replace(marker,latest+marker)
write(p,s)

print('updated Loto6 app to draw 2134')
