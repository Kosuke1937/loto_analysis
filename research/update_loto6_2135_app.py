#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import itertools, json, re
ROOT=Path(__file__).resolve().parents[1]
VER='0.14.0'; CACHE='20260907-2135'
WIN=[7,13,32,34,37,39]; BO=41; SUM=sum(WIN); BAND='1-1-0-4-0'
TICKETS=[
 [6,35,36,37,41,42],
 [1,6,28,29,33,35],
 [1,14,29,41,42,43],
 [1,2,26,33,36,37],
 [5,16,28,36,40,42],
 [4,6,14,26,28,36],
 [14,16,17,28,37,43],
 [2,8,29,33,38,42],
 [7,8,16,29,37,39],
 [8,16,28,29,36,37],
]
CORE22=[42,41,43,40,35,37,36,38,34,31,33,32,30,2,26,7,14,1,29,28,17,8]
SAT6=[39,16,4,6,3,5]
COMMITTEE={'score':-0.7582293152809143,'rank':4866995,'space':6096454,'top_percent':79.83321124050144,'stat_rank':4283260,'core5_rank':5452117,'core4_rank':5374600}

def read(p): return (ROOT/p).read_text(encoding='utf-8')
def write(p,s): (ROOT/p).write_text(s,encoding='utf-8')
def p2(a): return ' '.join(f'{x:02d}' for x in a)
def band(a):
 return (sum(1 for x in a if x<=9),sum(1 for x in a if 10<=x<=19),sum(1 for x in a if 20<=x<=29),sum(1 for x in a if 30<=x<=39),sum(1 for x in a if x>=40))
def evalrow(a):
 m=sorted(set(a)&set(WIN)); b=BO in a
 prize='1等' if len(m)==6 else ('2等' if len(m)==5 and b else ('3等' if len(m)==5 else ('4等' if len(m)==4 else ('5等' if len(m)==3 else 'なし'))))
 return len(m),m,b,prize

# 1. append history safely by parsing JSON payload
p='data/loto6-chunk-12.js'; s=read(p); pre='window.LOTO6_CHUNKS=window.LOTO6_CHUNKS||[];window.LOTO6_CHUNKS.push('
payload=s.split('push(',1)[1].rsplit(');',1)[0]; rows=json.loads(payload)
if not any(int(r[0])==2135 for r in rows): rows.append([2135,'2026/09/07',7,13,32,34,37,39,41,0])
write(p,pre+json.dumps(rows,ensure_ascii=False,separators=(',',':'))+');\n')
meta=json.loads(read('data/loto6-meta.json')); meta['count']=2135;meta['last']=2135;meta['latest']=[2135,'2026/09/07',7,13,32,34,37,39,41,0]
write('data/loto6-meta.json',json.dumps(meta,ensure_ascii=False,indent=2)+'\n')

# gather all prior shapes from local app data for temporal/long-frequency audit
allrows=[]
for k in range(1,13):
 ss=read(f'data/loto6-chunk-{k}.js'); pp=ss.split('push(',1)[1].rsplit(');',1)[0]; allrows.extend(json.loads(pp))
prior=[r for r in allrows if int(r[0])<2135]
prior_nums=[tuple(map(int,r[2:8])) for r in prior]
sh=band(WIN); sh20=Counter(band(a) for a in prior_nums[-20:]); sh2150=Counter(band(a) for a in prior_nums[-50:-20]); sh50=Counter(band(a) for a in prior_nums[-50:]); shall=Counter(band(a) for a in prior_nums)
long_count=shall[sh]; long_pct=100*long_count/len(prior_nums)
layer='A' if long_pct>=2 else ('B' if long_pct>=1 else ('C' if long_pct>=0.5 else 'D'))

# 2. purchase/result and analysis JSON
purchase={
 'draw':2135,'date':'2026-09-07','result':{'main':WIN,'bonus':BO,'sum':SUM,'band':BAND,'odd_even':'4:2'},
 'purchase_count':10,'investment_yen':2000,'winning_amount_yen':1000,
 'tickets':[{'no':i+1,'numbers':a,'main_match':evalrow(a)[0],'matched_numbers':evalrow(a)[1],'bonus_in_ticket':evalrow(a)[2],'prize':evalrow(a)[3]} for i,a in enumerate(TICKETS)]
}
write('data/loto6-purchase-2135.json',json.dumps(purchase,ensure_ascii=False,indent=2)+'\n')
U=set().union(*map(set,TICKETS)); WP=set(WIN); pairs=set(itertools.combinations(sorted(WP),2)); triples=set(itertools.combinations(sorted(WP),3)); tpairs=set(c for a in TICKETS for c in itertools.combinations(sorted(a),2)); ttrip=set(c for a in TICKETS for c in itertools.combinations(sorted(a),3))
pool=set(CORE22)|set(SAT6)
analysis={
 'draw':2135,'winner':WIN,'bonus':BO,'sum':SUM,'band':BAND,'odd_even':'4:2','previous_main':[5,9,10,19,26,35],'previous_main_overlap':0,'previous_bonus':18,'previous_bonus_in_winner':False,
 'band_state':{'prior20_count':sh20[sh],'prior21_50_count':sh2150[sh],'prior50_count':sh50[sh],'classification':'50回新規型' if sh50[sh]==0 else '21-50で1回','long_history_count':long_count,'long_history_percent':round(long_pct,3),'ABCD_layer':layer},
 'committee_exact':{'formula':'Z(Stat500)+0.20Z(5core300)+0.15Z(4core300)',**COMMITTEE},
 'candidate_pool':{'core22':CORE22,'satellite6':SAT6,'winner_numbers_in_pool':sorted(WP&pool),'winner_number_recall':f'{len(WP&pool)}/6','missing_from_pool':sorted(WP-pool)},
 'purchased':{'tickets':10,'best_main_match':max(evalrow(a)[0] for a in TICKETS),'five_prize_tickets':sum(evalrow(a)[3]=='5等' for a in TICKETS),'winning_amount_yen':1000,'winner_number_recall':f'{len(WP&U)}/6','winner_pair_recall':f'{len(pairs&tpairs)}/15','winner_triple_recall':f'{len(triples&ttrip)}/20','winner_numbers_in_union':sorted(WP&U)},
 'postmortem':[
  '合計162は事前に設定した156-175帯に入り、前回104より高い方向性は通過',
  '帯構成1-1-0-4-0は直前20回0、21-50回0、50回0で50回新規型。時系列帯構成条件は通過',
  '前回本数字05・09・10・19・26・35の再登場は0、前回BO18も不在で既存条件は通過',
  'Core22+Satellite6は07・32・34・37・39の5/6を保持したが13を落とし、入口で完全一致は不可能',
  '最終10口unionでは32・34も落ち、Winner Number Recallは07・37・39の3/6まで低下',
  '購入9口目07・08・16・29・37・39が07・37・39の3個一致で5等1000円',
  '実当選Committee score=-0.758、全6096454通り中4866995位（上位79.8%地点）。高Committee score最大化は今回も不適合',
  'score=-0.758は事前に検証した構造+合計上昇Winnerの中央50%帯（約-0.85〜+0.56）内で、Winner-score帯を使う方針を支持する1例'
 ]
}
write('data/loto6-analysis-2135.json',json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')

# 3. dashboard
p='index.html'; s=read(p)
s=s.replace('App v0.13.0','App v'+VER).replace('更新 2026/09/03','更新 2026/09/07')
s=s.replace('最新確定 第2134回｜2026/09/03','最新確定 第2135回｜2026/09/07')
old='<span class="ball">05</span><span class="ball">09</span><span class="ball">10</span><span class="ball">19</span><span class="ball">26</span><span class="ball">35</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">18</span>'
new='<span class="ball">07</span><span class="ball">13</span><span class="ball">32</span><span class="ball">34</span><span class="ball">37</span><span class="ball">39</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">41</span>'
s=s.replace(old,new)
s=s.replace('<div class="n">¥8,000</div><div class="t">累積投資額</div>','<div class="n">¥10,000</div><div class="t">累積投資額</div>',1)
s=s.replace('<div class="n positive">¥2,000</div><div class="t">累積当選額</div>','<div class="n positive">¥3,000</div><div class="t">累積当選額</div>',1)
s=s.replace('<div class="n">25.0%</div><div class="t">回収率</div>','<div class="n">30.0%</div><div class="t">回収率</div>',1)
s=s.replace('loto6.html?v=20260903-2134',f'loto6.html?v={CACHE}')
s=s.replace('第2134回：05・09・10・19・26・35、B18。合計104、帯構成2-2-1-1-0、09-10連番。20口購入で5等2口（計¥2,000）。','第2135回：07・13・32・34・37・39、B41。合計162、帯構成1-1-0-4-0。10口購入で5等1口（¥1,000）。')
write(p,s)

# 4. loto6 landing
p='loto6.html';s=read(p)
s=s.replace('App v0.13.0','App v'+VER).replace('20260903-2134',CACHE)
start=s.index('<div class="latest">'); end=s.index('</div><div class="menu">',start)
latest=f'<div class="latest"><b>最新確定：第2135回</b><br>07・13・32・34・37・39 ／ BO 41<br><span style="color:var(--muted)">抽せん日 2026/09/07｜合計162｜帯構成1-1-0-4-0｜前回本数字再登場0｜購入10口：5等1口｜Committee実順位4,866,995位（上位79.8%地点）</span></div>'
s=s[:start]+latest+s[end+6:]
s=s.replace('第1回〜第2134回。','第1回〜第2135回。')
write(p,s)

# 5. history/trends
p='loto6-history.html';s=read(p).replace('第1回〜第2134回','第1回〜第2135回').replace('全2134回','全2135回').replace('DATA.length!==2134','DATA.length!==2135').replace('20260903-2134',CACHE);write(p,s)
p='loto6-trends.html';s=read(p).replace('App v0.13.0','App v'+VER).replace('Loto Analysis App v0.13.0','Loto Analysis App v'+VER).replace('第1〜2134回','第1〜2135回').replace('20260903-2134',CACHE);write(p,s)

# 6. records
p='loto6-records.html';s=read(p).replace('App v0.13.0','App v'+VER).replace('Loto Analysis App v0.13.0','Loto Analysis App v'+VER).replace('20260903-2134',CACHE)
s=s.replace('<div class="n">¥8,000</div></div><div class="card"><div class="label">累積当選額</div><div class="n">¥2,000</div></div><div class="card"><div class="label">回収率</div><div class="n">25.0%</div>', '<div class="n">¥10,000</div></div><div class="card"><div class="label">累積当選額</div><div class="n">¥3,000</div></div><div class="card"><div class="label">回収率</div><div class="n">30.0%</div>')
trs=[]
for i,a in enumerate(TICKETS,1):
 n,ms,b,pr=evalrow(a); hitcls='win' if n>=3 else ('hit2' if n==2 else ('hit1' if n==1 else 'none')); hit='0' if n==0 else f'{n}（'+','.join(f'{x:02d}' for x in ms)+'）'; bo=f'{BO:02d}' if b else '—'; pcl='win' if pr!='なし' else 'none'
 trs.append(f'<tr><td>{i}</td><td class="set">{p2(a)}</td><td class="{hitcls}">{hit}</td><td>{bo}</td><td class="{pcl}">{pr}</td></tr>')
record='<div class="record"><div class="recordHead"><div><div class="draw">第2135回</div><div class="date">2026-09-07　10口購入（ABCD・合計分散＋Core/Satellite＋弱い頻度補正）</div></div><div class="result">本数字 <span class="numbers">07 13 32 34 37 39</span><br>BO <span class="numbers">41</span></div></div><div class="tableWrap"><table><thead><tr><th>No.</th><th>購入口</th><th>本数字一致</th><th>BO</th><th>当選</th></tr></thead><tbody>'+''.join(trs)+'</tbody></table></div><div class="summary"><span>投資額 <b>¥2,000</b></span><span>当選額 <b>¥1,000</b></span><span>最高一致 <b>3個</b></span><span>当選口数 <b>1口（5等）</b></span><span>回収率 <b>50.0%</b></span></div></div>'
marker='<div class="record"><div class="recordHead"><div><div class="draw">第2134回</div>'
if '第2135回</div><div class="date">2026-09-07' not in s:s=s.replace(marker,record+marker)
write(p,s)

# 7. development
p='loto6-development.html';s=read(p).replace('App v0.13.0','App v'+VER).replace('Loto Analysis App v0.13.0','Loto Analysis App v'+VER).replace('20260903-2134',CACHE)
post=f'<div class="section">第2135回 Postmortem</div><div class="grid"><div class="card"><div class="label">実当選</div><div class="n">07 13 32 34 37 39</div><div class="txt">BO41、合計162、帯構成1-1-0-4-0、奇偶4:2。前回本数字再登場0、前回BO18なし。</div></div><div class="card"><div class="label">Committee実順位</div><div class="n">4,866,995位</div><div class="txt">全6,096,454通り、Score -0.758、上位79.8%地点。高スコア最大化では拾えないWinner。</div></div><div class="card"><div class="label">購入実績</div><div class="n">5等 × 1口</div><div class="txt">9口目 07・08・16・29・37・39 が07・37・39の3個一致。10口¥2,000に対し¥1,000回収。</div></div><div class="card"><div class="label">入口→Assembly</div><div class="n">5/6 → 3/6</div><div class="txt">Core22+Satellite6は07・32・34・37・39を保持したが13を欠落。最終10口では32・34も落ち、unionは07・37・39のみ。</div></div></div><div class="note"><b>重要：</b> 合計上昇、50回新規型の帯構成、前回数字0、BO除外はすべて通過。一方、Committee最大化は再び外れ方向で、実Winner score -0.758は過去の同条件Winner中央50%帯（約-0.85〜+0.56）内。今後は構造条件を先に満たし、Committeeは「高いほど良い」ではなくWinner-score帯への校正に使う。</div>'
marker='<div class="section">第2134回 Postmortem</div>'
if '第2135回 Postmortem' not in s:s=s.replace(marker,post+marker)
write(p,s)

# 8. roadmap latest section replacement
p='loto6-roadmap.html';s=read(p).replace('App v0.13.0','App v'+VER).replace('Loto Analysis App v0.13.0','Loto Analysis App v'+VER).replace('20260903-2134',CACHE)
start=s.index('<div class="section">最新：第2134回で分かったこと</div>');end=s.index('<div class="section">これまでの流れ</div>',start)
latest='<div class="section">最新：第2135回で分かったこと</div><div class="grid"><div class="card"><b>合計・帯構成方向は通過</b><p>実当選合計162は156-175枠。帯構成1-1-0-4-0は直前50回にない新規型で、時系列条件も通過した。</p></div><div class="card"><b>Core+Satelliteは5/6まで保持</b><p>07・32・34・37・39は入口28数字に残ったが13を落とした。完全一致の最初のボトルネックは数字プール。</p></div><div class="card"><b>最終Assemblyで3/6へ低下</b><p>10口全体では32・34も消え、07・37・39だけ残った。ただし9口目がこの3数字を同一口に集約し5等。</p></div><div class="card"><b>Committee上位偏重は不採用</b><p>実当選はScore -0.758、4,866,995位（上位79.8%地点）。今後は条件優先＋過去Winnerスコア帯への校正に変更する。</p></div></div>'
s=s[:start]+latest+s[end:]
write(p,s)
print(json.dumps({'long_shape_count':long_count,'long_shape_percent':long_pct,'layer':layer,'purchase_best':max(evalrow(a)[0] for a in TICKETS)},ensure_ascii=False))
