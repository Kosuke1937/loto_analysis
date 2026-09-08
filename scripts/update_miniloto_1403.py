from pathlib import Path
import json, re, itertools, statistics

ROOT=Path(__file__).resolve().parents[1]
DRAW=1403
DATE='2026/09/08'
WIN=[8,11,14,17,19]
BONUS=24
VERSION='0.12.3'
STAMP='20260908-1403'

revised=[
[4,17,18,23,28],[4,14,19,23,31],[11,14,17,21,27],[13,15,16,26,29],[14,16,19,22,29],
[5,15,22,29,31],[2,9,18,27,31],[4,16,18,19,27],[4,5,17,27,31],[2,14,17,18,29],
]
committee=[
[9,12,21,23,29],[10,19,25,26,27],[1,11,21,24,30],[9,20,21,25,31],[1,4,19,27,31],
[2,10,24,29,30],[3,11,23,29,31],[12,19,20,24,26],[9,20,23,27,30],[4,11,25,26,28],
]

# 1) append history row
p=ROOT/'data/miniloto-chunk-8.js'
s=p.read_text(encoding='utf-8')
row=f'[{DRAW},"{DATE}",8,11,14,17,19,24,null]'
if row not in s:
    s=s.replace(']);',','+row+']);')
    p.write_text(s,encoding='utf-8')

# parse all rows after append
rows=[]
for i in range(1,9):
    q=ROOT/'data'/f'miniloto-chunk-{i}.js'
    t=q.read_text(encoding='utf-8')
    m=re.search(r'push\((\[.*\])\);?\s*$',t,re.S)
    if not m: raise RuntimeError(q)
    rows.extend(json.loads(m.group(1)))
rows.sort(key=lambda r:r[0])
assert rows[-1][0]==1403 and len(rows)==1403

def band(nums):
    c=[0,0,0,0]
    for n in nums: c[0 if n<=9 else 1 if n<=19 else 2 if n<=29 else 3]+=1
    return '-'.join(map(str,c))
shapes={}
for r in rows:
    b=band(r[2:7]); shapes[b]=shapes.get(b,0)+1
shape=band(WIN); shape_pct=100*shapes[shape]/len(rows)
layer='A' if shape_pct>=5 else 'B' if shape_pct>=2 else 'C' if shape_pct>=.5 else 'D'

W=set(WIN)
def block_review(name,tickets,labels):
    rs=[]; union=set(); pairs=set(); triples=set()
    for label,t in zip(labels,tickets):
        st=set(t); m=sorted(W&st); union|=st
        pairs|=set(itertools.combinations(m,2)); triples|=set(itertools.combinations(m,3))
        rs.append({'id':label,'numbers':t,'sum':sum(t),'main_match':len(m),'matched':m,'bonus_in_ticket':BONUS in st,
                   'grade':'4等' if len(m)==3 else '—'})
    wp=set(itertools.combinations(sorted(W),2)); wt=set(itertools.combinations(sorted(W),3))
    return {'name':name,'tickets':rs,'ticket_count':len(tickets),'investment_yen':len(tickets)*200,
            'best_match':max(x['main_match'] for x in rs),'winning_ticket_count':sum(x['main_match']==3 for x in rs),
            'winner_number_recall':len(union&W),'winner_numbers_found':sorted(union&W),'winner_numbers_missing':sorted(W-union),
            'winner_pair_recall':len(pairs&wp),'winner_triple_recall':len(triples&wt),'union_size':len(union)}
rev=block_review('合計遷移反映10口',revised,[str(i) for i in range(1,11)])
com=block_review('Committee再結成10口',committee,[f'C{i}' for i in range(1,11)])
allrev=block_review('購入20口',revised+committee,[str(i) for i in range(1,11)]+[f'C{i}' for i in range(1,11)])

sums=[sum(r[2:7]) for r in rows]
changes=[sums[i]-sums[i-1] for i in range(1,len(sums))]
# previous forecast was made using data through 1402: current-like 107..117 -> 74..84, drop -40..-25
prior_events=[]
for i in range(1,len(sums)-2): # stop before the just-observed 1401->1402->1403 triple
    a,b,c=sums[i-1],sums[i],sums[i+1]
    if 107<=a<=117 and 74<=b<=84 and -40<=b-a<=-25:
        prior_events.append((a,b,c,c-b))
prior_rise=sum(1 for *_,d in prior_events if d>0)
prior_fall=sum(1 for *_,d in prior_events if d<0)
prior_pm10=sum(1 for *_,d in prior_events if abs(d)<=10)

review={
 'draw':DRAW,'date':DATE,'winner':WIN,'bonus':BONUS,'sum':sum(WIN),'odd_even':'3:2','high_low':'3:2',
 'band':shape,'layer':layer,'long_term_shape_pct':shape_pct,'consecutive':False,
 'prev_draw':1402,'prev_numbers':[1,4,20,25,29],'prev_sum':79,'prev_overlap':0,'prev_bonus':22,'prev_bonus_excluded':True,
 'sum_sequence':[112,79,69],'sum_changes':[-33,-10],
 'prior_sum_transition_test':{
   'condition':'107<=prev2<=117, 74<=prev1<=84, drop -40..-25',
   'historical_n_before_1403':len(prior_events),'rise_count':prior_rise,'rise_pct':100*prior_rise/len(prior_events),
   'fall_count':prior_fall,'fall_pct':100*prior_fall/len(prior_events),'within_pm10_count':prior_pm10,
   'actual_next_sum':69,'actual_change':-10,'forecast_direction_hit':False,'actual_within_pm10':True,
 },
 'blocks':{'revised10':rev,'committee10':com,'all20':allrev},
 'key_findings':[
   '合計遷移反映10口の#3（11 14 17 21 27）が本数字11・14・17の3個一致で4等。',
   '購入20口のWinner Number Recallは4/5で、8は全20口に一度も入らなかった。',
   'Committee再結成10口はWinner Number Recall 2/5、best 1/5で、この回ではRescueとして機能しなかった。',
   '112→79後の上昇優勢仮説に対し、実際は79→69（-10）で下落継続。合計値はハード誘導ではなくsoft priorに留める。',
   '合計遷移反映10口は当選合計69をレンジ外にした一方、3数字core 11-14-17を1口へ集約できた。合計予測とAssembly品質は別評価が必要。'
 ]
}
(ROOT/'data/miniloto-review-1403.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')

# 2) VERSION
(ROOT/'VERSION').write_text(VERSION+'\n',encoding='utf-8')

# helpers
def read(name): return (ROOT/name).read_text(encoding='utf-8')
def write(name,text): (ROOT/name).write_text(text,encoding='utf-8')
def cache(text):
    text=re.sub(r'0\.12\.[0-9]+-1402',VERSION+'-1403',text)
    text=text.replace('0.12.2-1402',VERSION+'-1403').replace('0.12.1-1402',VERSION+'-1403')
    text=text.replace('20260901-1402',STAMP).replace('v=1402','v=1403')
    return text

# 3) main Mini Loto menu - targeted refresh
name='miniloto.html'; t=read(name)
t=cache(t)
t=t.replace('App v0.12.2','App '+VERSION).replace('App v0.12.1','App '+VERSION)
t=re.sub(r'<div class="sub">.*?</div>','<div class="sub">第1403回結果・購入20口・合計値遷移の事後検証まで反映</div>',t,count=1)
t=t.replace('第1402回まで反映。最新結果を含めて分析条件を確認','第1403回まで反映。最新結果 08・11・14・17・19（B24）を確認')
t=t.replace('第1402回の再構築10口レビュー、Committee、A-Stat、Assembly改善方針を確認','第1403回の購入20口レビュー、4等1口、合計遷移・Assemblyの改善方針を確認')
t=t.replace('第1402回は再構築10口のみ実購入。最大2個一致・当選なし、Union Recall 4/5を記録','第1403回は20口購入。#3が3個一致で4等、20口Union Recall 4/5（8欠落）')
write(name,t)

# 4) history page
name='miniloto-history-v4.html'; t=read(name); t=cache(t)
t=t.replace('第1回〜第1402回｜当選番号＋分析条件','第1回〜第1403回｜当選番号＋分析条件')
t=t.replace('data/miniloto-chunk-8.js?v=1402','data/miniloto-chunk-8.js?v=1403')
t=t.replace('DATA.length!==1402','DATA.length!==1403')
write(name,t)

# 5) trends page; dynamic stats use chunk data, add retrospective card
name='miniloto-trends.html'; t=read(name); t=cache(t)
t=t.replace('App v0.12.1','App '+VERSION).replace('第1402回まで反映。表示期間の抽選結果を動的集計。','第1403回まで反映。表示期間の抽選結果を動的集計。')
t=t.replace('全1402回の長期頻度','全1403回の長期頻度')
t=t.replace('data/miniloto-chunk-8.js?v=0.12.1-1402',f'data/miniloto-chunk-8.js?v={VERSION}-1403')
marker='<!-- SUM_1403_REVIEW -->'
if marker not in t:
    card=f'''{marker}<div class="card" style="margin-bottom:10px"><div class="label">直近の合計値遷移</div><div class="n">112 → 79 → 69</div><div class="small">変化量 -33 → -10。事前の類似23例では次回上昇78.3%だったが、第1403回は下落継続。合計値遷移はsoft priorとして扱い、購入レンジのハード固定には使わない。</div></div>'''
    t=t.replace('<div class="periods"',card+'<div class="periods"',1)
write(name,t)

# 6) records page
name='miniloto-records.html'; t=read(name); t=cache(t)
t=t.replace('累積投資額</div><div class="n">¥7,600','累積投資額</div><div class="n">¥11,600')
t=t.replace('累積当選額</div><div class="n">¥1,100','累積当選額</div><div class="n">¥1,100＋4等確定')
t=t.replace('登録購入口</div><div class="n">38口','登録購入口</div><div class="n">58口')
t=t.replace('回収率</div><div class="n">14.5%','回収率</div><div class="n">当せん金反映待ち')
t=t.replace('<details class="draw" open><summary><div class="sumleft"><div class="drawtitle">第1402回','<details class="draw"><summary><div class="sumleft"><div class="drawtitle">第1402回',1)
marker='<!-- DRAW1403_START -->'
if marker not in t:
    def ticket_rows(block,prefix):
        out=[]
        for i,x in enumerate(block,1):
            m=sorted(W&set(x)); bo=BONUS in x; grade='4等' if len(m)==3 else '—'
            out.append(f'<tr><td class="tag">{prefix}</td><td>{prefix[0]}{i if prefix.startswith("Committee") else i}</td><td>{" ".join(f"{n:02d}" for n in x)}</td><td class="{"hit" if m else ""}">{len(m)}</td><td class="{"hit" if m else ""}">{" ".join(f"{n:02d}" for n in m) if m else "—"}</td><td>{"24" if bo else "—"}</td><td class="{"hit" if grade!="—" else ""}">{grade}</td></tr>')
        return ''.join(out)
    new=f'''{marker}<details class="draw" open><summary><div class="sumleft"><div class="drawtitle">第1403回｜2026/09/08</div><div class="drawmeta">20口購入・¥4,000・最大3個一致</div></div><div class="sumright"><span class="badge win">4等 1口</span><span class="chev">⌄</span></div></summary><div class="drawbody"><div class="card" style="margin-top:12px"><div class="label">当選番号</div><div class="result"><span class="ball">08</span><span class="ball">11</span><span class="ball">14</span><span class="ball">17</span><span class="ball">19</span><span style="font-size:10px;color:var(--muted)">B</span><span class="ball bonus">24</span></div><div class="note">合計値遷移反映10口＋Committee再結成10口＝20口、4,000円。合計値遷移反映#3「11 14 17 21 27」が本数字11・14・17の3個一致で4等。当せん金は公式額確認後に累積へ確定反映。</div><div class="diag"><span class="good">購入20口 Winner Number Recall：4/5</span>（11・14・17・19、<span class="bad">08欠落</span>）<br>Best ticket：3/5 / Winner Pair Recall：4/10 / Winner Triple Recall：1/10<br>合計遷移反映10口：Recall 4/5、Best 3/5、4等1口<br>Committee再結成10口：Recall 2/5、Best 1/5<br><span class="warn">112→79後は類似23例で上昇78.3%だったが、実際は79→69（-10）。合計値を90前後へ寄せる考えは外れた。</span><br><span class="good">一方で合計遷移反映時に新規導入した11が14・17と同一口に集まり、3数字coreのAssemblyは成功した。</span></div></div><div class="tablewrap"><table><thead><tr><th>区分</th><th>口</th><th>購入数字</th><th>本数字一致</th><th>一致数字</th><th>BO</th><th>等級</th></tr></thead><tbody>{ticket_rows(revised,'合計遷移')}{ticket_rows(committee,'Committee')}</tbody></table></div></div></details><!-- DRAW1403_END -->'''
    t=t.replace('<div class="history">','<div class="history">'+new,1)
write(name,t)

# 7) development page: prepend 1403 review, preserve old research
name='miniloto-development.html'; t=read(name); t=cache(t)
marker='<!-- DRAW1403_REVIEW_START -->'
if marker not in t:
    section=f'''{marker}<div class="section">第1403回：購入20口の振り返り</div><div class="grid"><div class="kpi"><div class="label">当選本数字</div><div class="n">08・11・14・17・19</div><div class="delta">B24 / 合計69 / 帯{shape} / {layer}層</div></div><div class="kpi"><div class="label">購入成績</div><div class="n">4等 1口</div><div class="delta">#3 11・14・17 が3個一致</div></div><div class="kpi"><div class="label">20口 Winner Recall</div><div class="n">4 / 5</div><div class="red">08を全20口で欠落</div></div><div class="kpi"><div class="label">N/P/T Recall</div><div class="n">4/5・4/10・1/10</div></div></div><div class="note ok"><b>良かった点：</b>合計値遷移反映10口の#3「11 14 17 21 27」が3個一致。元の80前後候補では最大2個一致だったため、候補再構成で11を導入し、11-14-17 coreを同一口に集約できたこと自体は改善。</div><div class="note danger"><b>失敗点：</b>当選合計は69。112→79後の「反発して90前後」シナリオは外れ、調整後10口の合計80〜102、Committee再結成10口の82〜109はいずれも1等合計をカバーしなかった。さらに全20口でも08が一度も入らず、Exact成立は入口段階で不可能だった。</div><div class="note warn"><b>モデル判断：</b>合計値遷移はsoft priorに限定し、ポートフォリオ全体を同方向へ動かさない。Committee再結成10口はこの回でRecall 2/5・Best 1/5だったため、追加10口を常設する根拠はない。今後はMain/専門枝ごとに独立な数字Coverageを監査し、少なくとも1〜2口は低合計側を残す。</div><!-- DRAW1403_REVIEW_END -->'''
    # insert immediately after first sub block
    m=re.search(r'(<div class="sub">.*?</div>)',t,re.S)
    if m: t=t[:m.end()]+section+t[m.end():]
write(name,t)

# 8) river and score page: cache/nav version only
for name in ['miniloto-river.html','miniloto-score-traces.html']:
    p=ROOT/name
    if p.exists():
        t=cache(p.read_text(encoding='utf-8')).replace('App v0.12.1','App '+VERSION).replace('App v0.12.2','App '+VERSION)
        p.write_text(t,encoding='utf-8')

# 9) root index cache/version if present
p=ROOT/'index.html'
if p.exists():
    t=cache(p.read_text(encoding='utf-8')).replace('App v0.12.2','App '+VERSION).replace('App v0.12.1','App '+VERSION)
    p.write_text(t,encoding='utf-8')

print(json.dumps(review,ensure_ascii=False,indent=2))
