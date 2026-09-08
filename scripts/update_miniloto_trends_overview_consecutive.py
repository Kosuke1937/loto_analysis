from pathlib import Path
p=Path('miniloto-trends.html')
s=p.read_text(encoding='utf-8')
old='<thead><tr><th>回</th><th>本数字</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>合計</th></tr></thead><tbody id="recentBody"></tbody>'
new='<thead><tr><th>回</th><th>本数字</th><th>連番</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>合計</th></tr></thead><tbody id="recentBody"></tbody>'
if old not in s:
    raise SystemExit('overview table header marker not found')
s=s.replace(old,new,1)
old2="if(d)$('selectedBox').innerHTML=`<b>第${d.r[0]}回</b>　${d.a.map(pad).join('・')}<br>合計 ${d.sum}｜帯構成 ${d.s}｜${d.l}層｜長期頻度 ${d.f.toFixed(2)}%${d.cp.length?'<br>連番 '+d.cp.join(' / '):''}`"
new2="if(d)$('selectedBox').innerHTML=`<b>第${d.r[0]}回</b>　${d.a.map(pad).join('・')}<br>合計 ${d.sum}｜帯構成 ${d.s}｜${d.l}層｜長期頻度 ${d.f.toFixed(2)}%｜連番 ${d.cp.length?'あり':'なし'}${d.cp.length?'<br>連番内容 '+d.cp.join(' / '):''}`"
if old2 not in s:
    raise SystemExit('selectedBox marker not found')
s=s.replace(old2,new2,1)
old3="$('recentBody').innerHTML=[...R].reverse().map(x=>`<tr><td>第${x.r[0]}</td><td>${x.a.map(pad).join(' ')}</td><td>${x.s}</td><td style=\"color:${COL[x.l]}\">${x.l}</td><td>${x.f.toFixed(2)}%</td><td>${x.sum}</td></tr>`).join('');"
new3="$('recentBody').innerHTML=[...R].reverse().map(x=>`<tr><td>第${x.r[0]}</td><td>${x.a.map(pad).join(' ')}</td><td class=\"${x.cp.length?'consecYes':'consecNo'}\">${x.cp.length?'あり '+x.cp.join(' / '):'なし'}</td><td>${x.s}</td><td style=\"color:${COL[x.l]}\">${x.l}</td><td>${x.f.toFixed(2)}%</td><td>${x.sum}</td></tr>`).join('');"
if old3 not in s:
    raise SystemExit('recentBody marker not found')
s=s.replace(old3,new3,1)
# Add small styling for visibility.
css='.consecYes{color:#7ce0a7;font-weight:800}.consecNo{color:#8f99aa}'
if css not in s:
    s=s.replace('</style>',css+'</style>',1)
p.write_text(s,encoding='utf-8')
print('updated',p)
