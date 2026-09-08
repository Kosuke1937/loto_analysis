from pathlib import Path
p=Path('miniloto-history-v4.html')
s=p.read_text(encoding='utf-8')
old='<input id="search" class="search" placeholder="回号・数字・帯構成で検索"><select id="order"><option value="desc">新しい順</option><option value="asc">古い順</option></select>'
new='<input id="search" class="search" placeholder="回号・数字・帯構成・連番で検索"><select id="consFilter"><option value="all">連番：すべて</option><option value="yes">連番あり</option><option value="no">連番なし</option></select><select id="order"><option value="desc">新しい順</option><option value="asc">古い順</option></select>'
if old not in s: raise SystemExit('controls marker not found')
s=s.replace(old,new,1)
old='<th>回</th><th>日付</th><th>本数字</th><th>B</th><th>合計</th><th>奇偶比</th><th>高低比</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>連番</th><th>20番台</th>'
new='<th>回</th><th>日付</th><th>本数字</th><th>連番有無</th><th>連番内容</th><th>B</th><th>合計</th><th>奇偶比</th><th>高低比</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>20番台</th>'
if old not in s: raise SystemExit('header marker not found')
s=s.replace(old,new,1)
old="return{r,a,s,f,l,sum:a.reduce((x,y)=>x+y,0),oe:odd+':'+(5-odd),hl:lo+':'+(5-lo),co:cons(a),tw,th,ov,pass,crowd:crowd(r[8])}});"
new="const co=cons(a);return{r,a,s,f,l,sum:a.reduce((x,y)=>x+y,0),oe:odd+':'+(5-odd),hl:lo+':'+(5-lo),co,hasCons:co!=='—',tw,th,ov,pass,crowd:crowd(r[8])}});"
if old not in s: raise SystemExit('row object marker not found')
s=s.replace(old,new,1)
old="if(range!=='all')rows=rows.slice(-Number(range));const sc=$('search').value.trim();if(sc)rows=rows.filter(x=>[x.r[0],x.r[1],x.a.join(' '),x.s,x.l,x.sum].join(' ').includes(sc));"
new="if(range!=='all')rows=rows.slice(-Number(range));const cf=$('consFilter').value;if(cf==='yes')rows=rows.filter(x=>x.hasCons);if(cf==='no')rows=rows.filter(x=>!x.hasCons);const sc=$('search').value.trim();if(sc)rows=rows.filter(x=>[x.r[0],x.r[1],x.a.join(' '),x.s,x.l,x.sum,x.co,x.hasCons?'連番あり':'連番なし'].join(' ').includes(sc));"
if old not in s: raise SystemExit('filter marker not found')
s=s.replace(old,new,1)
old='<td class="nums">${x.a.map(pad).join(\' \')}</td><td class="bonus">${pad(x.r[7])}</td><td>${x.sum}</td><td>${x.oe}</td><td>${x.hl}</td><td>${x.s}</td><td class="${x.l}">${x.l}</td><td>${x.f.toFixed(2)}%</td><td>${x.co}</td><td>${x.tw}</td>'
new='<td class="nums">${x.a.map(pad).join(\' \')}</td><td class="${x.hasCons?\'ok\':\'\'}">${x.hasCons?\'あり\':\'なし\'}</td><td>${x.co}</td><td class="bonus">${pad(x.r[7])}</td><td>${x.sum}</td><td>${x.oe}</td><td>${x.hl}</td><td>${x.s}</td><td class="${x.l}">${x.l}</td><td>${x.f.toFixed(2)}%</td><td>${x.tw}</td>'
if old not in s: raise SystemExit('render marker not found')
s=s.replace(old,new,1)
old="$('search').oninput=render;$('order').onchange=render;"
new="$('search').oninput=render;$('consFilter').onchange=render;$('order').onchange=render;"
if old not in s: raise SystemExit('event marker not found')
s=s.replace(old,new,1)
old='高低比＝1〜16 / 17〜31。帯構成＝1〜9 / 10〜19 / 20〜29 / 30〜31。前回B除外＝前回ボーナス数字を今回本数字に含めない条件の通過可否。'
new='高低比＝1〜16 / 17〜31。帯構成＝1〜9 / 10〜19 / 20〜29 / 30〜31。連番は「あり/なし」と実際の連番内容を本数字の直後に表示。前回B除外＝前回ボーナス数字を今回本数字に含めない条件の通過可否。'
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('updated',p)
