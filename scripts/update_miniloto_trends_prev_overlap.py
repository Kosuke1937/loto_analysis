from pathlib import Path
p=Path('miniloto-trends.html')
s=p.read_text(encoding='utf-8')

# Add table column immediately after 本数字.
old='<tr><th>回</th><th>本数字</th><th>連番</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>合計</th></tr>'
new='<tr><th>回</th><th>本数字</th><th>前回数字</th><th>連番</th><th>帯構成</th><th>層</th><th>長期頻度</th><th>合計</th></tr>'
if old not in s:
    raise SystemExit('table header pattern not found')
s=s.replace(old,new,1)

# Add helper that always references the full history, not only the displayed period.
old2="function cons(a){let p=[];for(let i=1;i<a.length;i++)if(a[i]===a[i-1]+1)p.push(a[i-1]+'-'+a[i]);return p}function med(a)"
new2="function cons(a){let p=[];for(let i=1;i<a.length;i++)if(a[i]===a[i-1]+1)p.push(a[i-1]+'-'+a[i]);return p}function prevCount(r){let i=DATA.findIndex(x=>x[0]===r[0]);if(i<=0)return null;let p=new Set(DATA[i-1].slice(2,7));return r.slice(2,7).filter(n=>p.has(n)).length}function med(a)"
if old2 not in s:
    raise SystemExit('function insertion pattern not found')
s=s.replace(old2,new2,1)

# Store previous-draw overlap count in row objects.
old3="return{r,a,s,f,l:layer(f),sum:a.reduce((x,y)=>x+y,0),cp:cons(a)}})"
new3="return{r,a,s,f,l:layer(f),sum:a.reduce((x,y)=>x+y,0),cp:cons(a),prev:prevCount(r)}})"
if old3 not in s:
    raise SystemExit('rows pattern not found')
s=s.replace(old3,new3,1)

# Add it to the selected draw detail too.
old4="｜長期頻度 ${d.f.toFixed(2)}%｜連番 ${d.cp.length?'あり':'なし'}"
new4="｜長期頻度 ${d.f.toFixed(2)}%｜前回数字 ${d.prev==null?'—':d.prev+'個'}｜連番 ${d.cp.length?'あり':'なし'}"
if old4 not in s:
    raise SystemExit('selected box pattern not found')
s=s.replace(old4,new4,1)

# Add table body cell immediately after the main numbers.
old5="<td>${x.a.map(pad).join(' ')}</td><td class=\"${x.cp.length?'consecYes':'consecNo'}\">${x.cp.length?'あり '+x.cp.join(' / '):'なし'}</td>"
new5="<td>${x.a.map(pad).join(' ')}</td><td>${x.prev==null?'—':x.prev+'個'}</td><td class=\"${x.cp.length?'consecYes':'consecNo'}\">${x.cp.length?'あり '+x.cp.join(' / '):'なし'}</td>"
if old5 not in s:
    raise SystemExit('recentBody pattern not found')
s=s.replace(old5,new5,1)

p.write_text(s,encoding='utf-8')
print('updated',p)
