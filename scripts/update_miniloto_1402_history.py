from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

# 1) canonical chunk used by history/trends pages
p=ROOT/'data'/'miniloto-chunk-8.js'
s=p.read_text(encoding='utf-8')
row='[1402,"2026/09/01",1,4,20,25,29,22,7]'
if row not in s:
    m=re.search(r'push\((\[.*\])\);?$',s)
    if not m:
        raise SystemExit('unexpected miniloto-chunk-8.js format')
    arr=m.group(1)
    if '1402' not in arr:
        arr=arr[:-1]+','+row+']'
    s=s[:m.start(1)]+arr+s[m.end(1):]
    p.write_text(s,encoding='utf-8')

# 2) current history page
p=ROOT/'miniloto-history-v4.html'
s=p.read_text(encoding='utf-8')
s=s.replace('第1回〜第1401回｜当選番号＋分析条件','第1回〜第1402回｜当選番号＋分析条件')
s=s.replace('DATA.length!==1401','DATA.length!==1402')
s=s.replace('miniloto-chunk-8.js?v=1930','miniloto-chunk-8.js?v=1402')
s=s.replace('?v=20260825-1930','?v=20260901-1402')
p.write_text(s,encoding='utf-8')

# 3) trends page
p=ROOT/'miniloto-trends.html'
s=p.read_text(encoding='utf-8')
s=s.replace('第1401回まで反映。表示期間の抽選結果を動的集計。','第1402回まで反映。表示期間の抽選結果を動的集計。')
s=s.replace('全1401回の長期頻度','全1402回の長期頻度')
s=s.replace('miniloto-chunk-8.js?v=0.12.1','miniloto-chunk-8.js?v=0.12.1-1402')
p.write_text(s,encoding='utf-8')

# 4) menu wording: no longer "latest result reflected elsewhere first"
p=ROOT/'miniloto.html'
s=p.read_text(encoding='utf-8')
s=s.replace('履歴データ表示。第1402回の最新結果はトップ・購入記録・開発診断へ先行反映','履歴データも第1402回まで反映。最新結果を含めて分析条件を確認')
p.write_text(s,encoding='utf-8')

# validation
chunk=(ROOT/'data'/'miniloto-chunk-8.js').read_text(encoding='utf-8')
assert '[1401,"2026/08/25",15,17,23,26,31,22,17]' in chunk
assert row in chunk
hist=(ROOT/'miniloto-history-v4.html').read_text(encoding='utf-8')
assert '第1回〜第1402回' in hist and 'DATA.length!==1402' in hist
tr=(ROOT/'miniloto-trends.html').read_text(encoding='utf-8')
assert '第1402回まで反映' in tr and '全1402回の長期頻度' in tr
print('updated Mini Loto history/trends through draw 1402')
