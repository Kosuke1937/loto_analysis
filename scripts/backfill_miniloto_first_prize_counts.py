from pathlib import Path
import base64
import gzip
import json
import re

ROOT = Path(__file__).resolve().parents[1]

# KYO's MINILOTO / MINILOTO_ALL.csv の第1〜1399回「1等口数」を圧縮埋め込み。
# 1400回以降は既存データを保持し、未取得値を勝手に補完しない。
COUNTS_GZ_B64 = "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"


def load_counts():
    raw = gzip.decompress(base64.b64decode(COUNTS_GZ_B64)).decode('utf-8')
    out = {}
    for line in raw.splitlines():
        draw, count = line.split(',')
        out[int(draw)] = int(count)
    return out


def update_chunk(path: Path, counts):
    text = path.read_text(encoding='utf-8')
    m = re.search(r'push\((\[.*\])\);?\s*$', text, re.S)
    if not m:
        raise RuntimeError(f'chunk payload not found: {path}')
    rows = json.loads(m.group(1))
    changed = 0
    for row in rows:
        draw = int(row[0])
        if draw not in counts:
            continue
        while len(row) <= 8:
            row.append(None)
        if row[8] != counts[draw]:
            row[8] = counts[draw]
            changed += 1
    payload = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
    path.write_text(text[:m.start(1)] + payload + text[m.end(1):], encoding='utf-8')
    return changed


def bust_history_cache():
    path = ROOT / 'miniloto-history-v4.html'
    text = path.read_text(encoding='utf-8')
    version = '20260908-firstprize1'
    text = re.sub(
        r'(data/miniloto-chunk-\d+\.js)\?v=[^"<]+',
        lambda m: f'{m.group(1)}?v={version}',
        text,
    )
    path.write_text(text, encoding='utf-8')


def main():
    counts = load_counts()
    assert len(counts) == 1399
    total = 0
    for i in range(1, 9):
        total += update_chunk(ROOT / 'data' / f'miniloto-chunk-{i}.js', counts)
    bust_history_cache()
    print(f'backfilled first-prize counts: {len(counts)} draws; changed cells={total}')


if __name__ == '__main__':
    main()
