#!/usr/bin/env python3
"""Audit a pilot of at most 50 DNS monitor cards. No inference or raw-text export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

FIELDS = ('url', 'code', 'title', 'category', 'description', 'specs')
ATTRS = {'диагональ': 'Диагональ экрана (дюйм)', 'разрешение': 'Максимальное разрешение',
         'частота': 'Максимальная частота обновления экрана', 'тип матрицы': 'Тип матрицы',
         'яркость': 'Яркость', 'контрастность': 'Контрастность', 'изогнутость': 'Изогнутый экран'}

def normalize(text):
    return re.sub(r'\s+', ' ', str(text or '').replace('\\n', '\n')).strip()

def analyze(path):
    raw = Path(path).read_bytes()
    cards = json.loads(raw)
    if not isinstance(cards, list) or any(not isinstance(c, dict) for c in cards):
        raise ValueError('Expected a JSON array of card objects')
    monitors = [c for c in cards if isinstance(c.get('specs'), dict)
                and c['specs'].get('Тип') == 'монитор']
    pilot = monitors[:50]
    if len(pilot) != 50:
        raise ValueError(f'Expected 50 monitor cards, found {len(pilot)}')
    missing = {k: sum(not c.get(k) for c in pilot) for k in FIELDS}
    duplicate_counts = {}
    for k in ('url', 'code', 'title', 'description'):
        vals = [normalize(c.get(k)) for c in pilot if normalize(c.get(k))]
        duplicate_counts[k] = len(vals) - len(set(vals))
    clean, seen_url, seen_code, seen_text = [], set(), set(), set()
    removed_empty = removed_duplicates = changed_descriptions = 0
    for c in pilot:
        if any(not c.get(k) for k in ('url', 'code', 'title', 'description', 'specs')):
            removed_empty += 1
            continue
        text = normalize(c['description'])
        if c['url'] in seen_url or c['code'] in seen_code or text in seen_text:
            removed_duplicates += 1
            continue
        seen_url.add(c['url']); seen_code.add(c['code']); seen_text.add(text)
        changed_descriptions += text != c['description']
        clean.append({**c, 'description': text})
    if len(clean) != 50:
        raise ValueError(f'After cleaning {len(clean)} cards remain; review records before fixing a 30/20 split')
    lengths = [len(c['description']) for c in clean]
    brands = Counter(normalize(c.get('category')).split(' / ')[-1] for c in clean)
    summary = {
        'input_sha256': hashlib.sha256(raw).hexdigest(),
        'input_records': len(cards), 'monitor_records': len(monitors),
        'pilot_records': len(pilot), 'clean_records': len(clean),
        'missing_before_cleaning': missing, 'duplicate_excess_before_cleaning': duplicate_counts,
        'removed_empty': removed_empty, 'removed_duplicates': removed_duplicates,
        'normalized_descriptions': changed_descriptions,
        'description_length_characters_after_normalization': {
            'min': min(lengths), 'max': max(lengths), 'mean': round(mean(lengths), 2), 'median': median(lengths)},
        'brands': dict(sorted(brands.items())),
        'missing_reference_specs': {attr: sum(not c['specs'].get(key) for c in clean) for attr, key in ATTRS.items()},
        'note': 'Specs are reference characteristics, not text-extraction gold labels. Missing mentions require manual annotation.',
        'split_counts': {'development': 30, 'test': 20},
    }
    manifest = {'source_sha256': summary['input_sha256'],
                'selection': 'first 50 records with specs.Тип == монитор; original source order',
                'split_rule': 'first 30 cleaned cards development; last 20 test; no shuffling',
                'development': [c['url'] for c in clean[:30]], 'test': [c['url'] for c in clean[30:]]}
    assert not set(manifest['development']) & set(manifest['test'])
    return summary, manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    a = p.parse_args()
    summary, manifest = analyze(a.input)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [('pilot_summary.json', summary), ('split_manifest.json', manifest)]:
        (a.out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
