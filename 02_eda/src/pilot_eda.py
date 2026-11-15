#!/usr/bin/env python3
"""Analyze DNS monitor cards, prepare candidate labels and split identifiers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import platform
import importlib.metadata

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FIELDS = ('url', 'code', 'title', 'category', 'description', 'specs')

def number(s):
    m = re.search(r"\d+(?:[.,]\d+)?", str(s))
    return m.group().replace(",", ".") if m else ""

def resolution(s):
    m = re.findall(r"\d+", str(s))
    if len(m) >= 2:
        w, h = m[:2]
        return f"{w}x{h}" if int(w) >= 640 else None
    return None

def contrast(s):
    s = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", str(s).lower())
    m = re.search(r"\d+(?:[.,]\d+)?", s)
    if not m: return None
    v = float(m.group().replace(",", ".")) * (10**6 if re.search(r"m|м", s) else 1)
    return str(int(v)) if v >= 100 else None

ATTRS = {
    "диагональ":     ("Диагональ экрана (дюйм)", lambda s: f"{float(number(s)):g}" if number(s) else None, r'(\d+(?:[.,]\d+)?)\s*-?\s*(?:"|”|″|дюйм)'),
    "разрешение":    ("Максимальное разрешение", resolution, r"(\d{3,4}\s*[xх×*]\s*\d{3,4})"),
    "частота":       ("Максимальная частота обновления экрана", lambda s: number(s) or None, r"(\d{2,3})\s*(?:гц|герц)"),
    "тип матрицы":   ("Тип матрицы", lambda s: str(s).upper().replace("QD-", ""), r"\b(ips|va|(?:qd-)?oled|tn)\b"),
    "яркость":       ("Яркость", lambda s: number(s) or None, r"(\d{3,4})\s*(?:кд|нит|cd)"),
    "контрастность": ("Контрастность", contrast, r"((?:\d{1,3}(?:\s\d{3})+|\d+(?:[.,]\d+)?)\s*(?:m|млн)?)\s*(?::|к|/)\s*1\b"),
    "изогнутость":   ("Изогнутый экран", lambda s: str(s).lower(), r"(изогнут|curved|плоск)"),
}
CURVED = {"изогнут": "да", "curved": "да", "плоск": "нет"}

def normalize(text):
    return re.sub(r'\s+', ' ', str(text or '').replace('\\n', '\n')).strip()

def values_in_text(attr, text):
    _, norm, pattern = ATTRS[attr]
    found = set()
    for m in re.findall(pattern, text):
        v = CURVED.get(m) if attr == "изогнутость" else norm(m)
        if v: found.add(v)
    return found

def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return centre - half, centre + half


def analyze(path, n_pilot=100, head=None, val_share=0.4, seed=42):
    raw = Path(path).read_bytes()
    cards = json.loads(raw)
    if not isinstance(cards, list) or any(not isinstance(c, dict) for c in cards):
        raise ValueError('Expected a JSON array of card objects')

    monitors = [c for c in cards if isinstance(c.get('specs'), dict) and c['specs'].get('Тип') == 'монитор']
    if not monitors:
        raise ValueError('No monitor records found')
    for index, card in enumerate(monitors):
        for field in FIELDS:
            value = card.get(field)
            if not value or (field != 'specs' and not isinstance(value, str)):
                raise ValueError(f'Monitor record {index}: missing or invalid {field}')
    urls = [c['url'] for c in monitors]
    if len(urls) != len(set(urls)):
        raise ValueError('Duplicate monitor URLs; resolve source records before analysis')
    if n_pilot < 1 or not 0 < val_share < 1:
        raise ValueError('Expected n_pilot >= 1 and 0 < val_share < 1')
    df = pd.DataFrame(monitors)
    if head:
        df = df.head(head).copy()
    df['brand'] = df['category'].astype(str).str.split(' / ').str[-1]
    df['description_length'] = df['description'].fillna('').str.len()
    df['spec_count'] = df['specs'].apply(len)

    descriptions = df['description']
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(descriptions)
    sim = cosine_similarity(tfidf)
    dups = [(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if sim[i, j] >= 0.97]
    drop_indices = list({j for i, j in dups})
    df_clean = df.drop(index=drop_indices).reset_index(drop=True)

    if len(df_clean) <= n_pilot + 1:
        raise ValueError('Not enough cards for pilot, real_val and real_test')

    rows = []
    for _, c in df_clean.iterrows():
        text = c['description'].lower().replace("ё", "е")
        for attr, (key, norm, _) in ATTRS.items():
            if key not in c["specs"]:
                continue
            spec = norm(c["specs"][key])
            if not spec:
                continue
            found = values_in_text(attr, text)
            status = "в тексте" if spec in found else "другое значение" if found else "не упомянут"
            rows.append({
                "url": c["url"], "title": c["title"], "атрибут": attr,
                "в таблице": spec, "в тексте": ", ".join(sorted(found)),
                "статус": status, "эталон": spec if status == "в тексте" else None
            })
    pairs = pd.DataFrame(rows)

    pilot = df_clean.iloc[:n_pilot]
    rest = df_clean.iloc[n_pilot:].sample(frac=1, random_state=seed)
    n_val = round(len(rest) * val_share)

    manifest = {
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'split_rule': f'TF-IDF dedup -> first {n_pilot} pilot -> rest {val_share}/{1-val_share} val/test (seed {seed})',
        'pilot': pilot["url"].tolist(),
        'real_val': rest["url"].iloc[:n_val].tolist(),
        'real_test': rest["url"].iloc[n_val:].tolist()
    }

    lengths = df_clean['description_length']
    brands = Counter(df_clean['brand'])
    cov = pairs.pivot_table(index="атрибут", columns="статус", values="url", aggfunc="count", fill_value=0)
    for status in ["в тексте", "другое значение", "не упомянут"]:
        if status not in cov:
            cov[status] = 0
    cov = cov[["в тексте", "другое значение", "не упомянут"]]
    cov["всего"] = cov.sum(axis=1)
    cov["доля в тексте"] = (cov["в тексте"] / cov["всего"]).round(2)
    cov["±"] = [round((wilson(row["в тексте"], row["всего"])[1] - wilson(row["в тексте"], row["всего"])[0]) / 2, 2) for _, row in cov.iterrows()]

    summary = {
        'input_sha256': manifest['source_sha256'],
        'input_records': len(cards),
        'monitor_records': len(monitors),
        'analyzed_records': len(df),
        'removed_duplicates_tfidf': len(drop_indices),
        'clean_records': len(df_clean),
        'removed_non_monitors': len(cards) - len(monitors),
        'duplicate_pairs_tfidf': len(dups),
        'missing_reference_specs': {
            attr: int(sum(key not in c['specs'] for _, c in df_clean.iterrows()))
            for attr, (key, _, _) in ATTRS.items()
        },
        'environment': {
            'python': platform.python_version(),
            **{name: importlib.metadata.version(name)
               for name in ('pandas', 'matplotlib', 'scikit-learn')},
        },
        'split_parameters': {'n_pilot': n_pilot, 'val_share': val_share,
                             'seed': seed, 'similarity_threshold': 0.97},
        'description_length_characters': {
            'min': int(lengths.min()) if not lengths.empty else 0,
            'max': int(lengths.max()) if not lengths.empty else 0,
            'mean': round(lengths.mean(), 2) if not lengths.empty else 0,
            'median': float(lengths.median()) if not lengths.empty else 0
        },
        'brands': dict(sorted(brands.items())),
        'split_counts': {k: len(v) for k, v in manifest.items() if isinstance(v, list)},
        'coverage': cov.reset_index().to_dict(orient='records'),
        'conflict_card_count': int(pairs.loc[pairs['статус'] == 'другое значение', 'url'].nunique()),
    }

    return summary, manifest, pairs, df, df_clean

def plot_coverage(pairs, out_path):
    cov = pairs.pivot_table(index="атрибут", columns="статус", values="url", aggfunc="count", fill_value=0)
    for s in ["в тексте", "другое значение", "не упомянут"]:
        if s not in cov: cov[s] = 0
    cov = cov[["в тексте", "другое значение", "не упомянут"]]

    ax = cov.sort_values("в тексте").plot.barh(stacked=True, figsize=(8, 3.5), color=["#4c9a6a", "#d9a441", "#b8bec7"])
    ax.set_xlabel("число карточек")
    ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path,
                   help='Public aggregate results only')
    p.add_argument('--private-dir', required=True, type=Path,
                   help='Local directory for candidate labels and descriptions')
    a = p.parse_args()
    if a.out_dir.resolve() == a.private_dir.resolve():
        p.error('--private-dir must differ from --out-dir')

    summary, manifest, pairs, _, df_clean = analyze(a.input)

    gold_dict = {
        u: {r['атрибут']: (r['эталон'] if pd.notna(r['эталон']) else None)
            for _, r in group.iterrows()}
        for u, group in pairs.groupby('url')
    }
    # Missing reference fields stay absent, as in the research notebook.
    cards_gold = [
        {'url': r['url'], 'title': r['title'], 'description': normalize(r['description']),
         'gold': gold_dict.get(r['url'], {})}
        for _, r in df_clean.iterrows()
    ]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.private_dir.mkdir(parents=True, exist_ok=True)
    pairs.drop(columns=['title']).to_csv(
        a.private_dir / 'gold_pairs.csv', index=False, encoding='utf-8')
    (a.private_dir / f'{a.input.stem}_gold.json').write_text(
        json.dumps(cards_gold, ensure_ascii=False, indent=1), encoding='utf-8')
    (a.out_dir / 'eda_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (a.out_dir / 'split_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plot_coverage(pairs, a.out_dir / 'coverage_stacked.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
