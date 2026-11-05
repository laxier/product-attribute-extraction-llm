#!/usr/bin/env python3
"""Advanced EDA, gold label extraction, and split generation for DNS monitor cards."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

import pandas as pd
import numpy as np
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

def analyze(path, n_pilot=100, val_share=0.4, seed=42):
    raw = Path(path).read_bytes()
    cards = json.loads(raw)
    if not isinstance(cards, list) or any(not isinstance(c, dict) for c in cards):
        raise ValueError('Expected a JSON array of card objects')
        
    monitors = [c for c in cards if isinstance(c.get('specs'), dict) and c['specs'].get('Тип') == 'монитор']
    
    df = pd.DataFrame(monitors)
    df['norm_desc'] = df['description'].apply(normalize)
    
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(df['norm_desc'])
    sim = cosine_similarity(tfidf)
    dups = [(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if sim[i, j] >= 0.97]
    drop_indices = list({j for i, j in dups})
    df_clean = df.drop(index=drop_indices).reset_index(drop=True)
    
    rows = []
    for _, c in df_clean.iterrows():
        text = c['norm_desc'].lower().replace("ё", "е")
        for attr, (key, norm, _) in ATTRS.items():
            if key not in c["specs"]:
                continue
            spec = norm(c["specs"][key])
            if not spec: continue
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
    
    lengths = df_clean['norm_desc'].str.len()
    brands = Counter(df_clean['category'].apply(lambda x: str(x).split(' / ')[-1]))
    
    summary = {
        'input_sha256': manifest['source_sha256'],
        'input_records': len(cards),
        'monitor_records': len(monitors),
        'removed_duplicates_tfidf': len(drop_indices),
        'clean_records': len(df_clean),
        'description_length_characters': {
            'min': int(lengths.min()) if not lengths.empty else 0, 
            'max': int(lengths.max()) if not lengths.empty else 0, 
            'mean': round(lengths.mean(), 2) if not lengths.empty else 0, 
            'median': float(lengths.median()) if not lengths.empty else 0
        },
        'brands': dict(sorted(brands.items())),
        'split_counts': {k: len(v) for k, v in manifest.items() if isinstance(v, list)}
    }
    
    return summary, manifest, pairs, df_clean

def plot_coverage(pairs, out_path):
    cov = pairs.pivot_table(index="атрибут", columns="статус", values="url", aggfunc="count", fill_value=0)
    for s in ["в тексте", "другое значение", "не упомянут"]:
        if s not in cov: cov[s] = 0
    cov = cov[["в тексте", "другое значение", "не упомянут"]]
    
    ax = cov.sort_values("в тексте").plot.barh(stacked=True, figsize=(10, 4.5), color=["#4B9462", "#D6A445", "#BCC0C8"])
    ax.set_xlabel("число карточек", fontsize=11)
    ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--n-pilot', type=int, default=100)
    a = p.parse_args()
    
    a.out_dir.mkdir(parents=True, exist_ok=True)
    summary, manifest, pairs, df_clean = analyze(a.input, n_pilot=a.n_pilot)
    
    for name, obj in [('eda_summary.json', summary), ('split_manifest.json', manifest)]:
        (a.out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        
    pairs.drop(columns=["title"]).to_csv(a.out_dir / 'gold_pairs.csv', index=False, encoding="utf-8")
    gold_dict = {u: {r["атрибут"]: (r["эталон"] if pd.notna(r["эталон"]) else None) for _, r in g.iterrows()} for u, g in pairs.groupby("url")}
    cards_gold = [{"url": r["url"], "title": r["title"], "description": r["norm_desc"], "gold": gold_dict.get(r["url"], {})} for _, r in df_clean.iterrows()]
    
    gold_path = a.input.parent / f"{a.input.stem}_gold.json"
    gold_path.write_text(json.dumps(cards_gold, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    
    plot_coverage(pairs, a.out_dir / 'coverage_stacked.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
