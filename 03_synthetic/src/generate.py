"""Генерация синтетических описаний мониторов через Claude.

Зависимости: pip install anthropic python-dotenv pandas
Ключ ANTHROPIC_API_KEY задаётся в окружении или .env в корне проекта.

Запуск из корня репозитория после подготовки EDA:
    python 03_synthetic/src/generate.py fewshot 20 1
    python 03_synthetic/src/generate.py train 1000 3
    python 03_synthetic/src/generate.py train_names 200 4

Стиль, частоты упоминаний и значения берутся только из pilot (100 карточек).
Эталон задаётся до генерации; тексты проверяются правилами EDA.
В train_names разрешение всегда названо словом, примерно в половине случаев
без цифр (эталон null). Результаты сохраняются локально в data/synthetic/.
Повторный запуск дополняет существующий JSONL до указанного числа записей;
при продолжении используйте те же seed, модель и входные данные.
Seed фиксирует параметры запросов, но не гарантирует одинаковые ответы API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[2]

STARTS = ["с названия бренда и модели", "с назначения монитора (игры, работа, дизайн)", "с самой заметной особенности",
          "с короткой фразы-призыва", "с описания внешнего вида и корпуса"]   # чтобы описания начинались по-разному
N_STYLE = 3                                       # сколько реальных описаний показать в промпте
ATTRS = ["диагональ", "разрешение", "частота", "тип матрицы", "яркость", "контрастность", "изогнутость"]

# --- проверка эталона: те же правила, что в 02_eda/src/pilot_eda.py ---
def number(s):
    return re.search(r"\d+(?:[.,]\d+)?", s).group().replace(",", ".")

def resolution(s):
    w, h = re.findall(r"\d+", s)[:2]
    return f"{w}x{h}" if int(w) >= 640 else None

def contrast(s):
    s = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", s.lower())
    v = float(number(s)) * (10**6 if re.search(r"m|м", s) else 1)
    return str(int(v)) if v >= 100 else None

NORM = {"диагональ": lambda s: f"{float(number(s)):g}", "разрешение": resolution, "частота": number,
        "тип матрицы": lambda s: s.upper().replace("QD-", ""), "яркость": number, "контрастность": contrast}
PATTERN = {"диагональ": r'(\d+(?:[.,]\d+)?)\s*-?\s*(?:"|”|″|дюйм)',
           "разрешение": r"(\d{3,4}\s*[xх×*]\s*\d{3,4})",
           "частота": r"(\d{2,3})\s*(?:гц|герц)",
           "тип матрицы": r"\b(ips|va|(?:qd-)?oled|tn)\b",
           "яркость": r"(\d{3,4})\s*(?:кд|нит|cd)",
           "контрастность": r"((?:\d{1,3}(?:\s\d{3})+|\d+(?:[.,]\d+)?)\s*(?:m|млн)?)\s*(?::|к|/)\s*1\b",
           "изогнутость": r"(изогнут|curved|плоск)"}
CURVED = {"изогнут": "да", "curved": "да", "плоск": "нет"}
# названия разрешения словом и их цифры (для наборов names); варианты записи — как на пилоте
RES_NAMES = {"1920x1080": ["Full HD", "FullHD", "FHD"], "2560x1440": ["QHD", "WQHD", "2K"],
             "3840x2160": ["4K", "UHD", "4K UHD"]}
P_NAME_ONLY = 0.5     # доля описаний, где разрешение названо только словом (эталон null)

def found_in_text(attr, text):
    text = text.lower().replace("ё", "е")
    found = set()
    for m in re.findall(PATTERN[attr], text):
        v = CURVED[m] if attr == "изогнутость" else NORM[attr](m)
        if v: found.add(v)
    return found

def check(text, gold):
    """пустая строка, если всё в порядке; иначе что не так"""
    for attr in ATTRS:
        found = found_in_text(attr, text)
        want = {gold[attr]} if gold[attr] else set()
        if found != want:
            return f"{attr}: нужно {sorted(want) or 'не называть'}, в тексте {sorted(found) or 'нет'}"
    return ""

# --- что просим написать ---
def make_gold(rng, pool, p):
    gold = {}
    for attr in ATTRS:
        gold[attr] = rng.choice(pool[attr]) if rng.random() < p[attr] else None
    return gold

def make_format(rng, attr, v):
    """как именно записать значение в тексте (форматы взяты из EDA)"""
    if attr == "диагональ":
        s = v.replace(".", ",") if "." in v and rng.random() < 0.15 else v
        return rng.choice([f'{s}"', f"{s}-дюймовый", f"{s} дюймов"])
    if attr == "разрешение":
        w, h = v.split("x")
        return f"{w}{rng.choice(['x', 'x', 'x', 'х', '×'])}{h}"
    if attr == "частота":
        return rng.choice([f"{v} Гц", f"{v} Гц", f"{v}Гц"])
    if attr == "яркость":
        return rng.choice([f"{v} кд/м²", f"{v} нит", f"{v} cd/m2"])
    if attr == "контрастность":
        return f"{v}:1"
    if attr == "изогнутость":
        return "изогнутый" if v == "да" else "плоский"
    return v   # тип матрицы пишем как есть

def make_prompt(gold, fmt, style, brand, start, length, res_name=None):
    must = "\n".join(f"- {a}: {fmt[a]}" for a in ATTRS if gold[a])
    never = ", ".join(a for a in ATTRS if not gold[a] and not (res_name and a == "разрешение")) or "нет"
    if res_name and gold["разрешение"]:
        name_rule = f"\nРазрешение назови и словом «{res_name}», и цифрами, как указано выше.\n"
    elif res_name:
        name_rule = (f"\nРазрешение назови только словом «{res_name}» (например, «{res_name}-матрица» или «картинка в {res_name}»). "
                     "Цифрами разрешение не пиши нигде: ни «1920x1080», ни «1080p», ни «2560 на 1440».\n")
    else:
        name_rule = ""
    examples = "\n\n".join(f"Пример {i + 1}:\n{t}" for i, t in enumerate(style))
    return f"""Напиши описание монитора для карточки товара интернет-магазина электроники, по-русски.
Бренд: {brand}. Название модели придумай сам (в стиле артикулов, например «27G2» или «XG273QZ»).
Начни описание {start}.
Объём около {length} символов (не больше чем на 15% длиннее), связный текст в стиле примеров ниже: маркетинговые фразы, порты, подсветка, эргономика, режимы и т.п.

В тексте обязательно назови эти характеристики, именно в такой записи:
{must}
{name_rule}
Эти характеристики не называй совсем (ни числом, ни словом, ни намёком): {never}.
Каждую обязательную характеристику называй с одним и тем же значением, без противоречий.

Реальные описания для стиля (не копируй из них фразы):

{examples}

Ответь только текстом описания, без заголовка и пояснений."""

# размышление выключаем, чтобы условия были одинаковыми и не оплачивались лишние токены.
# Sonnet 5.5 не принимает "disabled", поэтому у моделей разные функции.
def ask_sonnet(client, prompt):
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=2000, thinking={"type": "between_tools"},
                               messages=[{"role": "user", "content": prompt}])
    print("токены: вход", r.usage.input_tokens, "выход", r.usage.output_tokens)   # для оценки затрат
    return "".join(b.text for b in r.content if b.type == "text").strip()

def ask_haiku(client, prompt):
    r = client.messages.create(model="claude-haiku-4-5", max_tokens=2000, thinking={"type": "disabled"},
                               messages=[{"role": "user", "content": prompt}])
    print("токены: вход", r.usage.input_tokens, "выход", r.usage.output_tokens)
    return "".join(b.text for b in r.content if b.type == "text").strip()

def grams(text, n=6):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def make_rng(seed, index, attempt):
    # свой генератор для каждой записи: параметры зависят только от seed, номера записи и попытки
    key = f"{seed}:{index}:{attempt}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

def generate(args):
    import anthropic
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv(ROOT / '.env')
    name, n, seed = args.name, args.count, args.seed
    ask = ask_sonnet if args.model == 'sonnet' else ask_haiku

    # только пилот: частоты и значения из gold_pairs, стиль из текстов пилота
    pilot = set(json.loads(args.splits.read_text(encoding="utf-8"))["pilot"])
    pairs = pd.read_csv(args.pairs)
    pairs = pairs[pairs["url"].isin(pilot)]
    p = {a: pairs[pairs["атрибут"] == a]["эталон"].notna().mean() for a in ATTRS}
    pool = {a: pairs[pairs["атрибут"] == a]["эталон"].dropna().astype(str).tolist() for a in ATTRS}
    cards = [c for c in json.loads(args.cards.read_text(encoding="utf-8")) if c["url"] in pilot]
    real = [c["description"] for c in cards]
    brands = sorted({c["category"].split(" / ")[-1] for c in cards})

    if len(cards) != len(pilot) or len(real) < N_STYLE:
        raise ValueError('Не все карточки pilot найдены во входных данных')
    if any(not pool[a] for a in ATTRS):
        raise ValueError('В pilot нет значений для одного из атрибутов')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{name}.jsonl"
    done = 0
    if path.exists():
        with path.open(encoding='utf-8') as existing:
            for line in existing:
                json.loads(line)
                done += 1
    if done >= n:
        print('уже готово:', path, '| записей:', done)
        return
    client = anthropic.Anthropic()
    bad = 0
    calls = 0
    spent = 0     # секунд на запросы к API
    attempt = 0   # номер попытки для текущей записи: после брака параметры другие
    names = name.startswith("names") or "_names" in name
    while done < n:
        rng = make_rng(seed, done, attempt)
        gold = make_gold(rng, pool, p)
        res_name = None
        if names:            # разрешение всегда названо словом; в половине случаев без цифр
            res = rng.choice(sorted(RES_NAMES))
            res_name = rng.choice(RES_NAMES[res])
            gold["разрешение"] = None if rng.random() < P_NAME_ONLY else res
        fmt = {a: make_format(rng, a, v) for a, v in gold.items() if v}
        style = rng.sample(real, N_STYLE)
        length = min(max(50 * round(len(rng.choice(real)) / 50), 250), 1300)   # длина как у случайного реального описания
        t0 = time.time()
        text = ask(client, make_prompt(gold, fmt, style, rng.choice(brands), rng.choice(STARTS), length, res_name))
        sec = time.time() - t0
        spent += sec
        calls += 1
        reason = check(text, gold) or ("скопирована фраза из образца" if any(grams(text) & grams(s) for s in style) else "")
        if names and not reason:
            if res_name.lower() not in text.lower():
                reason = f"нет названия разрешения «{res_name}»"
            elif gold["разрешение"] is None and re.search(r"\b(?:720|1080|1440|2160)\s*[pр]\b", text.lower()):
                reason = "разрешение цифрами вида 1080p при эталоне null"
        if reason:
            bad += 1
            attempt += 1
            print("отбраковано:", bad, "| принято:", done, "|", f"{sec:.1f} с |", reason)
            continue
        with open(path, "a", encoding="utf-8") as f:
            rec = {"description": text, "gold": gold, "формат": fmt}
            if names:
                rec["название_разрешения"] = res_name
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1
        attempt = 0
        print(done, "/", n, f"| {sec:.1f} с")
    print("готово:", path, "| отбраковано:", bad)
    if calls:
        print(f"запросов: {calls} | в среднем {spent / calls:.1f} с | всего {spent / 60:.1f} мин")

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', choices=['fewshot', 'train', 'train_names'])
    parser.add_argument('count', type=int, help='Итоговое число принятых описаний')
    parser.add_argument('seed', type=int)
    parser.add_argument('--model', choices=['sonnet', 'haiku'], default='sonnet')
    parser.add_argument('--cards', type=Path, default=ROOT / 'data/dns_cards_500.json')
    parser.add_argument('--pairs', type=Path, default=ROOT / 'data/eda/gold_pairs.csv')
    parser.add_argument('--splits', type=Path,
                        default=ROOT / '02_eda/results/split_manifest.json')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'data/synthetic')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count должен быть положительным')
    generate(args)


if __name__ == '__main__':
    main()
