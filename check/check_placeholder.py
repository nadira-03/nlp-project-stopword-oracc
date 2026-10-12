from collections import Counter
from pathlib import Path

RAW = Path("data/raw")

for name in ["cams", "dcclt", "ribo", "rinap", "saao"]:
    path = next(RAW.rglob(f"oracc_{name}.vrt"))
    lang_us, lang_other = Counter(), Counter()
    forms = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("<") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            if cols[1].strip() == "_":
                lang_us[cols[8]] += 1
                forms[cols[0]] += 1
            else:
                lang_other[cols[8]] += 1
    print("=" * 60)
    print(name.upper())
    print("language of '_' tokens:   ", lang_us.most_common(5))
    print("language of other tokens: ", lang_other.most_common(5))
    print("most common col-1 forms among '_' tokens:", forms.most_common(10))