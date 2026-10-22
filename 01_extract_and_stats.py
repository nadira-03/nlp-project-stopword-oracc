import html
from pathlib import Path
import pandas as pd

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

def extract(path):
    rows = []
    text_id, sent_id = None, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<text"):
                text_id = line.split('url="')[1].split('"')[0].rstrip("/").split("/")[-1]
            elif line.startswith("<sentence"):
                sent_id += 1
            elif line.startswith("<") or not line.strip():
                continue                          # other tags and blank lines
            else:
                cols = line.split("\t")
                if len(cols) >= 5:
                    token = html.unescape(cols[1]).strip().lower()   # column 2 shows normalised form
                    if token:
                        rows.append((text_id, sent_id, token, cols[4].strip()))  # column 5 = PoS
    return pd.DataFrame(rows, columns=["text_id", "sent_id", "token", "pos"])

stats = []
for name in CORPORA:
    path = next(Path("data/raw").rglob(f"oracc_{name}.vrt"))
    df = extract(path)
    df.to_csv(f"data/processed/{name}.csv", index=False)

    clean = df[df["token"] != "_"]                # leave out the "_" placeholder tokens
    chars = clean["token"].str.len()
    sent = clean.groupby("sent_id").size()
    stats.append({
        "corpus": name,
        "tokens": len(clean),
        "vocab_size": clean["token"].nunique(),
        "char_min": chars.min(), "char_max": chars.max(), "char_mean": round(chars.mean(), 2),
        "sent_min": sent.min(), "sent_max": sent.max(), "sent_mean": round(sent.mean(), 2),
    })

table = pd.DataFrame(stats)
table.to_csv("results/corpus_stats.csv", index=False)
print(table.to_string(index=False))