import html
import re
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
RES = Path("results")
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)

CORPORA = {
    "cams": "oracc_cams.vrt",
    "dcclt": "oracc_dcclt.vrt",
    "ribo": "oracc_ribo.vrt",
    "rinap": "oracc_rinap.vrt",
    "saao": "oracc_saao.vrt"
}

def extract(path, corpus):
    """Read one .vrt file line by line and return a DataFrame, one row per token."""
    rows = []
    text_id = period = genre = None
    sent_id = 0
    skipped = 0

    with open(path, encoding = "utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("<text"):
                m = re.search(r'url="([^"]*)"', line)
                text_id = m.group(1).rstrip("/").split("/")[-1] if m else None
                m = re.search(r'period="([^"]*)"', line)
                period = m.group(1) if m else None
                m = re.search(r'genre="([^"]*)"', line)
                genre = m.group(1) if m else None
            elif line.startswith("<sentence"):
                sent_id += 1
            elif line.startswith("<") or not line.strip():
                continue # closing tags, paragraph tags, blank lines
            else:
                cols = line.split("\t")
                if len(cols) < 5:
                    skipped += 1
                    continue
                token = html.unescape(cols[1]).strip().lower()  # column 2
                lemma = html.unescape(cols[2]).strip().lower()  # column 3
                pos = cols[4].strip()                           # column 5
                if not token:
                    skipped += 1
                    continue
                rows.append((corpus, text_id, sent_id, token, lemma, pos, period, genre))

    df = pd.DataFrame(
        rows,
        columns=["corpus", "text_id", "sent_id", "token", "lemma", "pos", "period", "genre"],
    )
    df["is_placeholder"] = df["token"] == "_"
    print(f"{corpus}: {len(df)} tokens kept, {skipped} lines skipped")
    return df

def compute_stats(df, corpus):
    char_len = df["token"].str.len()
    sent_len = df.groupby("sent_id").size()
    return {
        "corpus": corpus,
        "tokens": len(df),
        "vocab_size": df["token"].nunique(),
        "char_min": char_len.min(),
        "char_max": char_len.max(),
        "char_mean": round(char_len.mean(), 2),
        "sent_count": len(sent_len),
        "sent_min": sent_len.min(),
        "sent_max": sent_len.max(),
        "sent_mean": round(sent_len.mean(), 2),
    }

if __name__ == "__main__":
    all_stats = []
    for corpus, filename in CORPORA.items():
        path = next(RAW.rglob(filename))
        df = extract(path, corpus)
        df.to_csv(OUT / f"{corpus}.csv", index=False, encoding="utf-8")
        clean = df[~df["is_placeholder"]]
        all_stats.append(compute_stats(clean, corpus))

    table = pd.DataFrame(all_stats)
    table.to_csv(RES / "corpus_stats.csv", index=False)
    print()
    print(table.to_string(index=False))