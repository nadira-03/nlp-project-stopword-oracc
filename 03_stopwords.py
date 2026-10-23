import nltk
import pandas as pd
from nltk.corpus import stopwords

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

nltk.download("stopwords", quiet=True)
nltk_sw = set(stopwords.words("english"))

rows = []
for name in CORPORA:
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    counts = df.loc[df["token"] != "_", "token"].value_counts()    # most frequent first

    present = [w for w in nltk_sw if w in counts.index]   # NLTK stopwords that occur in this corpus
    K = len(present)
    top_k = counts.index[:K]                              # the K most frequent tokens
    hits = sum(w in nltk_sw for w in top_k)               # how many of them are NLTK stopwords

    rows.append({
        "corpus": name,
        "K": K,
        "hits_in_top_K": hits,
        "proportion": round(hits / K, 3) if K else float("nan"),
    })

table = pd.DataFrame(rows)
table.to_csv("results/nltk_coverage.csv", index=False)
print(table.to_string(index=False))