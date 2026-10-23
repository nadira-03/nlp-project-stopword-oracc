import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

rows = []
for name in CORPORA:
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    vocab = df.loc[df["token"] != "_", "token"].unique()      # the distinct words of the corpus
    oov = [w for w in vocab if not wn.synsets(w)]             # words with no WordNet entry
    rows.append({"corpus": name,
                 "vocab_size": len(vocab),
                 "oov_words": len(oov),
                 "pct_oov": round(100 * len(oov) / len(vocab), 2)})

table = pd.DataFrame(rows)
table.to_csv("results/oov_wordnet.csv", index=False)
print(table.to_string(index=False))