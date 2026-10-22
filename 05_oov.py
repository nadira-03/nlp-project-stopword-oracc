import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from common import CORPORA, load_corpus
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

def in_wordnet(word):
    return len(wn.synsets(word)) > 0

if __name__ == "__main__":
    rows = []
    for name in CORPORA:
        counts = load_corpus(name)["token"].value_counts()
        known = pd.Series([in_wordnet(w) for w in counts.index], index=counts.index)
        n_types, n_tokens = len(counts), int(counts.sum())
        oov_types = int((~known).sum())
        oov_tokens = int(counts[~known].sum())
        rows.append({
            "corpus": name,
            "types": n_types,
            "oov_types": oov_types,
            "pct_oov_types": round(100 * oov_types / n_types, 2),
            "tokens": n_tokens,
            "oov_tokens": oov_tokens,
            "pct_oov_tokens": round(100 * oov_tokens / n_tokens, 2),
        })
        print(f"\n{name.upper()}: most frequent tokens found in WordNet")
        print(counts[known].head(12).to_dict())

    table = pd.DataFrame(rows)
    table.to_csv("results/oov_wordnet.csv", index=False)
    print(table.to_string(index=False))