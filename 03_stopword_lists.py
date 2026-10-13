import json

import nltk
import pandas as pd
from nltk.corpus import stopwords

from common import CORPORA, load_corpus
from fitting import run_zipf, run_heaps, bootstrap_removal

nltk.download("stopwords", quiet=True)
NLTK_SW = set(stopwords.words("english"))
TOP_N = 20

# task 4 
def nltk_coverage(tokens):
    counts = tokens.value_counts()
    vocab = set(counts.index)
    present = sorted(w for w in NLTK_SW if w in vocab)
    K = len(present)
    top_k = list(counts.index[:K])
    hits = [w for w in top_k if w in NLTK_SW]
    tokens_covered = counts[present].sum() / counts.sum() if K else 0.0
    return {
        "K": K,
        "hits_in_top_K": len(hits),
        "proportion": round(len(hits) / K, 3) if K else float("nan"),
        "pct_tokens_that_are_nltk_sw": round(100 * tokens_covered, 2),
    }, present, hits

def task4():
    rows, matches = [], {}
    for name in CORPORA:
        tokens = load_corpus(name)["token"]
        stats, present, hits = nltk_coverage(tokens)
        rows.append({"corpus": name, "vocab": tokens.nunique(), **stats})
        matches[name] = {"nltk_words_present": present, "top_K_hits": hits}
    table = pd.DataFrame(rows)
    table.to_csv("results/nltk_coverage.csv", index=False)
    with open("results/nltk_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print("TASK 4: NLTK coverage")
    print(table.to_string(index=False))


# task 5
def without(tokens, stop):
    return tokens[~tokens.isin(stop)]

def top_n_stop(tokens, n=TOP_N):
    return set(tokens.value_counts().index[:n])

def nltk_removed(name):
    return without(load_corpus(name)["token"], NLTK_SW)

def top_removed(name):
    tokens = load_corpus(name)["token"]
    return without(tokens, top_n_stop(tokens))

def task5(n_boot=200):
    for label, fn in [("nltk_removed", nltk_removed), (f"top{TOP_N}_removed", top_removed)]:
        print(f"\nTASK 5: ZIPF after {label}")
        print(run_zipf(label, fn).to_string(index=False))
        print(f"\nTASK 5: HEAPS after {label}")
        print(run_heaps(label, fn).to_string(index=False))

    parts = []
    for name in CORPORA:
        df = load_corpus(name)
        stop_sets = {"nltk": NLTK_SW, f"top{TOP_N}": top_n_stop(df["token"])}
        parts.append(bootstrap_removal(df, name, stop_sets, n_boot=n_boot))
        print(f"bootstrap done: {name}")
    table = pd.concat(parts, ignore_index=True)
    table.to_csv("results/stopword_removal_tests.csv", index=False)
    print("\nTASK 5: bootstrap test (R2 rows; the CSV also has alpha and beta)")
    print(table[table["metric"].str.endswith("R2")].to_string(index=False))

if __name__ == "__main__":
    task4()
    task5()