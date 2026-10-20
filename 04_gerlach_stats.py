import sys
import numpy as np
sys.path.append("external")
from filter_words import run_stopword_statistics

from common import CORPORA, load_corpus

N_S = 200  # test value; use 1000 for the final run (the paper uses 1000)

def corpus_to_texts(name):
    df = load_corpus(name)
    return [g["token"].tolist() for _, g in df.groupby("text_id", sort=False)]

if __name__ == "__main__":
    np.random.seed(0)  # their shuffle uses numpy's global random state
    for name in CORPORA:
        texts = corpus_to_texts(name)
        sizes = np.array([len(t) for t in texts])
        print(f"\n{name.upper()}: {len(texts)} docs, tokens/doc "
              f"min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")
        stats = run_stopword_statistics(texts, N_s=N_S)
        stats.index.name = "token"
        stats.to_csv(f"results/gerlach_stats_{name}.csv", encoding="utf-8")
        print(f"words with I < 0: {(stats['I'] < 0).sum()} of {len(stats)}")
        print(stats.sort_values("N", ascending=False).head(12)[["N", "I", "tfidf"]].to_string())