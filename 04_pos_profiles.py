import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import jensenshannon, squareform
from common import CORPORA, load_corpus

MAIN = "abs_min10_p0.2"
TOP_N = 30
MIN_CONTEXTS = 20

with open("results/gerlach_lists.json", encoding="utf-8") as f:
    LISTS = json.load(f)

def neighbour_counts(df, words):
    """Counts of neighbouring PoS tags (previous and next token, same sentence) per word."""
    df = df.reset_index(drop=True)
    pos = df["pos"]
    ph = df["token"] == "_"
    prev_ok = (df["sent_id"] == df["sent_id"].shift(1)) & ~ph.shift(1, fill_value=True)
    next_ok = (df["sent_id"] == df["sent_id"].shift(-1)) & ~ph.shift(-1, fill_value=True)
    prev_pos = pos.shift(1).where(prev_ok)
    next_pos = pos.shift(-1).where(next_ok)

    sel = df["token"].isin(words)
    tok = df.loc[sel, "token"]
    rows = pd.concat([
        pd.DataFrame({"token": tok, "tag": prev_pos[sel]}),
        pd.DataFrame({"token": tok, "tag": next_pos[sel]}),
    ]).dropna()
    return rows.groupby(["token", "tag"]).size().unstack(fill_value=0)

def similarity_matrix(counts):
    P = counts.div(counts.sum(axis=1), axis=0).to_numpy()
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = jensenshannon(P[i], P[j], base=2)
    return 1 - D, D

def plot_heatmap(S, D, words, name):
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    S_o = S[np.ix_(order, order)]
    labels = [words[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(S_o, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f"{name.upper()}: PoS-profile similarity of stopwords (1 - JS distance)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"results/pos_similarity_{name}.png", dpi=150)
    plt.close(fig)

def plot_histograms(counts, name, n_words=12, n_tags=10):
    """Small-multiple bar charts: PoS-tag distribution next to each stopword."""
    P = counts.div(counts.sum(axis=1), axis=0)
    tags = counts.sum().sort_values(ascending=False).head(n_tags).index
    words = list(P.index[:n_words])
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    for ax, w in zip(axes.ravel(), words):
        ax.bar(range(len(tags)), P.loc[w, tags].to_numpy())
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=90, fontsize=7)
        ax.set_title(f"{w} (n={int(counts.loc[w].sum())})", fontsize=9)
    for ax in axes.ravel()[len(words):]:
        ax.axis("off")
    fig.suptitle(f"{name.upper()}: PoS-tag profile of the 12 most frequent stopwords "
                 f"(share of neighbours)")
    fig.tight_layout()
    fig.savefig(f"results/pos_histograms_{name}.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    for name in CORPORA:
        df = load_corpus(name, remove_placeholder=False)
        stats = pd.read_csv(f"results/gerlach_stats_{name}.csv",
                            index_col="token", keep_default_na=False)
        sw = [w for w in LISTS[name][MAIN] if w in stats.index]
        top = stats.loc[sw, "N"].sort_values(ascending=False).head(TOP_N).index

        counts = neighbour_counts(df, set(top))
        counts = counts.loc[counts.sum(axis=1) >= MIN_CONTEXTS]
        words = [w for w in top if w in counts.index]
        counts = counts.loc[words]

        S, D = similarity_matrix(counts)
        counts.to_csv(f"results/pos_profiles_{name}.csv", encoding="utf-8")
        pd.DataFrame(S, index=words, columns=words).to_csv(
            f"results/pos_similarity_{name}.csv", encoding="utf-8")
        plot_heatmap(S, D, words, name)
        plot_histograms(counts, name)

        n = len(words)
        iu = np.triu_indices(n, 1)
        hi, lo = S[iu].argmax(), S[iu].argmin()
        print(f"\n{name.upper()}: {n} words, {int(counts.to_numpy().sum())} neighbour counts, "
              f"{counts.shape[1]} PoS tags")
        print("  most similar pair: ", words[iu[0][hi]], words[iu[1][hi]], round(S[iu][hi], 3))
        print("  least similar pair:", words[iu[0][lo]], words[iu[1][lo]], round(S[iu][lo], 3))
        print("  top tags overall:", counts.sum().sort_values(ascending=False).head(6).to_dict())