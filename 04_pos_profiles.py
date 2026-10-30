import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

with open("results/gerlach_stoplist.json", encoding="utf-8") as f:
    stoplists = json.load(f)

for name in CORPORA:
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    stats = pd.read_csv(f"results/gerlach_stats_{name}.csv", index_col="token", keep_default_na=False)
    top = stats.loc[stoplists[name], "N"].sort_values(ascending=False).head(30).index

    ok_prev = (df["sent_id"] == df["sent_id"].shift(1)) & (df["token"].shift(1) != "_")
    ok_next = (df["sent_id"] == df["sent_id"].shift(-1)) & (df["token"].shift(-1) != "_")
    prev_tag = df["pos"].shift(1).where(ok_prev)
    next_tag = df["pos"].shift(-1).where(ok_next)

    is_sw = df["token"].isin(top)
    word = df.loc[is_sw, "token"]
    tags = pd.concat([pd.DataFrame({"word": word, "tag": prev_tag[is_sw]}),
                      pd.DataFrame({"word": word, "tag": next_tag[is_sw]})]).dropna()
    counts = tags.groupby(["word", "tag"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1) >= 20]
    counts = counts.loc[[w for w in top if w in counts.index]]
    counts.to_csv(f"results/pos_profiles_{name}.csv", encoding="utf-8")
    share = counts.div(counts.sum(axis=1), axis=0)

    common = counts.sum().sort_values(ascending=False).head(10).index
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    for ax, w in zip(axes.ravel(), share.index[:12]):
        ax.bar(common, share.loc[w, common])
        ax.set_title(f"{w} (n={int(counts.loc[w].sum())})")
        ax.set_xlabel("PoS tag of neighbour")
        ax.set_ylabel("share of neighbours")
        ax.tick_params(axis="x", rotation=90)
    for ax in axes.ravel()[min(12, len(share)):]:
        ax.axis("off")
    fig.suptitle(f"{name.upper()}: PoS profile of the most frequent stopwords")
    fig.tight_layout()
    fig.savefig(f"results/pos_histograms_{name}.png", dpi=150)
    plt.close(fig)

    P = share.to_numpy()
    n = len(P)
    sim = np.array([[1 - jensenshannon(P[i], P[j], base=2) for j in range(n)] for i in range(n)])
    pd.DataFrame(sim, index=share.index, columns=share.index).to_csv(f"results/pos_similarity_{name}.csv")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_xticklabels(share.index, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(share.index, fontsize=8)
    ax.set_title(f"{name.upper()}: similarity of stopword PoS profiles")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(f"results/pos_similarity_{name}.png", dpi=150)
    plt.close(fig)
    print(name, n, "stopwords profiled")