import json
import sys
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from nltk.corpus import stopwords

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

nltk.download("stopwords", quiet=True)
nltk_sw = set(stopwords.words("english"))
step = sys.argv[1] if len(sys.argv) > 1 else "task4-5"

def load_tokens(name):
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    return df[df["token"] != "_"]

def fit_and_plot(ax, x, y, xlabel, ylabel, name, symbol, sign):
    """Line fit on log-log scale, drawn with a 90% band. Returns exponent and R2."""
    lx, ly = np.log10(x), np.log10(y)
    X = sm.add_constant(lx)
    model = sm.OLS(ly, X).fit()
    band = model.get_prediction(X).summary_frame(alpha=0.10)
    exponent = sign * model.params[1]
    ax.scatter(lx, ly, s=5, alpha=0.5, label="data")
    ax.plot(lx, band["mean"], color="red", label="fit")
    ax.fill_between(lx, band["obs_ci_lower"], band["obs_ci_upper"],
                    color="red", alpha=0.15, label="90% prediction interval")
    ax.set_title(f"{name.upper()}: {symbol}={exponent:.2f}, R$^2$={model.rsquared:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return exponent, model.rsquared

def r2(x, y):
    """R2 of a straight-line fit on log-log scale (= squared correlation of the logs)."""
    return np.corrcoef(np.log10(x), np.log10(y))[0, 1] ** 2

def zipf_r2(codes):
    freq = np.sort(np.bincount(codes))[::-1]
    freq = freq[freq > 0]
    return r2(np.arange(1, len(freq) + 1), freq)

def heaps_r2(codes):
    first = np.zeros(len(codes))
    first[np.unique(codes, return_index=True)[1]] = 1
    vocab = first.cumsum()
    n = np.unique(np.logspace(0, np.log10(len(codes)), 300).astype(int))
    return r2(n, vocab[n - 1])

def verdict(delta, lo, hi, min_effect=0.001):
    if lo > 0 and delta > min_effect:
        return "enhances"
    if hi < 0 and delta < -min_effect:
        return "degrades"
    return "no meaningful change"

def refit_and_test(label, stop_of):
    """Remove each corpus's stopwords (stop_of[name] is a set), refit Zipf and Heaps,
    and test with a document bootstrap whether R2 got better or worse."""
    fig_z, axes_z = plt.subplots(2, 3, figsize=(15, 9))
    fig_h, axes_h = plt.subplots(2, 3, figsize=(15, 9))
    fit_rows, test_rows = [], []
    rng = np.random.default_rng(0)

    for i, name in enumerate(CORPORA):
        df = load_tokens(name)
        stop = stop_of[name]
        kept = df.loc[~df["token"].isin(stop), "token"]

        freq = kept.value_counts().to_numpy()
        rank = np.arange(1, len(freq) + 1)
        alpha, r2_zipf = fit_and_plot(axes_z.ravel()[i], rank, freq,
                                      "log10 rank", "log10 frequency", name, "alpha", -1)

        vocab = (~kept.duplicated()).cumsum().to_numpy()
        n = np.unique(np.logspace(0, np.log10(len(vocab)), 300).astype(int))
        beta, r2_heaps = fit_and_plot(axes_h.ravel()[i], n, vocab[n - 1], "log10 number of tokens", "log10 vocabulary size", name, "beta", 1)
        fit_rows.append({"corpus": name,
                         "types_removed": df["token"].nunique() - kept.nunique(),
                         "pct_tokens_removed": round(100 * (1 - len(kept) / len(df)), 2), 
                         "zipf_alpha": round(alpha, 3), "zipf_R2": round(r2_zipf, 4),
                         "heaps_beta": round(beta, 3), "heaps_R2": round(r2_heaps, 4)})

        codes, words = pd.factorize(df["token"])
        is_sw = np.array([w in stop for w in words])
        text = df["text_id"].to_numpy()
        docs = np.split(codes, np.flatnonzero(text[1:] != text[:-1]) + 1)
        laws = {"zipf": zipf_r2, "heaps": heaps_r2}
        delta = {law: f(codes[~is_sw[codes]]) - f(codes) for law, f in laws.items()}
        boot = {law: [] for law in laws}
        for _ in range(200):
            pick = rng.integers(0, len(docs), len(docs))
            sample = np.concatenate([docs[j] for j in pick])
            for law, f in laws.items():
                boot[law].append(f(sample[~is_sw[sample]]) - f(sample))
        for law in laws:
            lo, hi = np.percentile(boot[law], [5, 95])
            test_rows.append({"corpus": name, "law": law, "delta_R2": round(delta[law], 4),
                              "ci90_low": round(lo, 4), "ci90_high": round(hi, 4),
                              "verdict": verdict(delta[law], lo, hi)})

    axes_z.ravel()[-1].axis("off")
    axes_h.ravel()[-1].axis("off")
    fig_z.tight_layout()
    fig_h.tight_layout()
    fig_z.savefig(f"results/zipf_{label}.png", dpi=150)
    fig_h.savefig(f"results/heaps_{label}.png", dpi=150)

    fits, tests = pd.DataFrame(fit_rows), pd.DataFrame(test_rows)
    fits.to_csv(f"results/zipf_heaps_{label}.csv", index=False)
    tests.to_csv(f"results/{label}_tests.csv", index=False)
    print(fits.to_string(index=False))
    print()
    print(tests.to_string(index=False))


# TASK 6
if step == "gerlach":
    sys.path.append("external")
    from filter_words import run_stopword_statistics

    np.random.seed(0)
    for name in CORPORA:
        df = load_tokens(name)
        texts = [g["token"].tolist() for _, g in df.groupby("text_id", sort=False)]
        stats = run_stopword_statistics(texts, N_s=200)
        stats.index.name = "token"
        stats.to_csv(f"results/gerlach_stats_{name}.csv", encoding="utf-8")
        print(name, len(texts), "documents,", len(stats), "words")

# stopword lists
elif step == "lists":
    sys.path.append("external")
    from filter_words import make_stopwordlist
    rows, lists = [], {}
    for name in CORPORA:
        stats = pd.read_csv(f"results/gerlach_stats_{name}.csv", index_col="token",
                            keep_default_na=False)
        frequent = stats[stats["N"] >= 10].copy()
        frequent["I"] = frequent["I"].abs()
        sw = make_stopwordlist(frequent, method="INFOR", cutoff_type="p", cutoff_val=0.2)
        lists[name] = sw
        stats.loc[sw, ["N", "I", "tfidf"]].sort_values("N", ascending=False).to_csv(
            f"results/gerlach_list_{name}.csv", encoding="utf-8")

        K = len(sw)
        top_k = stats["N"].sort_values(ascending=False, kind="mergesort").index[:K]
        hits = len(set(top_k) & set(sw))
        rows.append({"corpus": name, "K": K,
                     "pct_tokens": round(100 * stats.loc[sw, "N"].sum() / stats["N"].sum(), 2),
                     "hits_in_top_K": hits, "proportion": round(hits / K, 3)})

    with open("results/gerlach_stoplist.json", "w", encoding="utf-8") as f:
        json.dump(lists, f, ensure_ascii=False)
    table6 = pd.DataFrame(rows)
    table6.to_csv("results/gerlach_coverage.csv", index=False)
    print(table6.to_string(index=False))

# TASK 7
elif step == "task7":
    with open("results/gerlach_stoplist.json", encoding="utf-8") as f:
        lists = json.load(f)
    refit_and_test("gerlach_removed", {name: set(lists[name]) for name in CORPORA})

# TASKS 4 and 5
else:
    rows = []
    for name in CORPORA:
        counts = load_tokens(name)["token"].value_counts()
        present = [w for w in nltk_sw if w in counts.index]
        K = len(present)
        hits = sum(w in nltk_sw for w in counts.index[:K])
        rows.append({"corpus": name, "K": K, "hits_in_top_K": hits,
                     "proportion": round(hits / K, 3) if K else float("nan")})
    table4 = pd.DataFrame(rows)
    table4.to_csv("results/nltk_coverage.csv", index=False)
    print(table4.to_string(index=False))
    print()
    refit_and_test("nltk_removed", {name: nltk_sw for name in CORPORA})