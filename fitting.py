import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common import CORPORA, fit_loglog

def zipf_fit(tokens):
    freq = tokens.value_counts().to_numpy()  # sorted high to low
    rank = np.arange(1, len(freq) + 1)
    model, lx, ly, pred = fit_loglog(rank, freq)
    return model, lx, ly, pred, len(tokens), len(freq)

def heaps_fit(tokens, n_points=300):
    first_seen = ~tokens.duplicated()
    vocab = first_seen.cumsum().to_numpy()
    N = len(vocab)
    n = np.unique(np.logspace(0, np.log10(N), n_points).astype(int))
    n = n[(n >= 1) & (n <= N)]
    V = vocab[n - 1]
    model, lx, ly, pred = fit_loglog(n, V)
    return model, lx, ly, pred, N, int(vocab[-1])

def _run(kind, label, get_tokens):
    fit_fn = zipf_fit if kind == "zipf" else heaps_fit
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    rows = []
    for ax, name in zip(axes, CORPORA):
        tokens = get_tokens(name)
        model, lx, ly, pred, n_tok, n_vocab = fit_fn(tokens)
        ci = model.conf_int(alpha=0.10)
        slope = model.params[1]
        exponent = -slope if kind == "zipf" else slope
        lo, hi = (-ci[1][1], -ci[1][0]) if kind == "zipf" else (ci[1][0], ci[1][1])
        sym = "alpha" if kind == "zipf" else "beta"
        rows.append({
            "corpus": name,
            "tokens": n_tok,
            "vocab": n_vocab,
            sym: round(exponent, 3),
            f"{sym}_ci90_low": round(lo, 3),
            f"{sym}_ci90_high": round(hi, 3),
            "log10_C": round(model.params[0], 3),
            "R2": round(model.rsquared, 4),
        })
        ax.scatter(lx, ly, s=5, alpha=0.5, label="data")
        ax.plot(lx, pred["mean"].to_numpy(), color="red", label="fit")
        ax.fill_between(
            lx,
            pred["obs_ci_lower"].to_numpy(),
            pred["obs_ci_upper"].to_numpy(),
            color="red", alpha=0.15, label="90% prediction interval",
        )
        ax.set_title(f"{name.upper()}: {sym}={exponent:.2f}, R$^2$={model.rsquared:.3f}")
        ax.set_xlabel("log10 rank" if kind == "zipf" else "log10 number of tokens")
        ax.set_ylabel("log10 frequency" if kind == "zipf" else "log10 vocabulary size")
        ax.legend()
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(f"results/{kind}_{label}.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(f"results/{kind}_{label}.csv", index=False)
    return table

def run_zipf(label, get_tokens):
    return _run("zipf", label, get_tokens)

def run_heaps(label, get_tokens):
    return _run("heaps", label, get_tokens)

# 5

def _linfit(lx, ly):
    slope, _ = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    return slope, r2

def zipf_quick(codes):
    freq = np.sort(np.bincount(codes))[::-1]
    freq = freq[freq > 0]
    rank = np.arange(1, len(freq) + 1)
    slope, r2 = _linfit(np.log10(rank), np.log10(freq))
    return -slope, r2

def heaps_quick(codes, n_points=300):
    N = len(codes)
    _, first_idx = np.unique(codes, return_index=True)
    indicator = np.zeros(N)
    indicator[first_idx] = 1
    vocab = np.cumsum(indicator)
    n = np.unique(np.logspace(0, np.log10(N), n_points).astype(int))
    n = n[(n >= 1) & (n <= N)]
    slope, r2 = _linfit(np.log10(n), np.log10(vocab[n - 1]))
    return slope, r2

METRICS = ["zipf_alpha", "zipf_R2", "heaps_beta", "heaps_R2"]

def _metrics(codes):
    a, r2z = zipf_quick(codes)
    b, r2h = heaps_quick(codes)
    return {"zipf_alpha": a, "zipf_R2": r2z, "heaps_beta": b, "heaps_R2": r2h}

def bootstrap_removal(df, corpus, stop_sets, n_boot=200, seed=0):
    """Compare Zipf/Heaps fits before and after removing each stopword set.
    Resamples whole documents (text_id) with replacement."""
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(df["token"])
    flags = {lab: np.array([t in s for t in uniques]) for lab, s in stop_sets.items()}

    text_ids = df["text_id"].to_numpy()
    bounds = np.flatnonzero(text_ids[1:] != text_ids[:-1]) + 1
    docs = np.split(codes, bounds)

    def compare(sample):
        base = _metrics(sample)
        after = {lab: _metrics(sample[~f[sample]]) for lab, f in flags.items()}
        return base, after

    base_pt, after_pt = compare(codes)          # point estimates on the real corpus
    reps = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(docs), len(docs))
        reps.append(compare(np.concatenate([docs[i] for i in pick])))

    rows = []
    for lab, f in flags.items():
        types_removed = int(f.sum())
        pct_removed = round(100 * f[codes].mean(), 2)
        for m in METRICS:
            diff = np.array([after[lab][m] - base[m] for base, after in reps])
            if np.all(diff == 0):
                p = 1.0
            else:
                p = min(1.0, 2 * min((diff > 0).mean(), (diff < 0).mean()))
                p = max(p, 1 / n_boot)          # bootstrap p cannot go below 1/n_boot
            rows.append({
                "corpus": corpus,
                "stoplist": lab,
                "types_removed": types_removed,
                "pct_tokens_removed": pct_removed,
                "metric": m,
                "before": round(base_pt[m], 4),
                "after": round(after_pt[lab][m], 4),
                "delta": round(after_pt[lab][m] - base_pt[m], 4),
                "boot_mean_delta": round(diff.mean(), 4),
                "ci90_low": round(np.percentile(diff, 5), 4),
                "ci90_high": round(np.percentile(diff, 95), 4),
                "p_boot": round(p, 3),
            })
    return pd.DataFrame(rows)