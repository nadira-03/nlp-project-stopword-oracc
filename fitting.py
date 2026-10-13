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