import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from common import CORPORA, load_corpus, fit_loglog

def zipf_fit(tokens):
    freq = tokens.value_counts().to_numpy()  # sorted high to low
    rank = np.arange(1, len(freq) + 1)
    model, lx, ly, pred = fit_loglog(rank, freq)
    return model, lx, ly, pred, len(tokens), len(freq)

def run_zipf(label, get_tokens):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    rows = []
    for ax, name in zip(axes, CORPORA):
        tokens = get_tokens(name)
        model, lx, ly, pred, n_tok, n_vocab = zipf_fit(tokens)
        ci = model.conf_int(alpha=0.10)
        alpha = -model.params[1]
        rows.append({
            "corpus": name,
            "tokens": n_tok,
            "vocab": n_vocab,
            "alpha": round(alpha, 3),
            "alpha_ci90_low": round(-ci[1][1], 3),
            "alpha_ci90_high": round(-ci[1][0], 3),
            "log10_C": round(model.params[0], 3),
            "R2": round(model.rsquared, 4),
        })
        ax.scatter(lx, ly, s=4, alpha=0.5, label="data")
        ax.plot(lx, pred["mean"].to_numpy(), color="red", label="fit")
        ax.fill_between(
            lx,
            pred["obs_ci_lower"].to_numpy(),
            pred["obs_ci_upper"].to_numpy(),
            color="red", alpha=0.15, label="90% prediction interval",
        )
        ax.set_title(f"{name.upper()}: alpha={alpha:.2f}, R$^2$={model.rsquared:.3f}")
        ax.set_xlabel("log10 rank")
        ax.set_ylabel("log10 frequency")
        ax.legend()
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(f"results/zipf_{label}.png", dpi=150)

    table = pd.DataFrame(rows)
    table.to_csv(f"results/zipf_{label}.csv", index=False)
    return table

if __name__ == "__main__":
    table = run_zipf("clean", lambda name: load_corpus(name)["token"])
    print(table.to_string(index=False))