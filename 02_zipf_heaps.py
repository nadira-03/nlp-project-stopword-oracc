import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

def fit_and_plot(ax, x, y, xlabel, ylabel, name, symbol, sign):
    """Straight-line fit on log-log scale, drawn with a 90% band. Returns exponent and R2."""
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

fig_z, axes_z = plt.subplots(2, 3, figsize=(15, 9))
fig_h, axes_h = plt.subplots(2, 3, figsize=(15, 9))
rows = []

for i, name in enumerate(CORPORA):
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    tokens = df.loc[df["token"] != "_", "token"]          # leave out "_" placeholders

    # Task 2: Zipf's law (frequency against rank)
    freq = tokens.value_counts().to_numpy()               # high to low
    rank = np.arange(1, len(freq) + 1)
    alpha, r2_zipf = fit_and_plot(axes_z.ravel()[i], rank, freq,
                                  "log10 rank", "log10 frequency", name, "alpha", -1)

    # Task 3: Heaps' law (vocabulary size against number of tokens read)
    vocab = (~tokens.duplicated()).cumsum().to_numpy()    # distinct words so far
    n = np.unique(np.logspace(0, np.log10(len(vocab)), 300).astype(int))
    beta, r2_heaps = fit_and_plot(axes_h.ravel()[i], n, vocab[n - 1],
                                  "log10 number of tokens", "log10 vocabulary size",
                                  name, "beta", 1)

    rows.append({"corpus": name, "zipf_alpha": round(alpha, 3), "zipf_R2": round(r2_zipf, 4),
                 "heaps_beta": round(beta, 3), "heaps_R2": round(r2_heaps, 4)})

axes_z.ravel()[-1].axis("off")
axes_h.ravel()[-1].axis("off")
fig_z.tight_layout()
fig_h.tight_layout()
fig_z.savefig("results/zipf_clean.png", dpi=150)
fig_h.savefig("results/heaps_clean.png", dpi=150)

table = pd.DataFrame(rows)
table.to_csv("results/zipf_heaps_clean.csv", index=False)
print(table.to_string(index=False))