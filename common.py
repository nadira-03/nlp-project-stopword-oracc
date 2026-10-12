import numpy as np
import pandas as pd
import statsmodels.api as sm

CORPORA = ["cams", "dcclt", "ribo", "rinap", "saao"]

def load_corpus(name, remove_placeholder=True):
    df = pd.read_csv(f"data/processed/{name}.csv", keep_default_na=False)
    if remove_placeholder:
        df = df[df["token"] != "_"]
    return df

def fit_loglog(x, y):
    """OLS fit of log10(y) on log10(x), with 90% intervals."""
    lx, ly = np.log10(x), np.log10(y)
    X = sm.add_constant(lx)
    model = sm.OLS(ly, X).fit()
    pred = model.get_prediction(X).summary_frame(alpha=0.10)
    return model, lx, ly, pred