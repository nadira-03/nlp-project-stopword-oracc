import json
import sys

import numpy as np
import pandas as pd

sys.path.append("external")
from filter_words import make_stopwordlist

from common import CORPORA

P_VALUES = [0.1, 0.2, 0.3]  # fraction of tokens to remove
VARIANTS = {
    "signed": {"abs": False, "min_n": 1},      # repo default: most negative I first
    "abs": {"abs": True, "min_n": 1},          # rank by |I|
    "abs_min10": {"abs": True, "min_n": 10},   # rank by |I|, ignore words seen < 10 times
}
MAIN_VARIANT, MAIN_P = "abs_min10", 0.2


def load_stats(name):
    return pd.read_csv(f"results/gerlach_stats_{name}.csv", index_col="token",
                       keep_default_na=False)


def build_list(stats, use_abs, min_n, p):
    df = stats[stats["N"] >= min_n].copy()
    if use_abs:
        df["I"] = df["I"].abs()
    return make_stopwordlist(df, method="INFOR", cutoff_type="p", cutoff_val=p)


def evaluate(stats, sw):
    sw_set = set(sw)
    K = len(sw)
    in_list = stats.index.isin(sw_set)
    listed, rest = stats[in_list], stats[~in_list]
    ranked = stats["N"].sort_values(ascending=False, kind="mergesort")
    top_k = set(ranked.index[:K])
    return {
        "K": K,
        "pct_types": round(100 * K / len(stats), 2),
        "pct_tokens": round(100 * listed["N"].sum() / stats["N"].sum(), 2),
        "top_K_proportion": round(len(top_k & sw_set) / K, 3) if K else np.nan,
        "pct_listed_types_freq_le2": round(100 * (listed["N"] <= 2).mean(), 1) if K else np.nan,
        "median_I_listed": round(listed["I"].median(), 3) if K else np.nan,
        "median_tfidf_listed": round(listed["tfidf"].median(), 3) if K else np.nan,
        "median_tfidf_rest": round(rest["tfidf"].median(), 3),
    }


if __name__ == "__main__":
    rows, all_lists = [], {}
    for name in CORPORA:
        stats = load_stats(name)
        all_lists[name] = {}
        for vname, cfg in VARIANTS.items():
            for p in P_VALUES:
                sw = build_list(stats, cfg["abs"], cfg["min_n"], p)
                all_lists[name][f"{vname}_p{p}"] = sw
                rows.append({"corpus": name, "variant": vname, "p_target": p,
                             **evaluate(stats, sw)})
                if vname == MAIN_VARIANT and p == MAIN_P:
                    out = stats.loc[stats.index.isin(set(sw)), ["N", "I", "tfidf"]]
                    out = out.sort_values("N", ascending=False)
                    out.to_csv(f"results/gerlach_list_{name}.csv", encoding="utf-8")
                    print(f"\n{name.upper()} ({vname}, p={p}): top 15 listed words")
                    print(out.head(15).to_string())

    table = pd.DataFrame(rows)
    table.to_csv("results/gerlach_lists_summary.csv", index=False)
    with open("results/gerlach_lists.json", "w", encoding="utf-8") as f:
        json.dump(all_lists, f, ensure_ascii=False)
    print("\nSUMMARY (main variant; all variants are in the CSV)")
    print(table.to_string(index=False))