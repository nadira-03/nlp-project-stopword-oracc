import json
import sys

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

sys.path.append("external")
from filter_words import make_stopwordlist, run_stopword_statistics

from common import CORPORA, load_corpus
from fitting import run_zipf, run_heaps, bootstrap_removal

nltk.download("stopwords", quiet=True)
NLTK_SW = set(stopwords.words("english"))

TOP_N = 20                 # baseline: remove the top-N most frequent tokens
N_S = 200                  # Gerlach null-model shuffles (the paper uses 1000)
P_VALUES = [0.1, 0.2, 0.3]  # fraction of tokens to remove
VARIANTS = {
    "signed": {"abs": False, "min_n": 1},      # repo default: most negative I first
    "abs": {"abs": True, "min_n": 1},          # rank by |I|
    "abs_min10": {"abs": True, "min_n": 10},   # rank by |I|, ignore words seen < 10 times
}
MAIN_VARIANT, MAIN_P = "abs_min10", 0.2
MAIN_KEY = f"{MAIN_VARIANT}_p{MAIN_P}"
COMPARE_KEY = "signed_p0.2"


def without(tokens, stop):
    return tokens[~tokens.isin(stop)]


# ======================= TASK 4: NLTK coverage =======================
def nltk_coverage(tokens):
    counts = tokens.value_counts()
    vocab = set(counts.index)
    present = sorted(w for w in NLTK_SW if w in vocab)
    K = len(present)
    top_k = list(counts.index[:K])
    hits = [w for w in top_k if w in NLTK_SW]
    tokens_covered = counts[present].sum() / counts.sum() if K else 0.0
    return {
        "K": K,
        "hits_in_top_K": len(hits),
        "proportion": round(len(hits) / K, 3) if K else float("nan"),
        "pct_tokens_that_are_nltk_sw": round(100 * tokens_covered, 2),
    }, present, hits


def task4():
    rows, matches = [], {}
    for name in CORPORA:
        tokens = load_corpus(name)["token"]
        stats, present, hits = nltk_coverage(tokens)
        rows.append({"corpus": name, "vocab": tokens.nunique(), **stats})
        matches[name] = {"nltk_words_present": present, "top_K_hits": hits}
    table = pd.DataFrame(rows)
    table.to_csv("results/nltk_coverage.csv", index=False)
    with open("results/nltk_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    print("TASK 4: NLTK coverage")
    print(table.to_string(index=False))


# ======================= TASK 5: NLTK removal refits =======================
def top_n_stop(tokens, n=TOP_N):
    return set(tokens.value_counts().index[:n])


def nltk_removed(name):
    return without(load_corpus(name)["token"], NLTK_SW)


def top_removed(name):
    tokens = load_corpus(name)["token"]
    return without(tokens, top_n_stop(tokens))


def task5(n_boot=200):
    for label, fn in [("nltk_removed", nltk_removed), (f"top{TOP_N}_removed", top_removed)]:
        print(f"\nTASK 5: ZIPF after {label}")
        print(run_zipf(label, fn).to_string(index=False))
        print(f"\nTASK 5: HEAPS after {label}")
        print(run_heaps(label, fn).to_string(index=False))

    parts = []
    for name in CORPORA:
        df = load_corpus(name)
        stop_sets = {"nltk": NLTK_SW, f"top{TOP_N}": top_n_stop(df["token"])}
        parts.append(bootstrap_removal(df, name, stop_sets, n_boot=n_boot))
        print(f"bootstrap done: {name}")
    table = pd.concat(parts, ignore_index=True)
    table.to_csv("results/stopword_removal_tests.csv", index=False)
    print("\nTASK 5: bootstrap test (R2 rows; the CSV has all metrics)")
    print(table[table["metric"].str.endswith("R2")].to_string(index=False))


# ======================= TASK 6a: Gerlach statistics =======================
def corpus_to_texts(name):
    df = load_corpus(name)
    return [g["token"].tolist() for _, g in df.groupby("text_id", sort=False)]


def gerlach_stats():
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


# ======================= TASK 6b: Gerlach lists and evaluation =======================
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


def gerlach_lists():
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
    print("\nSUMMARY (all variants)")
    print(table.to_string(index=False))


# ======================= TASK 7: Gerlach removal refits =======================
def freq_matched(tokens, share):
    """Most frequent words, up to the same share of tokens as the Gerlach list."""
    counts = tokens.value_counts()
    cum = counts.cumsum() / counts.sum()
    n = int((cum < share).sum()) + 1
    return set(counts.index[:n])


def task7(n_boot=200):
    with open("results/gerlach_lists.json", encoding="utf-8") as f:
        lists = json.load(f)

    def removed_main(name):
        return without(load_corpus(name)["token"], set(lists[name][MAIN_KEY]))

    def removed_signed(name):
        return without(load_corpus(name)["token"], set(lists[name][COMPARE_KEY]))

    for label, fn in [("gerlach_abs_min10", removed_main), ("gerlach_signed", removed_signed)]:
        print(f"\nTASK 7: ZIPF after {label}")
        print(run_zipf(label, fn).to_string(index=False))
        print(f"\nTASK 7: HEAPS after {label}")
        print(run_heaps(label, fn).to_string(index=False))

    parts = []
    for name in CORPORA:
        df = load_corpus(name)
        main = set(lists[name][MAIN_KEY])
        share = df["token"].isin(main).mean()
        stop_sets = {
            "gerlach": main,
            "gerlach_signed": set(lists[name][COMPARE_KEY]),
            "freq_matched": freq_matched(df["token"], share),
        }
        parts.append(bootstrap_removal(df, name, stop_sets, n_boot=n_boot))
        print(f"bootstrap done: {name}")
    table = pd.concat(parts, ignore_index=True)
    table.to_csv("results/stopword_removal_tests_gerlach.csv", index=False)
    print("\nTASK 7: bootstrap test (R2 rows; the CSV has all metrics)")
    print(table[table["metric"].str.endswith("R2")].to_string(index=False))


# ======================= run =======================
STEPS = {
    "task4": task4,
    "task5": task5,
    "gerlach-stats": gerlach_stats,
    "gerlach-lists": gerlach_lists,
    "task7": task7,
}

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: python 03_stopwords.py <step> [<step> ...]")
        print("Steps:", ", ".join(STEPS), "| or: all")
        sys.exit(0)
    chosen = list(STEPS) if args == ["all"] else args
    for step in chosen:
        if step not in STEPS:
            sys.exit(f"Unknown step: {step}")
    for step in chosen:
        STEPS[step]()