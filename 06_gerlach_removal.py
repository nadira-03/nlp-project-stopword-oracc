import json
import pandas as pd
from common import CORPORA, load_corpus
from fitting import run_zipf, run_heaps, bootstrap_removal

with open("results/gerlach_lists.json", encoding="utf-8") as f:
    LISTS = json.load(f)

MAIN, COMPARE = "abs_min10_p0.2", "signed_p0.2"

def without(tokens, stop):
    return tokens[~tokens.isin(stop)]

def freq_matched(tokens, share):
    """Most frequent words, up to the same share of tokens as the Gerlach list."""
    counts = tokens.value_counts()
    cum = counts.cumsum() / counts.sum()
    n = int((cum < share).sum()) + 1
    return set(counts.index[:n])

def removed_main(name):
    return without(load_corpus(name)["token"], set(LISTS[name][MAIN]))

def removed_signed(name):
    return without(load_corpus(name)["token"], set(LISTS[name][COMPARE]))

def task7(n_boot=200):
    for label, fn in [("gerlach_abs_min10", removed_main), ("gerlach_signed", removed_signed)]:
        print(f"\nTASK 7: ZIPF after {label}")
        print(run_zipf(label, fn).to_string(index=False))
        print(f"\nTASK 7: HEAPS after {label}")
        print(run_heaps(label, fn).to_string(index=False))

    parts = []
    for name in CORPORA:
        df = load_corpus(name)
        main = set(LISTS[name][MAIN])
        share = df["token"].isin(main).mean()
        stop_sets = {
            "gerlach": main,
            "gerlach_signed": set(LISTS[name][COMPARE]),
            "freq_matched": freq_matched(df["token"], share),
        }
        parts.append(bootstrap_removal(df, name, stop_sets, n_boot=n_boot))
        print(f"bootstrap done: {name}")
    table = pd.concat(parts, ignore_index=True)
    table.to_csv("results/stopword_removal_tests_gerlach.csv", index=False)
    print("\nTASK 7: bootstrap test (R2 rows; the CSV has all metrics)")
    print(table[table["metric"].str.endswith("R2")].to_string(index=False))

if __name__ == "__main__":
    task7()