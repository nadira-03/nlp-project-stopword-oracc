import pandas as pd

for corpus in ["cams", "dcclt", "ribo", "rinap", "saao"]:
    df = pd.read_csv(f"data/processed/{corpus}.csv", keep_default_na=False)
    n = len(df)
    print("=" * 60)
    print(corpus.upper(), n, "tokens")
    print("single-character tokens: %.1f%%" % (100 * (df["token"].str.len() == 1).mean()))
    print("all-digit tokens:        %.1f%%" % (100 * df["token"].str.isdigit().mean()))
    print("token 'x':               %.1f%%" % (100 * (df["token"] == "x").mean()))
    print("\nTop 15 tokens:")
    print(df["token"].value_counts().head(15).to_string())
    print("\nTop 8 PoS tags:")
    print(df["pos"].value_counts().head(8).to_string())