from common import load_corpus
from fitting import run_zipf, run_heaps

get_tokens = lambda name: load_corpus(name)["token"]

print("ZIPF")
print(run_zipf("clean", get_tokens).to_string(index=False))
print()
print("HEAPS")
print(run_heaps("clean", get_tokens).to_string(index=False))