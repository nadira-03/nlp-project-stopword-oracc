# Stopword behaviour in ORACC cuneiform corpora

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Put the ORACC `.vrt` files (cams, dcclt, ribo, rinap, saao) anywhere under `data/raw/`.

## Run order (from the project root)

1. `python 01_extract_and_stats.py` (task 1)
2. `python 02_zipf_heaps.py` (tasks 2, 3)
3. `python 03_stopwords.py` (tasks 4, 5)
4. `python 03_stopwords.py gerlach` (task 6, statistics, slow)
5. `python 03_stopwords.py lists` (task 6, stopword lists)
6. `python 03_stopwords.py task7` (task 7)
7. `python 04_pos_profiles.py` (task 8)
8. `python 05_oov.py` (task 9)

Results (figures and tables) are written to `results/`.
Steps 5 to 7 need step 4, and steps 6 and 7 also need step 5.
