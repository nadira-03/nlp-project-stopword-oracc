import numpy as np
from scipy.sparse import csr_matrix

def texts_nwd_csr(list_texts):
    """Build the word x document count matrix from a list of documents
    (each a list of words). Words are indexed in sorted order."""
    vocab = sorted({w for doc in list_texts for w in doc})
    dict_w_iw = {w: i for i, w in enumerate(vocab)}
    rows, cols = [], []
    for d, doc in enumerate(list_texts):
        rows.extend(dict_w_iw[w] for w in doc)
        cols.extend([d] * len(doc))
    data = np.ones(len(rows), dtype=np.int64)
    n_wd = csr_matrix((data, (rows, cols)), shape=(len(vocab), len(list_texts)))
    return n_wd, dict_w_iw