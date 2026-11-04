import numpy as np
import pytest

from captor.io import Table


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


@pytest.fixture
def toy_corpus():
    """Images made of two kinds of patch; captions name what each image holds."""
    rng = np.random.default_rng(0)
    D, P, N = 16, 16, 120
    proto = _unit(rng.normal(size=(3, D)))          # cat, dog, background
    tokens, texts = [], []
    for i in range(N):
        kind = i % 3
        lab = np.full(P, 2)
        if kind < 2:
            lab[: P // 2] = kind
        tokens.append(_unit(proto[lab] + 0.05 * rng.normal(size=(P, D))))
        texts.append(["a cat on a sofa", "a dog in a park", "an empty room"][kind])
    return np.stack(tokens), texts, proto


@pytest.fixture
def toy_table():
    rng = np.random.default_rng(1)
    V, K, D, T = 2, 5, 8, 3
    return Table(terms=["a", "b", "c"], centroids=_unit(rng.normal(size=(V, K, D))),
                 signatures=rng.normal(size=(V, K, T)).astype(np.float32),
                 corpus_mean=np.full((V, K), 0.2, np.float32),
                 corpus_sd=np.full((V, K), 0.1, np.float32),
                 n_mentions=np.array([10, 20, 30]),
                 meta={"encoder": "none", "readout": {"soft_m": 1, "affinity": False}})
