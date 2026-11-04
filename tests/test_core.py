import numpy as np
import pytest

from captor import BuildConfig, CAPTOR, Corpus, build_table, load_table, save_table
from captor.model import patch_scores
from captor.propagation import affinity
from captor.signatures import (difference_signature, membership_phrase,
                               membership_substring, permutation_null, word_forms)
from captor.vocabulary import assign, joint_codes, soft_assign, volumes


def test_table_roundtrip(tmp_path, toy_table):
    save_table(toy_table, tmp_path / "t.npz")
    t = load_table(tmp_path / "t.npz")
    assert t.terms == toy_table.terms
    np.testing.assert_array_equal(t.signatures, toy_table.signatures)
    with pytest.raises(KeyError):
        t.term_index(["missing"])


def test_nearest_centroid_matches_sklearn():
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8)).astype(np.float32)
    km = KMeans(n_clusters=7, n_init=1, random_state=0).fit(x)
    np.testing.assert_array_equal(assign(x, km.cluster_centers_), km.predict(x))


def test_soft_with_one_word_is_hard(toy_table):
    x = np.random.default_rng(2).normal(size=(30, 8)).astype(np.float32)
    nearest, idx, w = soft_assign(x, toy_table.centroids[0], 1, 10.0)
    np.testing.assert_array_equal(idx[:, 0], assign(x, toy_table.centroids[0]))
    assert np.all(w == 1)
    _, idx, w = soft_assign(x, toy_table.centroids[0], 3, 10.0)
    np.testing.assert_allclose(w.sum(1), 1.0, rtol=1e-6)
    assert np.all(np.isin(nearest, idx[np.arange(30), w.argmax(1)]))


def test_hard_readout_is_a_lookup(toy_table):
    tok = np.random.default_rng(3).normal(size=(2, 9, 8)).astype(np.float32)
    out = patch_scores(tok, toy_table)
    exp = np.mean([toy_table.signatures[j][assign(tok, toy_table.centroids[j])]
                   for j in range(2)], 0)
    np.testing.assert_allclose(out, exp, rtol=1e-6)


def test_trust_infinite_lambda_returns_average(toy_table):
    tok = np.random.default_rng(4).normal(size=(1, 9, 8)).astype(np.float32)
    base = patch_scores(tok, toy_table)
    codes = joint_codes([assign(tok[0], c) for c in toy_table.centroids], 5)
    toy_table.joint_cells = np.unique(codes)
    toy_table.joint_counts = np.full(len(toy_table.joint_cells), 4.0, np.float32)
    toy_table.joint_signatures = np.ones((len(toy_table.joint_cells), 3), np.float32)
    toy_table.meta["readout"]["trust_lambda"] = 1e12
    np.testing.assert_allclose(patch_scores(tok, toy_table), base, atol=1e-6)
    toy_table.meta["readout"]["trust_lambda"] = 4.0     # gamma = 1/2
    np.testing.assert_allclose(patch_scores(tok, toy_table), 0.5 * base + 0.5,
                               atol=1e-6)


def test_affinity_is_row_stochastic():
    tok = np.random.default_rng(5).normal(size=(25, 8)).astype(np.float32)
    W = affinity(tok)
    np.testing.assert_allclose(W.sum(1), 1.0, rtol=1e-5)
    assert np.all(np.diag(W) > 0)


def test_phrase_matching_consumes_longer_names():
    terms = ["bear", "teddy bear", "dog", "hot dog"]
    forms = {t: word_forms(t) for t in terms}
    forms["teddy bear"] = ("teddy bear", "teddy bears", "bear", "bears", "teddy")
    texts = ["a teddy bear on a bed", "two bears in a river", "a hot dog with mustard",
             "a dog runs"]
    M = membership_phrase(terms, texts, forms)
    assert M.tolist() == [[False, True, False, False], [True, False, False, False],
                          [False, False, False, True], [False, False, True, False]]
    assert membership_substring(["tumor"], ["Tumoral cells", "stroma"]).tolist() \
        == [[True, False]]


def test_signature_and_null(toy_corpus):
    tokens, texts, proto = toy_corpus
    lab = assign(tokens, proto)
    V = volumes(lab, 3)
    member = membership_substring(["cat", "dog"], texts)
    S = difference_signature(V, member)
    assert S[:, 0].argmax() == 0 and S[:, 1].argmax() == 1
    null = permutation_null(V, member, n_perm=50)
    assert np.all(np.abs(S).max(0) > null)


@pytest.mark.parametrize("preset", ["natural", "pathology"])
def test_build_segment_add_terms(toy_corpus, preset):
    tokens, texts, proto = toy_corpus
    cfg = BuildConfig.preset(preset)
    cfg.words, cfg.vocabs, cfg.n_perm, cfg.min_mentions = 3, 2, 20, 10
    cfg.soft_m = min(cfg.soft_m, 2)
    corpus = Corpus(tokens=tokens, texts=texts)
    centroids = np.stack([proto, proto])
    table = build_table(corpus, ["cat", "dog"], "none", cfg, centroids=centroids,
                        log=lambda s: None)
    model = CAPTOR(table)
    sc = model.patch_scores(tokens[:3])
    lab = sc.argmax(-1)
    assert np.all(lab[0, :8] == 0) and np.all(lab[1, :8] == 1)
    before = model.table
    model.add_terms(["room", "room", "giraffe"], corpus)
    assert model.terms == ["cat", "dog", "room"]       # deduplicated, unnamed dropped
    assert before.terms == ["cat", "dog"]               # the original is not changed
    assert model.patch_scores(tokens[2:3], ["room"]).shape == (1, 16, 1)
