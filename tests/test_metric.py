import numpy as np
from mk12.metric import map3, ap3, map3_probs


def test_eval_tab_examples():
    for p in ['ALG-06 CAL-01 ANS-01', 'ALG-06 ALG-06 ALG-06', 'ALG-06 CAL-01 ALG-06']:
        assert ap3(p, 'ALG-06') == 1.0


def test_ranks():
    assert ap3('A B C', 'B') == 0.5
    assert abs(ap3('A B C', 'C') - 1 / 3) < 1e-12
    assert ap3('A B C D', 'D') == 0.0
    assert ap3('A A B', 'B') == 0.5  # repeat skipped
    assert ap3('XX A B', 'B', valid={'A', 'B'}) == 0.5  # unknown ignored
    assert map3(['A', 'B'], ['A', 'A']) == 0.5


def test_probs():
    P = np.array([[.5, .3, .2, 0], [.1, .2, .3, .4]])
    assert map3_probs(P, [1, 0]) == (0.5 + 0) / 2
