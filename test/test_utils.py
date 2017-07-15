from src.data.utils import random_shift, normalize, scale_to_start

import pandas as pd
import numpy as np


def test_random_shift():
    s = pd.Series(np.random.random(100)) * 10 + 3
    s1 = random_shift(s.copy(), 0)
    assert (s == s1).all(), 'should not do anything if given 0'

    s2 = random_shift(s.copy(), 0.05)
    shift = (s2 / s)
    np.testing.assert_almost_equal(shift.mean(), 1.00, 2)
    np.testing.assert_almost_equal(shift.max(), 1.05, 2)


def test_normalize():
    s = pd.Series(np.random.random(10)) * 10 + 3
    s1 = normalize(s)
    np.testing.assert_almost_equal(s1.std(), 1, err_msg='should make std 0')
    np.testing.assert_almost_equal(s1.mean(), 0, err_msg='')


def test_scale_to_start():
    s = pd.Series(np.random.random(10))
    s1 = scale_to_start(s)
    assert s1[0] == 1, 'start should be 1'
    assert s1[1] != 1, 'not start should not be 1'
