from rl_portfolio_management.data.utils import random_shift, normalize, scale_to_start
from rl_portfolio_management.util import sharpe, MDD, softmax

import pandas as pd
import numpy as np


def test_softmax():
    x = np.random.random((20, 20))
    y = softmax(x)
    np.testing.assert_almost_equal(y.sum(), 1)

    x = np.array([0, 1])
    y = softmax(x)
    assert y[0] < y[1]
    assert (y > 0).all()
    assert (y < 1).all()


def test_maxdrawdown():
    assert MDD(np.array([0, 0, 0, 0, 1, 2, 3])) == 0
    assert MDD(np.array([0, 0, 0, 0, 1, 2, 1])) == -1
    assert MDD(np.array([0, 0, 0, 3, 1, 2, 1])) == -2
    assert MDD(np.array([0, 0, 0, 3, 1, 2, 1, 1, 2])) == -2


def test_sharpe():
    x = np.random.normal(loc=1, scale=1, size=1000000)
    np.testing.assert_almost_equal(sharpe(
        x, freq=1), 1, decimal=2, err_msg='sharpe of normal dist with mean=1, std=1 should be ~1')

    x = np.random.normal(loc=1 / np.sqrt(30), scale=1, size=1000000)
    np.testing.assert_almost_equal(sharpe(
        x, freq=30), 1, decimal=2, err_msg='sharpe of normal dist with mean=1/sqrt(30), std=1, at freq of 30 should be ~1')


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
