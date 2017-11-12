import numpy as np
from .config import eps

def sharpe(returns, freq=30, rfr=0.0):
    """Given a set of returns, calculates naive (rfr=0) sharpe (eq 28) """
    return (np.sqrt(freq) * np.mean(returns - rfr)) / (np.std(returns - rfr) + eps)


def MDD(returns):
    """Max drawdown."""
    peak = returns.max()
    i = returns.argmax()
    trough = returns[i:].min()
    return (trough - peak) / trough


def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, -log_eps, log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist
