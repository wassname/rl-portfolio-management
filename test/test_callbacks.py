import pytest
import numpy as np

import os
os.sys.path.append('.')
from rl_portfolio_management.callbacks.notebook_plot import LivePlotNotebook

def test_live_plot_notebook():
    # Test
    import time
    liveplot = LivePlotNotebook(labels=['a', 'b'])
    x = np.random.random((10,))
    for i in range(10):
        time.sleep(0.1)
        liveplot.update(
            x=x + np.random.random(x.shape),
            ys=[np.random.randint(0, 3, size=(10,)), np.random.randint(0, 3, size=(10,))]
        )
