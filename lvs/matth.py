# math functions

import numpy as np


# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
# moving average, or "moving mean"
def movavg(a, n=3, pad=False):
    # TODO: add padding into this
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    avg = ret[n - 1:] / n
    return avg
