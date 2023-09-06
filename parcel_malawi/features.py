import scipy
import numpy as np


def tsteps(x, n_steps=6):
    return scipy.signal.resample(x, n_steps, axis=0)


def percentile_iqr(x, q=[10, 50, 90], iqr=[25, 75]):

    q_all = list(set(q + iqr))
    q_ids = [q_all.index(qv) for qv in q]
    iqr_ids = [q_all.index(qv) for qv in iqr]

    perc = np.percentile(x, q=q_all, axis=0)

    perc_arr = perc[q_ids]
    iqr_arr = perc[iqr_ids[1]] - perc[iqr_ids[0]]

    iqr_arr = np.expand_dims(iqr_arr, axis=0)

    arr = np.concatenate([perc_arr, iqr_arr], axis=0)

    return arr
