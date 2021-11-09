"""Adapted from matplotlib"""
# from matplotlib.cbook import _reshape_2D
import dask.array
import numpy as np

import itertools
import warnings

from dwd_dl import log


def boxplot_stats(X, whis=1.5, bootstrap=None, labels=None,
                  autorange=False):
    r"""
    Return a list of dictionaries of statistics used to draw a series of box
    and whisker plots using `~.Axes.bxp`.

    Parameters
    ----------
    X : array-like
        Data that will be represented in the boxplots. Should have 2 or
        fewer dimensions.

    whis : float or (float, float), default: 1.5
        The position of the whiskers.

        If a float, the lower whisker is at the lowest datum above
        ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum below
        ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and third
        quartiles.  The default value of ``whis = 1.5`` corresponds to Tukey's
        original definition of boxplots.

        If a pair of floats, they indicate the percentiles at which to draw the
        whiskers (e.g., (5, 95)).  In particular, setting this to (0, 100)
        results in whiskers covering the whole range of the data.  "range" is
        a deprecated synonym for (0, 100).

        In the edge case where ``Q1 == Q3``, *whis* is automatically set to
        (0, 100) (cover the whole range of the data) if *autorange* is True.

        Beyond the whiskers, data are considered outliers and are plotted as
        individual points.

    bootstrap : int, optional
        Number of times the confidence intervals around the median
        should be bootstrapped (percentile method).

    labels : array-like, optional
        Labels for each dataset. Length must be compatible with
        dimensions of *X*.

    autorange : bool, optional (False)
        When `True` and the data are distributed such that the 25th and 75th
        percentiles are equal, ``whis`` is set to (0, 100) such that the
        whisker ends are at the minimum and maximum of the data.

    Returns
    -------
    list of dict
        A list of dictionaries containing the results for each column
        of data. Keys of each dictionary are the following:

        ========   ===================================
        Key        Value Description
        ========   ===================================
        label      tick label for the boxplot
        mean       arithmetic mean value
        med        50th percentile
        q1         first quartile (25th percentile)
        q3         third quartile (75th percentile)
        cilo       lower notch around the median
        cihi       upper notch around the median
        whislo     end of the lower whisker
        whishi     end of the upper whisker
        fliers     outliers
        ========   ===================================

    Notes
    -----
    Non-bootstrapping approach to confidence interval uses Gaussian-based
    asymptotic approximation:

    .. math::

        \mathrm{med} \pm 1.57 \times \frac{\mathrm{iqr}}{\sqrt{N}}

    General approach from:
    McGill, R., Tukey, J.W., and Larsen, W.A. (1978) "Variations of
    Boxplots", The American Statistician, 32:12-16.
    """

    def _bootstrap_median(data, N=5000):
        # determine 95% confidence intervals of the median
        M = len(data)
        percentiles = [2.5, 97.5]

        bs_index = np.random.randint(M, size=(N, M))
        bsData = data[bs_index]
        estimate = dask.array.median(bsData, axis=1, overwrite_input=True)

        CI = dask.array.percentile(estimate, percentiles)
        return CI

    def _compute_conf_interval(data, med, iqr, bootstrap):
        if bootstrap is not None:
            # Do a bootstrap estimate of notch locations.
            # get conf. intervals around median
            CI = _bootstrap_median(data, N=bootstrap)
            notch_min = CI[0]
            notch_max = CI[1]
        else:

            N = data.size
            notch_min = med - 1.57 * iqr / np.sqrt(N)
            notch_max = med + 1.57 * iqr / np.sqrt(N)

        return notch_min, notch_max

    # output is a list of dicts
    bxpstats = []

    # # convert X to a list of lists
    # X = _reshape_2D(X, "X")

    ncols = len(X)
    if labels is None:
        labels = itertools.repeat(None)
    elif len(labels) != ncols:
        raise ValueError("Dimensions of labels and X must be compatible")

    input_whis = whis
    for ii, (x, label) in enumerate(zip(X, labels)):

        # empty dict
        stats = {}
        if label is not None:
            stats['label'] = label

        # restore whis to the input values in case it got changed in the loop
        whis = input_whis

        # note tricksiness, append up here and then mutate below
        bxpstats.append(stats)

        # # if empty, bail
        # if len(x) == 0:
        #     stats['fliers'] = np.array([])
        #     stats['mean'] = np.nan
        #     stats['med'] = np.nan
        #     stats['q1'] = np.nan
        #     stats['q3'] = np.nan
        #     stats['cilo'] = np.nan
        #     stats['cihi'] = np.nan
        #     stats['whislo'] = np.nan
        #     stats['whishi'] = np.nan
        #     stats['med'] = np.nan
        #     continue

        # up-convert to an array, just to be safe
        assert isinstance(x, dask.array.Array)

        # arithmetic mean
        stats['mean'] = dask.array.mean(x)

        # medians and quartiles
        q1, med, q3 = dask.array.percentile(x, [25, 50, 75])

        # interquartile range
        q3.compute()
        q1.compute()
        stats['iqr'] = q3 - q1
        max_min = dask.array.max(x).compute() - dask.array.min(x).compute()
        log.info("Interquartile range is %s and max_min is %s", stats["iqr"], max_min)
        if stats['iqr']/max_min < 0.05 and autorange:
            whis = (0, 100)

        # conf. interval around median
        log.info("Starting computation of confidence interval around median. bootstrap is %s", bootstrap)
        stats['cilo'], stats['cihi'] = _compute_conf_interval(
            x, med, stats['iqr'], bootstrap
        )

        # lowest/highest non-outliers
        log.info("Starting computation of lowest/highest non-outliers")
        if np.isscalar(whis):
            if np.isreal(whis):
                loval = q1 - whis * stats['iqr']
                hival = q3 + whis * stats['iqr']
            elif whis in ['range', 'limit', 'limits', 'min/max']:
                warnings.warn(message=DeprecationWarning(f"Setting whis to {whis!r} is deprecated "
                                                         "since %(since)s and support for it will be removed "
                                                         "%(removal)s; set it to [0, 100] to achieve the same "
                                                         "effect.")
                              )
                loval = dask.array.min(x)
                hival = dask.array.max(x)
            else:
                raise ValueError('whis must be a float or list of percentiles')
        else:
            loval, hival = dask.array.percentile(x, whis)

        # get high extreme
        log.info("Getting high extreme.")
        wiskhi = x.where(x <= hival)
        max_wiskhi = dask.array.max(wiskhi).compute()
        if wiskhi.size == 0 or max_wiskhi < q3:
            stats['whishi'] = q3
        else:
            stats['whishi'] = max_wiskhi

        # get low extreme
        log.info("Getting low extreme.")
        wisklo = x.where(x >= loval)
        min_wisklo = dask.array.min(wisklo).compute()
        if wisklo.size == 0 or min_wisklo > q1:
            stats['whislo'] = q1
        else:
            stats['whislo'] = min_wisklo

        # compute a single array of outliers
        log.info("Computing outliers.")
        stats['fliers'] = dask.array.hstack([
            x[x < stats['whislo']],
            x[x > stats['whishi']],
        ])

        # add in the remaining stats
        stats['q1'], stats['med'], stats['q3'] = q1, med, q3

    return bxpstats