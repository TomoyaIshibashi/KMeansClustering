# -*- coding : UTF-8 -*-

import pandas as pd
import os
import glob
import openpyxl

from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels

from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import _check_sample_weight
from scipy.spatial.distance import cdist
import warnings
import numpy as np

try:
    from sklearn.cluster._kmeans import _kmeans_plusplus
except:
    try:
        from sklearn.cluster._kmeans import _k_init
        warnings.warn(
            "Scikit-learn <0.24 will be deprecated in a "
            "future release of tslearn"
        )
    except:
        from sklearn.cluster._kmeans import _k_init
        warnings.warn(
            "Scikit-learn <0.24 will be deprecated in a "
            "future release of tslearn"
        )
    
    def _kmeans_plusplus(*args, **kwargs):
        return _k_init(*args, **kwargs), None

from tslearn.metrics import cdist_gak, cdist_soft_dtw, sigma_gak, cdist_dtw # delete "cdist_dtw"
from tslearn.barycenters import euclidean_barycenter, \
    dtw_barycenter_averaging, softdtw_barycenter
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tslearn.utils import (to_time_series_dataset, to_sklearn_dataset, check_dims)
from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator

from tslearn.clustering.utils import (EmptyClusterError, _check_initial_guess, _check_no_empty_cluster,
                    _check_full_length, TimeSeriesCentroidBasedClusteringMixin, _compute_inertia)

#from tslearn.utils import to_time_series_dataset
from joblib import Parallel, delayed
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from matplotlib import pyplot as plt

__author__ = 'Romain Tavenard remain.tavenard[at]univ-rennnes2.fr'

"""
# 原音声のF0データ(smoothed)のフォルダのパスを取得
DirectorySession_original = '/Users/6ashi/Documents/MATLAB/balance_F02_LC/balance_F02_LC_N/FFS'
os.chdir(DirectorySession_original)

# 目的感情音声のF0データ(smoothed)のフォルダのパスを取得
DirectorySession_target = '/Users/6ashi/Documents/MATLAB/balance_F02_LC/balance_F02_LC_A/FFS'

# F0データファイルを読み込んで行列化
fileList = sorted(glob.glob('*.ffs'))
cnt = 0
df_original = pd.DataFrame().fillna(0)
df_target = pd.DataFrame().fillna(0)

for file in fileList:
  cnt += 1
  fnCompos = file.split('_')
  # balance_F02_LC_N_0001.ffs
  fileId_original = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_' + fnCompos[3] + '_' + fnCompos[4]
  fileId_target = fnCompos[0] + '_' + fnCompos[1] + '_' + fnCompos[2] + '_A_' + fnCompos[4]
  f = open(file, 'r')
  tsdata = pd.read_table(f, header=None)
  if 350 < len(tsdata) and len(tsdata) < 400:
    os.chdir(DirectorySession_target)
    if os.path.exists(fileId_target):
      df_original[fnCompos[4]] = tsdata
      g = open(fileId_target, 'r')
      tsdata_target = pd.read_table(g, header=None)
      df_target[fnCompos[4]] = tsdata_target
      g.close
  f.close
  os.chdir(DirectorySession_original)
  #if cnt == 500:
  #  break
"""

# 原音声のF0データ(smoothed)のフォルダのパスを取得
DirectorySession_original = '/Users/6ashi/Documents/MATLAB/EXL/EXL_N'

DirectorySession_target = '/Users/6ashi/Documents/MATLAB/EXL/EXL_A'


# データファイルを読み込んで行列化
os.chdir(DirectorySession_original)
df_original = pd.read_excel('F0data_N_select_1.xlsx')
#print(df_original)
os.chdir(DirectorySession_target)
df_target = pd.read_excel('F0data_A_select_1.xlsx')
#print(df_target)

dataset_original = np.array(df_original.T)
dataset_target = np.array(df_target.T)

n_jobs=None
verbose=0
compute_diagonal=False
global_constraint=None
sakoe_chiba_radius=None
itakura_max_slope=None

def _k_init_metric(X, Y, n_clusters, cdist_metric, random_state,
                   n_local_trials=None):
    """Init n_clusters seeds according to k-means++ with a custom distance
    metric.
    Parameters
    ----------
    X : array, shape (n_samples, n_timestamps, n_features)
        The data to pick seeds for.
    n_clusters : integer
        The number of seeds to choose
    cdist_metric : function
        Function to be called for cross-distance computations
    random_state : RandomState instance
        Generator used to initialize the centers.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version adapted from scikit-learn for use with a custom metric in place of
    Euclidean distance.
    """
    n_samples, n_timestamps, n_features = X.shape
    n_samples_y, n_timestamps_y, n_features_y = Y.shape


    centers = np.empty((n_clusters, n_timestamps, n_features),
                          dtype=X.dtype)
    centers_y = np.empty((n_clusters, n_timestamps_y, n_features_y),
                          dtype=Y.dtype)

    #print(centers, centers_y)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples) # warn this
    center_id_y = random_state.randint(n_samples_y)
    centers[0] = X[center_id]
    centers_y[0] = Y[center_id_y]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = ((cdist_metric(centers[0, np.newaxis], X)+ cdist_metric(centers_y[0, np.newaxis], Y)) / 2 ) ** 2
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                           rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                   out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = ((cdist_metric(X[candidate_ids], X) + cdist_metric(Y[candidate_ids], Y)) / 2 ) **2

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        centers_y[c] = Y[best_candidate]

    return centers, centers_y

def _cdist_generic(dist_fun, dataset1, dataset2, n_jobs, verbose,
                   compute_diagonal=True, dtype=float, *args, **kwargs):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.
    Parameters
    ----------
    dist_fun : function
        Similarity function to be used
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.
    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.
    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.
    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix
    """  # noqa: E501
    dataset1 = to_time_series_dataset(dataset1, dtype=dtype)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = np.zeros((len(dataset1), len(dataset1)))
        indices = np.triu_indices(len(dataset1),
                                     k=0 if compute_diagonal else 1,
                                     m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs,
                                   prefer="threads",
                                   verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset1[j],
                *args, **kwargs
            )
            for i in range(len(dataset1))
            for j in range(i if compute_diagonal else i + 1,
                           len(dataset1))
        )
        indices = np.tril_indices(len(dataset1), k=-1, m=len(dataset1)) # 上三角行列
        matrix[indices] = matrix.T[indices]
        return matrix
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=dtype)
        matrix = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(dist_fun)(
                dataset1[i], dataset2[j],
                *args, **kwargs
            )
            for i in range(len(dataset1)) for j in range(len(dataset2))
        )
        return np.array(matrix).reshape((len(dataset1), -1))

"""
def cdist_dtw(dataset_centroid, dataset_original, dataset_target, n_jobs=None, verbose=0,
                   compute_diagonal=True, dtype=float, *args, **kwargs):

  # dataset1 : Centroid
  # dataset2 : original ff's dataset
  # dataset3 : target ff's dataset

    matrix_original = _cdist_generic(dist_fun=dtw, dataset1=dataset_original, dataset2=dataset_centroid,
                            n_jobs=n_jobs, verbose=verbose,
                            compute_diagonal=True,
                            global_constraint=None,
                            sakoe_chiba_radius=None,
                            itakura_max_slope=None)

    matrix_target = _cdist_generic(dist_fun=dtw, dataset1=dataset_target, dataset2=dataset_centroid,
                            n_jobs=n_jobs, verbose=verbose,
                            compute_diagonal=True,
                            global_constraint=None,
                            sakoe_chiba_radius=None,
                            itakura_max_slope=None)

    return (np.array(matrix_original) + np.array(matrix_target)) / 2

"""

class TimeSeriesKMeans(TransformerMixin, ClusterMixin,
                       TimeSeriesCentroidBasedClusteringMixin,
                       BaseModelPackage, TimeSeriesBaseEstimator):
    """K-means clustering for time-series data.
    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia.
    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter
        computation. If "dtw", DBA is used for barycenter
        computation.
    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used
        if `metric="dtw"` or `metric="softdtw"`.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    dtw_inertia: bool (default: False)
        Whether to compute DTW inertia even if DTW is not the chosen metric.
    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.
    init : {'k-means++', 'random' or an ndarray} (default: 'k-means++')
        Method for initialization:
        'k-means++' : use k-means++ heuristic. See `scikit-learn's k_init_
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/\
        cluster/k_means_.py>`_ for more.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.
    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.
    cluster_centers_ : numpy.ndarray of shape (n_clusters, sz, d)
        Cluster centers.
        `sz` is the size of the time series used at fit time if the init method
        is 'k-means++' or 'random', and the size of the longest initial
        centroid if those are provided as a numpy array through init parameter.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    n_iter_ : int
        The number of iterations performed during fit.
    Notes
    -----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of
        equal-sized time series.
    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,
    ...                       random_state=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 32, 1)
    >>> km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,
    ...                           max_iter_barycenter=5,
    ...                           random_state=0).fit(X)
    >>> km_dba.cluster_centers_.shape
    (3, 32, 1)
    >>> km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,
    ...                            max_iter_barycenter=5,
    ...                            metric_params={"gamma": .5},
    ...                            random_state=0).fit(X)
    >>> km_sdtw.cluster_centers_.shape
    (3, 32, 1)
    >>> X_bis = to_time_series_dataset([[1, 2, 3, 4],
    ...                                 [1, 2, 3],
    ...                                 [2, 5, 6, 7, 8, 9]])
    >>> km = TimeSeriesKMeans(n_clusters=2, max_iter=5,
    ...                       metric="dtw", random_state=0).fit(X_bis)
    >>> km.cluster_centers_.shape
    (2, 6, 1)
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1,
                 metric="euclidean", max_iter_barycenter=100,
                 metric_params=None, n_jobs=None, dtw_inertia=False,
                 verbose=0, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters # クラスタ数
        self.max_iter = max_iter # 最大試行回数
        self.tol = tol 
        self.n_init = n_init 
        self.metric = metric # 距離関数
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.dtw_inertia = dtw_inertia
        self.verbose = verbose
        self.random_state = random_state # 確率変数の再現性を設定
        self.init = init

    def _is_fitted(self):
        check_is_fitted(self, ['cluster_centers_']) # 推定値が適合しているか検証(?)
        return True

    def _get_metric_params(self):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"] # 削除
        return metric_params

    #def _fit_one_init(self, X, x_squared_norms, rs):
    def _fit_one_init(self, X, Y, x_squared_norms, rs): # 3rd
        metric_params = self._get_metric_params()
        n_ts, sz, d = X.shape
        #n_ts_y, sz_y, d_y = Y.shape
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif isinstance(self.init, str) and self.init == "k-means++":
            if self.metric == "euclidean":
                self.cluster_centers_ = _kmeans_plusplus(
                    X.reshape((n_ts, -1)),
                    self.n_clusters,
                    x_squared_norms=x_squared_norms,
                    random_state=rs
                )[0].reshape((-1, sz, d))
            else:
                if self.metric == "dtw": # 4th
                    def metric_fun(x, y):
                        return cdist_dtw(x, y, n_jobs=self.n_jobs,
                                         verbose=self.verbose, **metric_params) # compute distance

                elif self.metric == "softdtw":
                    def metric_fun(x, y):
                        return cdist_soft_dtw(x, y, **metric_params)
                else:
                    raise ValueError(
                        "Incorrect metric: %s (should be one of 'dtw', "
                        "'softdtw', 'euclidean')" % self.metric
                    )
                self.cluster_centers_, self.cluster_centers_y_ = _k_init_metric(X, Y, self.n_clusters,
                                                       cdist_metric=metric_fun,
                                                       random_state=rs)
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init'"
                             "is invalid" % self.init)
        self.cluster_centers_ = _check_full_length(self.cluster_centers_)
        self.cluster_centers_y_ = _check_full_length(self.cluster_centers_y_)
        old_inertia = np.inf

        for it in range(self.max_iter):
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")
            self._update_centroids(X)

            if np.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def _transform(self, X):
        metric_params = self._get_metric_params()
        if self.metric == "euclidean":
            return cdist(X.reshape((X.shape[0], -1)),
                         self.cluster_centers_.reshape((self.n_clusters, -1)),
                         metric="euclidean")
        elif self.metric == "dtw":
            return cdist_dtw(X, self.cluster_centers_, n_jobs=self.n_jobs,
                             verbose=self.verbose, **metric_params)
        elif self.metric == "softdtw":
            return cdist_soft_dtw(X, self.cluster_centers_, **metric_params)
        else:
            raise ValueError("Incorrect metric: %s (should be one of 'dtw', "
                             "'softdtw', 'euclidean')" % self.metric)

    def _assign(self, X, update_class_attributes=True):
        dists = self._transform(X)
        matched_labels = dists.argmin(axis=1)
        if update_class_attributes:
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            if self.dtw_inertia and self.metric != "dtw":
                inertia_dists = cdist_dtw(X, self.cluster_centers_,
                                          n_jobs=self.n_jobs,
                                          verbose=self.verbose)
            else:
                inertia_dists = dists
            self.inertia_ = _compute_inertia(inertia_dists,
                                             self.labels_,
                                             self._squared_inertia)
        return matched_labels

    def _update_centroids(self, X):
        metric_params = self._get_metric_params()
        for k in range(self.n_clusters):
            if self.metric == "dtw":
                self.cluster_centers_[k] = dtw_barycenter_averaging(
                    X=X[self.labels_ == k],
                    barycenter_size=None,
                    init_barycenter=self.cluster_centers_[k],
                    metric_params=metric_params,
                    verbose=False)
            elif self.metric == "softdtw":
                self.cluster_centers_[k] = softdtw_barycenter(
                    X=X[self.labels_ == k],
                    max_iter=self.max_iter_barycenter,
                    init=self.cluster_centers_[k],
                    **metric_params)
            else:
                self.cluster_centers_[k] = euclidean_barycenter(
                    X=X[self.labels_ == k])

    def fit(self, X, Y, y=None): # 2nd
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Y : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y
            Ignored
        """

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        Y = check_array(Y, allow_nd=True, force_all_finite='allow-nan')

        if hasattr(self.init, '__array__'):
            X = check_dims(X, X_fit_dims=self.init.shape,
                           extend=True,
                           check_n_features_only=(self.metric != "euclidean"))
            Y = check_dims(Y, Y_fit_dims=self.init.shape,
                           extend=True,
                           check_n_features_only=(self.metric != "euclidean"))

        self.labels_ = None
        self.inertia_ = np.inf
        self.cluster_centers_ = None
        self._X_fit = None
        self._Y_fit = None
        self._squared_inertia = True

        self.n_iter_ = 0

        max_attempts = max(self.n_init, 10)

        X_ = to_time_series_dataset(X)
        Y_ = to_time_series_dataset(Y)
        rs = check_random_state(self.random_state)

        if isinstance(self.init, str) and self.init == "k-means++" and \
                        self.metric == "euclidean":
            n_ts, sz, d = X_.shape
            n_ts_y, sz_y, d_y = Y_.shape
            x_squared_norms = cdist(X_.reshape((n_ts, -1)),
                                    np.zeros((1, sz * d)),
                                    metric="sqeuclidean").reshape((1, -1))
            y_squared_norms = cdist(Y_.reshape((n_ts_y, -1)),
                                    np.zeros((1, sz_y * d_y)),
                                    metric="sqeuclidean").reshape((1, -1))
        else:
            x_squared_norms = None
            y_squared_norms = None
        _check_initial_guess(self.init, self.n_clusters)

        best_correct_centroids = None
        min_inertia = np.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                #self._fit_one_init(X_, x_squared_norms, rs)
                self._fit_one_init(X_, Y_, x_squared_norms, rs) #######
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, Y, y=None): # 1st
        """Fit k-means clustering using X, Y and then predict the closest cluster
        each time series in X belongs to.
        It is more efficient to use this method than to sequentially call fit
        and predict.
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Y : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored
        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        Y = check_array(Y, allow_nd=True, force_all_finite='allow-nan')
        return self.fit(X, Y, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.
        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._assign(X, update_class_attributes=False)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster 
        centers.
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset
        Returns
        -------
        distances : array of shape=(n_ts, n_clusters)
            Distances to cluster centers
        """
        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, X_fit_dims=self.cluster_centers_.shape,
                       extend=True,
                       check_n_features_only=(self.metric != "euclidean"))
        return self._transform(X)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}

# 前処理として標準化を行う
nm_data = TimeSeriesScalerMeanVariance().fit_transform(dataset_original)
nm_data_y = TimeSeriesScalerMeanVariance().fit_transform(dataset_target)

nm_data_np = np.array(nm_data)
nm_data_y_np = np.array(nm_data_y)

# クラスタ数
n=5

# 標準化データのプロットレンジ
nm_ymax = 5
nm_ymin = -5


# クラスタリング
km_dtw = TimeSeriesKMeans(n_clusters=n, max_iter=300, random_state=42, metric="dtw")
labels_dtw = km_dtw.fit_predict(nm_data, nm_data_y)

"""
# 結果の表示
x1 = df_original.T
df = pd.DataFrame(x1.iloc[:,0])
cl = labels_dtw.tolist() # クラスタ番号を配列にする
df.insert(1, '1', cl) # クラスタ番号を追加
pd.set_option('display.max_rows', None) # 表示サイズを調節
pd.set_option('display.max_columns', None) # 表示サイズを調節
df.columns = ['data', 'cluster']
os.chdir('/Users/6ashi/Documents/Python_Code')
#df.drop('data', axis=1).to_excel('dataframe_cluster.xlsx') # ClusterNumberと音声データ番号のDataFrameを出力
#df.drop('data', axis=1).query('cluster == 0').to_excel('df_cluster1.xlsx') # cluster X を出力
#df.drop('data', axis=1).query('cluster == 1').to_excel('df_cluster2.xlsx')
#df.drop('data', axis=1).query('cluster == 2').to_excel('df_cluster3.xlsx')
#df.drop('data', axis=1).query('cluster == 3').to_excel('df_cluster4.xlsx')
#df.drop('data', axis=1).query('cluster == 4').to_excel('df_cluster5.xlsx')
"""

"""
# クラスタ内のファイル一覧を取得
idx = df.drop('data', axis=1).query('cluster == 0').index
f = open('./try.txt', 'w')
for i in range(len(idx)):
    f.write(str(idx[i])+'\n')
f.close()
"""


# 標準化したデータのプロット
fig, axes = plt.subplots(n, figsize=(8.0, 18.0))
plt.subplots_adjust(hspace=0.5)
for i in range(n):
    ax = axes[i]
    for xx in nm_data[labels_dtw == i]:
        # データのプロット
        #print(nm_data[labels_dtw == i])
        ax.plot(xx.ravel(), "k-", alpha=.2)

    # 重心のプロット
    ax.plot(km_dtw.cluster_centers_[i].ravel(), "r-")
    #(km_dtw.cluster_centers_[i].ravel())

    #""
    # クラスタ重心のデータを出力
    os.chdir('/Users/6ashi/Documents/Python_Code')
    fname = 'N_centroid_G1_' + str(i+1) + '.txt'
    #print(fname)
    f = open(fname, 'w') 
    np.set_printoptions(linewidth=5)
    #print(km_dtw.cluster_centers_[i].ravel())
    data_centroid = list(km_dtw.cluster_centers_[i].ravel())
    for num in range(len(data_centroid)):
        f.write(str(data_centroid[num]) + '\n')
    f.close()
    #""

    #print(km_dtw.cluster_centers_[i].ravel(), "r-")
    # 軸の設定とテキストの表示
    ax.set_xlim(0, 300)
    ax.set_ylim(nm_ymin, nm_ymax)
    datanum = np.count_nonzero(labels_dtw == i)
    ax.text(0.5, (nm_ymax*0.9+nm_ymin*0.1), f'Cluster {(i + 1)} : n = {datanum}')
    if i == 0:
        ax.set_title("DTW k-means smoothed original F0 (z-score normalization)")
#plt.show()


"""
# 元データのプロット
fig, axes = plt.subplots(n, figsize=(8.0, 18.0))
plt.subplots_adjust(hspace=0.5)
for i in range(n):
    ax = axes[i]
    # データのプロット
    for xx in dataset_original[labels_dtw == i]:
        ax.plot(xx.ravel(), "k-", alpha=.2)
    # 軸の設定とテキストの表示
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 500)
    datanum = np.count_nonzero(labels_dtw == i)
    ax.text(0.5, (nm_ymax*0.9+nm_ymin*0.1), f'Cluster {(i + 1)} : n = {datanum}')
    if i == 0:
        ax.set_title("DTW k-means smoothed original F0")

plt.show()
"""

"""
fig, axes = plt.subplots(n, figsize=(8.0, 18.0))
plt.subplots_adjust(hspace=0.5)
for i in range(n):
    ax = axes[i]
    # データのプロット
    #print(data[labels_dtw == i])
    for xx in nm_data_y[labels_dtw == i]:
        ax.plot(xx.ravel(), "k-", alpha=.2)
    # 重心のプロット
    ax.plot(km_dtw.cluster_centers_y_[i].ravel(), "r-")
    # 軸の設定とテキストの表示
    ax.set_xlim(0, 500)
    ax.set_ylim(nm_ymin, nm_ymax)
    datanum = np.count_nonzero(labels_dtw == i)
    ax.text(0.5, (nm_ymax*0.9+nm_ymin*0.1), f'Cluster {(i + 1)} : n = {datanum}')
    if i == 0:
        ax.set_title("DTW k-means Target F0 (z-score normalization)")
plt.show()
"""


# 1~10クラスタまで一気に計算
# elbow法

"""
distortions = []
for i in range(1,6):
    km = TimeSeriesKMeans(n_clusters=i,
                max_iter=300,
                random_state=42,
                metric="dtw")
    # クラスタリングの計算を実行
    km.fit_predict(nm_data, nm_data_y)
    distortions.append(km.inertia_) # km.fitするとkm.inertia_が得られる

plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
"""
