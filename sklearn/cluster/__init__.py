from .. import BaseEstimator

class KMeans(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X):
        self._fitted = True
        return self
    def predict(self, X):
        return [0] * len(X) if hasattr(X, '__len__') else []
    @property
    def labels_(self):
        return []
    @property
    def cluster_centers_(self):
        return []

class DBSCAN(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X):
        self._fitted = True
        return self
    @property
    def labels_(self):
        return []

class SpectralClustering(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X):
        self._fitted = True
        return self
    @property
    def labels_(self):
        return []
