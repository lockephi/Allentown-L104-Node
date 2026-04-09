"""Minimal sklearn mock for L104 ML Engine compatibility."""

class MockEstimator:
    """Base mock estimator."""
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return [0] * len(X)
    def score(self, X, y):
        return 0.5

class MockSVC(MockEstimator):
    """Mock SVC."""
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', **kwargs):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

class MockSVR(MockEstimator):
    """Mock SVR."""
    pass

class MockOneClassSVM(MockEstimator):
    """Mock OneClassSVM."""
    pass

# svm module
class SVC(MockSVC):
    pass
class SVR(MockSVR):
    pass
class OneClassSVM(MockOneClassSVM):
    pass

# ensemble module  
class RandomForestClassifier(MockEstimator):
    def __init__(self, n_estimators=100, **kwargs):
        self.n_estimators = n_estimators

class GradientBoostingClassifier(MockEstimator):
    pass

class AdaBoostClassifier(MockEstimator):
    pass

# cluster module
class KMeans(MockEstimator):
    def __init__(self, n_clusters=8, **kwargs):
        self.n_clusters = n_clusters
        self.labels_ = []

class DBSCAN(MockEstimator):
    def __init__(self, eps=0.5, **kwargs):
        self.eps = eps
        self.labels_ = []

class SpectralClustering(MockEstimator):
    pass

# preprocessing module
class StandardScaler:
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def fit_transform(self, X):
        return X

# metrics module
def silhouette_score(X, labels, **kwargs):
    return 0.5

def pairwise_distances(X, Y=None, metric='euclidean'):
    import numpy as np
    if Y is None:
        Y = X
    return np.zeros((len(X), len(Y)))
