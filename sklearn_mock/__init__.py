"""Mock sklearn for L104 compatibility without heavy dependencies."""

class MockEstimator:
    """Base mock estimator."""
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        import numpy as np
        return np.zeros(len(X))
    
    def score(self, X, y):
        return 0.5

# SVM module
class svm:
    """Mock SVM module."""
    class SVC(MockEstimator):
        def __init__(self, kernel='rbf', C=1.0, gamma='scale', **kwargs):
            super().__init__()
            self.kernel = kernel
            self.C = C
            self.gamma = gamma
    
    class SVR(MockEstimator):
        pass
    
    class OneClassSVM(MockEstimator):
        pass

# Clustering module
class cluster:
    """Mock clustering module."""
    class KMeans:
        def __init__(self, n_clusters=8, **kwargs):
            self.n_clusters = n_clusters
            self.labels_ = None
        
        def fit(self, X):
            import numpy as np
            self.labels_ = np.zeros(len(X), dtype=int)
            return self
    
    class DBSCAN:
        def __init__(self, eps=0.5, min_samples=5, **kwargs):
            self.eps = eps
            self.min_samples = min_samples
            self.labels_ = None
        
        def fit(self, X):
            import numpy as np
            self.labels_ = np.zeros(len(X), dtype=int)
            return self
    
    class SpectralClustering:
        def __init__(self, n_clusters=2, **kwargs):
            self.n_clusters = n_clusters
            self.labels_ = None
        
        def fit(self, X):
            import numpy as np
            self.labels_ = np.zeros(len(X), dtype=int)
            return self

# Metrics module
class metrics:
    """Mock metrics module."""
    @staticmethod
    def silhouette_score(X, labels, **kwargs):
        return 0.5
    
    @staticmethod
    def pairwise_distances(X, Y=None, metric='euclidean'):
        import numpy as np
        if Y is None:
            Y = X
        return np.sqrt(((X[:, None] - Y[None, :]) ** 2).sum(axis=2))

# Preprocessing module
class preprocessing:
    """Mock preprocessing module."""
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit(self, X):
            import numpy as np
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            return self
        
        def transform(self, X):
            return X
        
        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

# Ensemble module
class ensemble:
    """Mock ensemble module."""
    class RandomForestClassifier(MockEstimator):
        def __init__(self, n_estimators=100, **kwargs):
            super().__init__()
            self.n_estimators = n_estimators
    
    class RandomForestRegressor(MockEstimator):
        pass
    
    class GradientBoostingClassifier(MockEstimator):
        pass
    
    class GradientBoostingRegressor(MockEstimator):
        pass
    
    class ExtraTreesClassifier(MockEstimator):
        pass
    
    class AdaBoostClassifier(MockEstimator):
        pass
    
    class BaggingClassifier(MockEstimator):
        pass
    
    class VotingClassifier:
        def __init__(self, estimators, voting='hard', **kwargs):
            self.estimators = estimators
            self.voting = voting
        
        def fit(self, X, y):
            return self
        
        def predict(self, X):
            import numpy as np
            return np.zeros(len(X), dtype=int)

