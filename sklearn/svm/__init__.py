from .. import BaseEstimator, ClassifierMixin

class SVC(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0] * len(X) if hasattr(X, '__len__') else []
    def decision_function(self, X):
        return [0.0] * len(X) if hasattr(X, '__len__') else []

class SVR(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0.0] * len(X) if hasattr(X, '__len__') else []

class OneClassSVM(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X):
        self._fitted = True
        return self
    def predict(self, X):
        return [1] * len(X) if hasattr(X, '__len__') else []
