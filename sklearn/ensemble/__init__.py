from .. import BaseEstimator, ClassifierMixin

class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0] * len(X) if hasattr(X, '__len__') else []

class RandomForestRegressor(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0.0] * len(X) if hasattr(X, '__len__') else []

class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0] * len(X) if hasattr(X, '__len__') else []

class GradientBoostingRegressor(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0.0] * len(X) if hasattr(X, '__len__') else []

class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0] * len(X) if hasattr(X, '__len__') else []

class AdaBoostRegressor(BaseEstimator):
    def __init__(self, **kwargs):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        return self
    def predict(self, X):
        return [0.0] * len(X) if hasattr(X, '__len__') else []
