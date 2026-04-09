"""Stub sklearn module for systems without scikit-learn installed."""
import warnings
warnings.warn("Using stub sklearn - install scikit-learn for full ML functionality", ImportWarning)

class BaseEstimator:
    pass

class TransformerMixin:
    pass

class ClassifierMixin:
    pass
