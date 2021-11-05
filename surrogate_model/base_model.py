import numpy as np
from sklearn.impute import SimpleImputer


class BaseModel(object):
    """
    A surrogate model is used to evaluate the potential loss distribution, it should be able to predict the
    mean and variance values on the given position.
    """
    def __init__(self):
        self.is_trained = False
        self.model = None
        self.imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=np.finfo(float).max)

    def train(self, X: np.ndarray, y: np.ndarray):
        if len(X.shape) == 1:
            X = X[np.newaxis, :]
        X = self._impute_nan(X)
        self.model.fit(X, np.squeeze(y))
        self.is_trained = True

    def predict(self, X: np.ndarray, **kwargs):
        if len(X.shape) == 1:
           X = X[np.newaxis, :]
        if not self.is_trained:
            raise ValueError("model needs to be trained first!")
        X = self._impute_nan(X)
        return self._predict(X, **kwargs)

    def _predict(self, X: np.ndarray, **kwargs):
        raise NotImplementedError

    def _impute_nan(self, input):
        return self.imp.fit_transform(input)

