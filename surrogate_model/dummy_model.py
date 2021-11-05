import numpy as np
from sklearn.dummy import DummyRegressor

from .base_model import BaseModel

class DummyModel(BaseModel):
    """
    This is only an example showing you how to implement a surrogate model
    """
    def __init__(self, constant_value=0.5):
        super(DummyModel, self).__init__()
        self.model = DummyRegressor(strategy='constant', constant=constant_value)

    def _predict(self, X: np.ndarray, **kwargs):
        mu, std = self.model.predict(X, return_std=True)
        return mu, std ** 2
