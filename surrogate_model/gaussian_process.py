import typing
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, Kernel

from .base_model import BaseModel

class GaussianProcess(BaseModel):
    def __init__(self, kernel: typing.Optional[Kernel] = None):
        super(GaussianProcess, self).__init__()
        if kernel is None:
            constant_kernel = ConstantKernel(constant_value_bounds=(np.exp(-10), np.exp(2)))
            exp_kernel = Matern(length_scale_bounds=(np.exp(-6.754111155189306), np.exp(0.0858637988771976)), nu=2.5)
            noise_kernel = WhiteKernel(noise_level=1e-8, noise_level_bounds=(np.exp(-25), np.exp(2)))
            kernel = constant_kernel * exp_kernel + noise_kernel
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=10,
            alpha=0,
            )

    def _predict(self, X: np.ndarray, return_cov=False):
        if not return_cov:
            mu, std = self.model.predict(X, return_std=True)
            var = std ** 2
            var = np.clip(var, 1e-16, np.inf)
        else:
            mu, var = self.model.predict(X, return_cov=True)
        return mu, var