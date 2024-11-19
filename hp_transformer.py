import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import norm
from scipy.optimize import minimize

class HPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.params = None

    def fit(self, X, y=None):
        # Center the data at the median
        self.median = np.median(X)
        X_centered = X - self.median

        # Initial parameter estimation
        beta_plus, lambda_plus = self._initial_params(X_centered[X_centered >= 0])
        beta_minus, lambda_minus = self._initial_params(-X_centered[X_centered < 0])
        alpha = self._estimate_alpha(X_centered, beta_minus, beta_plus, lambda_minus, lambda_plus)
        
        # Maximum likelihood estimation
        initial_params = [alpha, beta_minus, beta_plus, lambda_minus, lambda_plus]
        result = minimize(lambda params: -self._log_likelihood(X_centered, params),
                          initial_params, method='Nelder-Mead')
        self.params = result.x
        return self

    def transform(self, X):
        X_centered = X - self.median
        return self._hp_transform(X_centered, self.params)
    
    def _initial_params(self, X):
        # Sort the data
        X_sorted = np.sort(X)
        n = len(X_sorted)
    
        # Find xp and xq such that xq ≈ 2*xp
        for i in range(n//6, 2*(n//3)):
            xp = X_sorted[i]
            j = np.searchsorted(X_sorted, 2*xp)
            if j < n:
                xq = X_sorted[j]
                if np.isclose(xq, 2*xp, rtol=0.0025):  # Allow 2.5% tolerance
                    break
        else:
            raise ValueError("Could not find suitable xp and xq")
    
        # Calculate percentiles and z-scores
        p = i / n
        q = j / n
        zp = norm.ppf((p+1)/2)
        zq = norm.ppf((q+1)/2)
        
        # Determine initial beta and lambda
        if zq < 2*zp:
            beta = np.arccosh(zp / (zq - zp)) / abs(xp)
            lambda_ = 1.0
        else:
            beta = np.arccosh(zq / (2*zp)) / abs(xp)
            lambda_ = 0.0
    
        # Fine-tune lambda using additional quantiles
        xs, xt = np.percentile(X, [7, 93])
        zs, zt = norm.ppf([0.535, 0.965])
        lambda_ = (np.log(zs/zt) + np.log(np.sinh(beta*xt)) - np.log(np.sinh(beta*xs))) / \
                  (np.log(1/np.cosh(beta*xs)) - np.log(1/np.cosh(beta*xt)))
    
        return beta, min(lambda_, 1.0)

    def _estimate_alpha(self, X, beta_minus, beta_plus, lambda_minus, lambda_plus):
        xr_minus, xr_plus = np.percentile(X, [49, 51])
        zr_minus, zr_plus = norm.ppf([0.49, 0.51])
        
        numerator = beta_plus * zr_plus - beta_minus * zr_minus
        denominator = (np.sinh(beta_plus * xr_plus) / np.cosh(beta_plus * xr_plus)**lambda_plus) - \
                      (np.sinh(beta_minus * xr_minus) / np.cosh(beta_minus * xr_minus)**lambda_minus)
        
        return numerator / denominator

    def _hp_transform(self, X, params):
        alpha, beta_minus, beta_plus, lambda_minus, lambda_plus = params
        
        def transform_side(x, beta, lambda_):
            return alpha * np.sinh(beta * x) / (np.cosh(beta * x)**lambda_) / beta
    
        result = np.zeros_like(X)
        mask_pos = X >= 0
        mask_neg = X < 0
        
        result[mask_pos] = transform_side(X[mask_pos], beta_plus, lambda_plus)
        result[mask_neg] = transform_side(X[mask_neg], beta_minus, lambda_minus)
        
        return result

    def _log_likelihood(self, X, params):
        alpha, beta_minus, beta_plus, lambda_minus, lambda_plus = params
        y = self._hp_transform(X, params)
        
        log_likelihood = -0.5 * np.sum(y**2)
        log_likelihood += len(X) * np.log(alpha)
        
        X_neg = X[X < 0]
        X_pos = X[X >= 0]
        
        # Negative side
        log_likelihood += np.sum(np.log(1 - lambda_minus * np.tanh(beta_minus * X_neg)**2))
        log_likelihood += (lambda_minus - 1) * np.sum(np.log(1 / np.cosh(beta_minus * X_neg)))
        
        # Positive side
        log_likelihood += np.sum(np.log(1 - lambda_plus * np.tanh(beta_plus * X_pos)**2))
        log_likelihood += (lambda_plus - 1) * np.sum(np.log(1 / np.cosh(beta_plus * X_pos)))
        
        return log_likelihood
