import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
# Import the new NumPy-based functions
from hyperbolic_np import (
    hp_transform_np,
    initial_params_np,
    estimate_alpha_np,
    log_likelihood_np
)

class HPTransformer(BaseEstimator, TransformerMixin):
    """Apply a Hyperbolic arcsinh-like transformation to normalize data.

    This transformer estimates parameters for a two-sided hyperbolic
    transformation that aims to make the input data more closely resemble a
    standard normal distribution. The transformation is fitted using maximum
    likelihood estimation.

    The transformation is defined separately for values above and below the
    median of the input data, allowing for asymmetry.

    Attributes
    ----------
    params_ : numpy.ndarray
        The estimated parameters for the transformation, stored as a NumPy array
        in the order: [alpha, beta_minus, beta_plus, lambda_minus, lambda_plus].
        - alpha: Scaling parameter.
        - beta_minus: Beta parameter for the negative side (X < median).
        - beta_plus: Beta parameter for the positive side (X >= median).
        - lambda_minus: Lambda parameter for the negative side.
        - lambda_plus: Lambda parameter for the positive side.
        These are available after `fit` has been called.
    median_ : float
        The median of the training data, used for centering.
        Available after `fit` has been called.

    Methods
    -------
    fit(X, y=None)
        Estimate transformation parameters from the input data X.
    transform(X)
        Apply the learned hyperbolic transformation to X.
    fit_transform(X, y=None)
        Fit to data, then transform it.

    Examples
    --------
    >>> import numpy as np
    >>> from hp_transformer import HPTransformer
    >>> # Generate some skewed sample data
    >>> np.random.seed(42)
    >>> X_skewed = np.random.lognormal(mean=1, sigma=0.8, size=200).reshape(-1, 1)
    >>> # Instantiate and fit the transformer
    >>> transformer = HPTransformer()
    >>> transformer.fit(X_skewed)
    >>> # Transform the data
    >>> X_transformed = transformer.transform(X_skewed)
    >>> print("Fitted Parameters (alpha, beta_m, beta_p, lambda_m, lambda_p):")
    >>> print(np.round(transformer.params_, 3))
    >>> print("\\nFitted Median:")
    >>> print(f"{transformer.median_:.3f}")
    >>> print("\\nOriginal data (first 5 samples):")
    >>> print(np.round(X_skewed[:5], 3))
    >>> print("\\nTransformed data (first 5 samples):")
    >>> print(np.round(X_transformed[:5], 3))
    >>> # Check skewness (transformed data should ideally be less skewed)
    >>> from scipy.stats import skew
    >>> print(f"\\nOriginal skewness: {skew(X_skewed.ravel()):.3f}")
    >>> print(f"Transformed skewness: {skew(X_transformed.ravel()):.3f}")
    """
    def __init__(self):
        # No hyperparameters to set in __init__
        # Fitted attributes will be params_ and median_
        self.params_ = None
        self.median_ = None

    def fit(self, X, y=None):
        """Estimate transformation parameters from the input data X.

        The method involves:
        1. Centering the data by subtracting its median.
        2. Estimating initial parameters (alpha, betas, lambdas) using
           quantile-based heuristics (`initial_params_np`, `estimate_alpha_np`).
        3. Refining these parameters using maximum likelihood estimation by
           minimizing the negative log-likelihood (`log_likelihood_np`)
           via L-BFGS-B.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            The input data to estimate the transformation parameters from.
            Multi-feature input is not supported.
        y : None
            Ignored. This parameter is included for sklearn compatibility.

        Returns
        -------
        self : HPTransformer
            The fitted transformer instance.
        """
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # For now, assuming X is 1D as per original design. If multiple columns,
        # transformation should be column-wise or adapted.
        # This implementation processes X as a single feature.
        if X_array.shape[1] > 1:
            # This is a simplification. For multi-feature, would need to loop or vectorize params
            raise ValueError("HPTransformer currently supports single-feature input. For multi-feature, apply column by column.")
        
        X_flat = X_array.ravel()


        # Center the data at the median
        self.median_ = np.median(X_flat)
        X_centered = X_flat - self.median_

        # Filter out zero values for initial parameter estimation on each side
        X_pos_initial = X_centered[X_centered > 0.00001] # Added small epsilon to avoid issues with exact zero
        X_neg_initial = -X_centered[X_centered < 0] # Pass positive values for the negative side

        # Initial parameter estimation using NumPy functions
        # Handle cases with no data on one side
        if len(X_pos_initial) > 0:
            beta_plus, lambda_plus = initial_params_np(X_pos_initial)
        else: # Default if no positive data
            beta_plus, lambda_plus = 1.0, 1.0 
        
        if len(X_neg_initial) > 0:
            beta_minus, lambda_minus = initial_params_np(X_neg_initial)
        else: # Default if no negative data
            beta_minus, lambda_minus = 1.0, 1.0

        # Estimate alpha using NumPy function
        # estimate_alpha_np expects (X_centered, beta_minus, beta_plus, lambda_minus, lambda_plus)
        alpha = estimate_alpha_np(X_centered, beta_minus, beta_plus, lambda_minus, lambda_plus)
        
        # Maximum likelihood estimation
        # Parameter order for storage and optimization: [alpha, beta_minus, beta_plus, lambda_minus, lambda_plus]
        initial_params = [alpha, beta_minus, beta_plus, lambda_minus, lambda_plus]
        
        # Define the objective function for minimization
        # log_likelihood_np expects (X_centered, alpha, beta_plus, lambda_plus, beta_minus, lambda_minus)
        # params from minimize: [alpha, beta_m, beta_p, lambda_m, lambda_p]
        # So, params[0]=alpha, params[1]=beta_m, params[2]=beta_p, params[3]=lambda_m, params[4]=lambda_p
        # Mapping to log_likelihood_np:
        #   alpha_arg = params[0]
        #   beta_plus_arg = params[2]
        #   lambda_plus_arg = params[4]
        #   beta_minus_arg = params[1]
        #   lambda_minus_arg = params[3]
        objective_function = lambda p: -log_likelihood_np(
            X_centered, p[0], p[2], p[4], p[1], p[3]
        )
        
        bounds = [
            (1e-7, None),  # alpha
            (1e-6, None),  # beta_minus
            (1e-6, None),  # beta_plus
            (0, 1.0),      # lambda_minus
            (0, 1.0)       # lambda_plus
        ]
        
        result = minimize(objective_function,
                          initial_params, 
                          method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-7, 'gtol': 1e-5}) # Standard options
                          
        self.params_ = result.x
        
        # Safeguard parameters after optimization, ensuring they are within valid ranges.
        # alpha (params_[0]) is already lower-bounded by 1e-7.
        self.params_[1] = max(1e-7, self.params_[1])  # beta_minus
        self.params_[2] = max(1e-7, self.params_[2])  # beta_plus
        self.params_[3] = np.clip(self.params_[3], 0.0, 1.0)  # lambda_minus
        self.params_[4] = np.clip(self.params_[4], 0.0, 1.0)  # lambda_plus

        return self

    def transform(self, X):
        """Apply the learned hyperbolic transformation to X.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            The input data to transform. It should have a single feature.

        Returns
        -------
        numpy.ndarray
            The transformed data, with shape (n_samples, 1).

        Raises
        ------
        RuntimeError
            If the transformer has not been fitted yet (i.e., `fit` has not been called).
        ValueError
            If the input `X` has more than one feature.
        """
        if self.params_ is None or self.median_ is None:
            raise RuntimeError("Transformer has not been fitted yet.")
        
        X_array = np.asarray(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1,1) # Ensure 2D for consistency, even if 1 feature
        
        if X_array.shape[1] > 1:
             raise ValueError("HPTransformer currently supports single-feature input. For multi-feature, apply column by column.")
        
        X_flat = X_array.ravel()

        # hp_transform_np expects (X_input (original), alpha, beta_plus, lambda_plus, beta_minus, lambda_minus, median)
        # self.params_ order: [alpha, beta_minus, beta_plus, lambda_minus, lambda_plus]
        transformed_X_flat = hp_transform_np(
            X_flat, 
            alpha=self.params_[0],
            beta_plus=self.params_[2],
            lambda_plus=self.params_[4],
            beta_minus=self.params_[1],
            lambda_minus=self.params_[3],
            median=self.median_
        )
        return transformed_X_flat.reshape(X_array.shape)


if __name__ == '__main__':
    import numpy as np
    from scipy.stats import skew # For example

    # Generate some skewed sample data
    np.random.seed(42)
    # X_skewed_example = np.random.lognormal(mean=1, sigma=0.8, size=200) 
    # Using a slightly different distribution for variety in example if preferred
    X_skewed_example = np.random.gamma(shape=2, scale=2, size=200) + 1.0
    # HPTransformer expects 2D array for fit if from sklearn pipeline, but internally ravels.
    # For direct use, 1D is fine, but let's show with reshape for sklearn typical usage.
    if X_skewed_example.ndim == 1:
        X_skewed_example = X_skewed_example.reshape(-1, 1)

    print("--- HPTransformer Example ---")
    print(f"Original Data (first 5 samples):\\n{np.round(X_skewed_example[:5], 3)}")
    print(f"Original Skewness: {skew(X_skewed_example.ravel()):.3f}\\n")

    # Instantiate and fit the transformer
    transformer = HPTransformer()
    transformer.fit(X_skewed_example)

    # Display fitted parameters
    print("Fitted Parameters (alpha, beta_m, beta_p, lambda_m, lambda_p):")
    print(np.round(transformer.params_, 3))
    print(f"Fitted Median: {transformer.median_:.3f}\\n")

    # Transform the data
    X_transformed_example = transformer.transform(X_skewed_example)

    print(f"Transformed Data (first 5 samples):\\n{np.round(X_transformed_example[:5], 3)}")
    print(f"Transformed Skewness: {skew(X_transformed_example.ravel()):.3f}")

    # Example of transforming new data
    X_new_data = np.array([[0.5], [2.0], [10.0], [25.0]])
    X_new_transformed = transformer.transform(X_new_data)
    print(f"\\nNew data (original):\\n{X_new_data}")
    print(f"New data (transformed):\\n{np.round(X_new_transformed,3)}")
    print("-" * 30)
