# This file will contain the core NumPy logic for the hyperbolic transformer.
import numpy as np
from scipy.stats import norm


def hp_transform_np(X, alpha, beta_plus, lambda_plus, beta_minus, lambda_minus, median):
    """Apply the hyperbolic transformation to the input data.

    This function transforms data `X` using a set of hyperbolic parameters.
    The transformation is applied differently to data points above or below
    the median.

    The core transformation for one side (e.g., positive, x_centered > 0) is:
    y = alpha * sinh(beta * x_centered) / (cosh(beta * x_centered)^lambda * beta)

    Parameters
    ----------
    X : numpy.ndarray
        Input data array. Can be 1D or 2D.
    alpha : float
        The scaling parameter of the transformation.
    beta_plus : float
        The beta parameter for data points greater than or equal to the median.
        Controls the "spread" on the positive side.
    lambda_plus : float
        The lambda parameter for data points greater than or equal to the median.
        Controls the tail behavior on the positive side.
    beta_minus : float
        The beta parameter for data points less than the median.
        Controls the "spread" on the negative side.
    lambda_minus : float
        The lambda parameter for data points less than the median.
        Controls the tail behavior on the negative side.
    median : float
        The median of the original dataset, used to center the data `X`.

    Returns
    -------
    numpy.ndarray
        The transformed data array, with the same shape as `X`.

    Notes
    -----
    If beta_plus or beta_minus is very close to zero (<= 1e-9), the transformation
    approximates to `alpha * (X - median)` for that respective side to maintain
    numerical stability.
    """
    X_centered = X - median
    result = np.zeros_like(X_centered, dtype=float)  # Ensure float type for results

    mask_pos = X_centered >= 0
    mask_neg = X_centered < 0

    # Helper to avoid division by zero if beta is zero, though beta should be positive
    # In practice, optimization should prevent beta from being exactly zero.
    # However, adding a small epsilon or check might be robust in a general utility.

    # Positive side
    cosh_beta_x_pos = np.cosh(beta_plus * X_centered[mask_pos])
    # Avoid issues if cosh_beta_x_pos is zero or very small, or if beta_plus is zero.
    # Assuming beta_plus > 0 as per constraints often found in such transformations.
    if beta_plus > 1e-9: # Check to prevent division by zero if beta_plus is too small
        result[mask_pos] = alpha * np.sinh(beta_plus * X_centered[mask_pos]) / \
                           (cosh_beta_x_pos**lambda_plus) / beta_plus
    else: # Handle edge case of beta_plus being zero or near zero
        result[mask_pos] = alpha * X_centered[mask_pos] # Approximation for small beta*x (sinh(ax)/a -> x)


    # Negative side
    cosh_beta_x_neg = np.cosh(beta_minus * X_centered[mask_neg])
    # Similar check for beta_minus
    if beta_minus > 1e-9: # Check to prevent division by zero if beta_minus is too small
        result[mask_neg] = alpha * np.sinh(beta_minus * X_centered[mask_neg]) / \
                           (cosh_beta_x_neg**lambda_minus) / beta_minus
    else: # Handle edge case of beta_minus being zero or near zero
        result[mask_neg] = alpha * X_centered[mask_neg] # Approximation

    return result


def initial_params_np(X_one_side):
    """
    Estimates initial beta and lambda parameters for one side of a distribution.

    This function takes one side of a distribution (e.g., all positive centered values,
    or all negative centered values made positive) and estimates initial `beta`
    and `lambda` parameters. These are used as starting points for more complex
    optimization in the `HPTransformer`. The method involves finding specific
    quantiles (`xp`, `xq`) and corresponding standard normal z-scores (`zp`, `zq`)
    to derive the parameters, with fallbacks for small or difficult datasets.

    Parameters
    ----------
    X_one_side : numpy.ndarray
        A 1D array of data points from one side of the distribution, assumed to be positive.
        For example, if `X_centered` is the original data minus its median, then
        `X_one_side` could be `X_centered[X_centered > 0]` or
        `-X_centered[X_centered < 0]`. It should not be empty.

    Returns
    -------
    tuple of (float, float)
        A tuple containing the estimated initial (`beta`, `lambda`) parameters.
        `beta` is clipped to `[1e-4, 100.0]`. `lambda` is clipped to `[0.0, 1.0]`.

    Notes
    -----
    The function includes several fallbacks and numerical stability adjustments:
    - For `n < 3` data points (after filtering), it uses a robust percentile-based
      estimation with default lambda.
    - If the primary `xp`/`xq` search fails, it falls back to 25th/75th percentiles.
    - Beta values are clipped to prevent them from being too small or excessively large.
    - Lambda fine-tuning is only attempted for `n >= 10` and includes error handling
      for numerical issues.
    """
    if len(X_one_side) == 0:
        # Default parameters if no data on this side (should be handled by HPTransformer.fit)
        return 1.0, 1.0

    # Ensure X_one_side contains only positive values for consistency with original logic
    # The original logic operates on X_centered[X_centered >= 0] or -X_centered[X_centered < 0]
    # So, values passed here should be positive.
    X_processed = X_one_side[X_one_side > 0]
    if len(X_processed) == 0: # If all values were zero or negative after filtering
        return 1.0, 1.0


    X_sorted = np.sort(X_processed)
    n = len(X_sorted)

    if n < 3: # Fallback for very small arrays (n=0, 1, 2 after filtering)
        if n == 0: return 1.0, 1.0 # No data points
        # For n=1 or n=2, percentile-based estimation is unstable / ill-defined for xp, xq logic.
        # Return robust default parameters.
        # Example: use the single point (if n=1) or midpoint (if n=2) as a crude xq1.
        # This path should ideally lead to simple, stable parameters.
        # Let's use a simple default: beta=1, lambda=1 or 0.5.
        # The original fallback used quantiles, let's try to make that more robust.
        xq1_fallback = np.percentile(X_sorted, 25) if n > 0 else 0.01 # Avoid 0 if possible
        xq2_fallback = np.percentile(X_sorted, 75) if n > 0 else 0.02
        
        # Corrected norm.ppf usage for one-sided data mapped to upper half of normal
        zp1_fallback = norm.ppf(0.5 + 0.25 / 2) # norm.ppf(0.625)
        zp2_fallback = norm.ppf(0.5 + 0.75 / 2) # norm.ppf(0.875)

        if xq1_fallback < 1e-9: xq1_fallback = 1e-9 # Avoid zero or negative
        if xq2_fallback < xq1_fallback + 1e-9: xq2_fallback = xq1_fallback + 1e-9 # Ensure xq2 > xq1

        if np.isclose(xq1_fallback, xq2_fallback):
            beta = 1.0 # Default if quantiles are too close
        else:
            beta = np.abs((zp1_fallback - zp2_fallback) / (xq1_fallback - xq2_fallback))
        
        beta = np.clip(beta, 1e-4, 100.0) # Clip beta to a reasonable range
        return beta, 0.5 # Default lambda for very small N, 0.5 is neutral.


    xp, xq = -1, -1 # Initialize xp, xq
    # Find xp and xq such that xq ≈ 2*xp
    # Original iteration range: range(n//6, 2*(n//3))
    # Ensure indices are within bounds
    start_idx = n // 6
    end_idx = 2 * (n // 3)
    if start_idx >= end_idx and n > 0 : # if n is small, e.g. n=1,2,3,4,5
        start_idx = 0
        end_idx = n -1


    found_xp_xq = False
    for i in range(start_idx, end_idx):
        if i >= n: continue # boundary check
        xp_candidate = X_sorted[i]
        if xp_candidate <= 0: continue # Ensure xp is positive, original data is X_centered >=0 or -X_centered[X < 0]

        # Search for 2*xp_candidate in the remainder of the sorted array
        # np.searchsorted finds insertion point, check element at j-1 and j
        j_candidate = np.searchsorted(X_sorted, 2 * xp_candidate, side='left')

        # Check values around j_candidate to find closest to 2*xp_candidate
        # within the 0.25% rtol specified in original code.
        # (Original: np.isclose(xq, 2*xp, rtol=0.0025))
        # Let's check X_sorted[j_candidate] and X_sorted[j_candidate-1]
        
        potential_j = []
        if j_candidate < n: potential_j.append(j_candidate)
        if j_candidate > 0 and (j_candidate-1) > i : potential_j.append(j_candidate-1) # j-1 must be after i

        for j_idx in potential_j:
            xq_candidate = X_sorted[j_idx]
            if np.isclose(xq_candidate, 2 * xp_candidate, rtol=0.025): # Increased rtol from 0.0025
                xp = xp_candidate
                xq = xq_candidate
                p = (i + 1) / n # Use (i+1)/n for percentile p, as i is 0-indexed
                q = (j_idx + 1) / n # Use (j_idx+1)/n for percentile q
                found_xp_xq = True
                break
        if found_xp_xq:
            break
    
    if not found_xp_xq:
        # Fallback if suitable xp, xq not found; use quantiles as in the previous version of initial_params_np
        # This indicates the data might not fit the assumptions well.
        xp = np.percentile(X_sorted, 25) # q1_val * 100
        xq = np.percentile(X_sorted, 75) # q2_val * 100
        if xp <= 1e-9 and xq <= 1e-9: return 1.0, 0.5 # all data effectively zero, return defaults
        if xp <= 1e-9: xp = 1e-4 # avoid zero for xp
        if xq <= xp: xq = xp + 1e-4 # ensure xq > xp, avoid division by zero later
        # if xp == xq and xp != 0 : xq = 2*xp # ensure xq > xp # Old logic
        # if xp > xq : xp, xq = xq, xp # ensure xp < xq # Old logic


        p = 0.25
        q = 0.75
        # raise ValueError("Could not find suitable xp and xq in initial_params_np. Data might be sparse or not fit assumptions.")

    # Calculate z-scores using ppf of (1+p)/2 not p - this is for one-sided from median
    # Original: zp = norm.ppf((p+1)/2), zq = norm.ppf((q+1)/2)
    # If p,q are from i/n, j/n, they are already "one-sided" percentiles from 0.
    # The (p+1)/2 was likely to map a two-sided percentile to a one-sided for norm.ppf.
    # If X_one_side is already e.g. X[X>0], then p=0.25 means the 25th percentile of *that positive data*.
    # So norm.ppf(p) or norm.ppf(q) might be more direct if p,q are 0 to 1.
    # Let's test norm.ppf(p_val) where p_val is like 0.25, 0.75 for the sub-distribution.
    # The original paper implies these are quantiles of the FOLDED normal if starting from two-sided.
    # Given X_one_side is already one side, p and q are direct quantiles for that side.
    # So norm.ppf(0.5 + p/2) if p was a two-sided percentile.
    # If p = i/n for X_sorted (which is one side), then this p is effectively like (p_two_sided - 0.5)*2
    # The original HPTransformer passed `X_centered[X_centered >= 0]` which means p=i/n is a direct percentile of the positive side.
    # So, norm.ppf(p) would map to Z if the positive side itself were normal.
    # But we want it to map to Z of a std normal that was folded.
    # The original `zp = norm.ppf((p+1)/2)` means if p was 0.5 (median of one side), zp = norm.ppf(0.75)
    # This implies mapping the percentiles of the one-sided data (0 to 1) to the upper half of the normal distribution (0.5 to 1).
    zp = norm.ppf(0.5 + p / 2)
    zq = norm.ppf(0.5 + q / 2)


    # Determine initial beta and lambda
    if xp <= 1e-9: # Should have been caught by fallback's xp handling, but as safeguard
        beta = 1.0 
        lambda_ = 0.5 # Default lambda
    elif np.isclose(xp, xq) or np.isclose(zp, zq) or zp == 0: # Avoid division by zero or unstable conditions
        beta = 1.0 # Default beta
        lambda_ = 0.5 # Default lambda
    else:
        # Original conditions: if zq < 2*zp ...
        ratio_zq_zp = zq / zp
        if ratio_zq_zp < 2.0 : # Corresponds to zq < 2*zp
            val_for_arccosh = zp / (zq - zp) # This can be zq-zp == 0 or zp/(negative)
            if (zq - zp) <= 1e-9: # If zq is very close or less than zp
                 beta = 1.0 # Default beta
            else:
                val_for_arccosh = max(1.0, val_for_arccosh) # Ensure arccosh input >= 1
                beta = np.arccosh(val_for_arccosh) / xp
            lambda_ = 1.0
        else: # Corresponds to zq >= 2*zp
            val_for_arccosh = zq / (2 * zp) # zp cannot be zero here due to earlier check
            val_for_arccosh = max(1.0, val_for_arccosh) # Ensure arccosh input >= 1
            beta = np.arccosh(val_for_arccosh) / xp
            lambda_ = 0.0
    
    beta = np.clip(beta, 1e-4, 100.0) # Ensure beta is within a reasonable range, avoid extreme values

    # Fine-tune lambda using additional quantiles
    if n >= 10: # Only fine-tune if enough data points for stable 7th/93rd percentiles
        xs = np.percentile(X_sorted, 7)
        xt = np.percentile(X_sorted, 93)
        
        zs = norm.ppf(0.5 + 0.07 / 2) 
        zt = norm.ppf(0.5 + 0.93 / 2) 

        if not (np.isclose(xs, xt) or beta <= 1e-4 or xs <= 1e-9 or xt <= 1e-9 or zt == 0 or zs ==0):
            # Use np.errstate to handle potential overflows/invalid logs temporarily for fine-tuning
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                log_sinh_xt_val = np.log(np.sinh(beta * xt))
                log_sinh_xs_val = np.log(np.sinh(beta * xs))
                log_cosh_beta_xs = np.log(np.cosh(beta * xs))
                log_cosh_beta_xt = np.log(np.cosh(beta * xt))

                # Check for inf/-inf from logs before proceeding
                if not np.any(np.isinf([log_sinh_xt_val, log_sinh_xs_val, log_cosh_beta_xs, log_cosh_beta_xt])):
                    numerator = np.log(zs / zt) + log_sinh_xt_val - log_sinh_xs_val
                    denominator = -log_cosh_beta_xs - (-log_cosh_beta_xt) # log(1/cosh(xs)) - log(1/cosh(xt))
                    
                    if not np.isclose(denominator, 0):
                        lambda_tuned = numerator / denominator
                        # Only update lambda if tuned value is plausible and different
                        if not np.isnan(lambda_tuned) and not np.isinf(lambda_tuned):
                             lambda_ = np.clip(lambda_tuned, 0.0, 1.0)

    return beta, lambda_


def estimate_alpha_np(X_centered, beta_minus, beta_plus, lambda_minus, lambda_plus):
    """
    Estimates the alpha scaling parameter for the hyperbolic transformation.

    The alpha parameter is estimated such that the variance of the data,
    after being transformed by the hyperbolic shape function (without alpha),
    would be approximately 1. This function calculates alpha based on the
    sum of squares of these "pre-transformed" values.

    The formula used is `alpha = sqrt(N / sum(Y_i^2))`, where `Y_i` is the
    data transformed by the hyperbolic shape function (with alpha conceptually
    set to 1 for this estimation step) and `N` is the number of data points.

    Parameters
    ----------
    X_centered : numpy.ndarray
        The input data, already centered by subtracting the median.
    beta_minus : float
        The beta parameter for the negative side (X_centered < 0).
    beta_plus : float
        The beta parameter for the positive side (X_centered >= 0).
    lambda_minus : float
        The lambda parameter for the negative side.
    lambda_plus : float
        The lambda parameter for the positive side.

    Returns
    -------
    float
        The estimated alpha parameter, constrained to be at least 1e-4.
        Returns 1.0 if `X_centered` is empty or if the sum of squares of
        transformed values is too small (close to zero).
    """
    transformed_X = np.zeros_like(X_centered, dtype=float)

    mask_pos = X_centered >= 0
    mask_neg = X_centered < 0

    # Positive side transformation (without alpha, as we are estimating it)
    if beta_plus > 1e-9:
        transformed_X[mask_pos] = np.sinh(beta_plus * X_centered[mask_pos]) / \
                                  (np.cosh(beta_plus * X_centered[mask_pos])**lambda_plus) / beta_plus
    else:
        transformed_X[mask_pos] = X_centered[mask_pos]


    # Negative side transformation (without alpha)
    if beta_minus > 1e-9:
        transformed_X[mask_neg] = np.sinh(beta_minus * X_centered[mask_neg]) / \
                                  (np.cosh(beta_minus * X_centered[mask_neg])**lambda_minus) / beta_minus
    else:
        transformed_X[mask_neg] = X_centered[mask_neg]

    # Estimate alpha by setting variance of transformed_X to 1
    # alpha = 1 / np.std(transformed_X) if np.std(transformed_X) > 0 else 1.0
    # The original code uses a sum of squares approach, which relates to variance.
    # alpha^2 * sum(Y_i^2 / N) = 1 (if Y is Y/alpha from the formula)
    # So alpha = sqrt(N / sum(Y_i^2))
    # where Y_i is the transformed value without alpha.
    N = len(X_centered)
    if N == 0:
        return 1.0 # Default alpha if no data

    sum_sq_transformed = np.sum(transformed_X**2)

    if sum_sq_transformed < 1e-9: # Avoid division by zero or extremely small values
        alpha_est = 1.0 # Default or indicates issues with transformation/data
    else:
        alpha_est = np.sqrt(N / sum_sq_transformed)
    
    if alpha_est < 1e-4: alpha_est = 1e-4 # Constraint from original code

    return alpha_est


def log_likelihood_np(X_centered, alpha, beta_plus, lambda_plus, beta_minus, lambda_minus):
    """
    Calculates the log-likelihood of the data given hyperbolic transformation parameters.

    The log-likelihood is computed assuming the transformed data follows a
    standard normal distribution. The formula involves the sum of log-PDF values
    of the transformed data and the sum of log-absolute-Jacobian-determinants
    of the transformation.

    LL = N * (-0.5 * log(2*pi)) - 0.5 * sum(y_i^2) + sum(log|dy_i/dx_i|)

    where `y_i` is the transformed value of `x_i` (an element of `X_centered`),
    and `dy_i/dx_i` is the derivative of the transformation function. Note that
    the derivative formula used here is slightly different for the positive and
    negative sides, matching the historical implementation.

    Parameters
    ----------
    X_centered : numpy.ndarray
        The input data, already centered by subtracting the median.
    alpha : float
        The scaling parameter of the transformation.
    beta_plus : float
        The beta parameter for the positive side (X_centered >= 0).
    lambda_plus : float
        The lambda parameter for the positive side.
    beta_minus : float
        The beta parameter for the negative side (X_centered < 0).
    lambda_minus : float
        The lambda parameter for the negative side.

    Returns
    -------
    float
        The calculated log-likelihood value. Returns -numpy.inf if `X_centered`
        is empty or if numerical issues result in NaN/Inf likelihood.

    Notes
    -----
    The function uses different Jacobian determinant formulas for positive and negative sides:
    - Positive side: `log(alpha) - lambda_p * log(cosh(beta_p*x)) + log(abs(1 - tanh(beta_p*x)^2 * lambda_p))`
    - Negative side: `log(alpha) + log(abs(1 - lambda_m * tanh(beta_m*x)^2)) - (lambda_m - 1) * log(cosh(beta_m*x))`
    This matches the structure used in the `HPTransformer` class.
    Numerical issues like `log(0)` are handled by replacing resulting NaNs/Infs in
    `log_abs_deriv` with -numpy.inf.
    """
    N = len(X_centered)
    if N == 0:
        return -np.inf # Or some other indicator of invalid input

    # Transform the data using the current parameters
    # Note: hp_transform_np requires 'median' but it's already centered.
    # We can call a simplified version or pass 0 as median if X_centered is truly X-median.
    # For directness, let's re-implement the transformation part specific to log-likelihood's needs.

    y = np.zeros_like(X_centered, dtype=float)
    log_abs_deriv = np.zeros_like(X_centered, dtype=float)

    mask_pos = X_centered >= 0
    mask_neg = X_centered < 0

    # Positive side
    if np.any(mask_pos):
        X_pos = X_centered[mask_pos]
        cosh_beta_x_pos = np.cosh(beta_plus * X_pos)
        # Transformation
        if beta_plus > 1e-9:
            y[mask_pos] = alpha * np.sinh(beta_plus * X_pos) / (cosh_beta_x_pos**lambda_plus) / beta_plus
        else:
            y[mask_pos] = alpha * X_pos # Approx for small beta
        # Log of absolute derivative
        log_abs_deriv[mask_pos] = np.log(alpha) - lambda_plus * np.log(cosh_beta_x_pos) + \
                                  np.log(np.abs(1 - np.tanh(beta_plus * X_pos)**2 * lambda_plus)) \
                                  if lambda_plus > 0 else np.log(alpha) - lambda_plus * np.log(cosh_beta_x_pos)


    # Negative side
    if np.any(mask_neg):
        X_neg = X_centered[mask_neg]
        cosh_beta_x_neg = np.cosh(beta_minus * X_neg)
        # Transformation
        if beta_minus > 1e-9:
            y[mask_neg] = alpha * np.sinh(beta_minus * X_neg) / (cosh_beta_x_neg**lambda_minus) / beta_minus
        else:
            y[mask_neg] = alpha * X_neg # Approx for small beta

        # Log of absolute derivative
        # Note: derivative of sinh(b*x) / cosh(b*x)^l / b is complex.
        # The original code's derivative: alpha * (1 - lambda_ * np.tanh(beta * x)**2) / (np.cosh(beta * x)**(lambda_ -1))
        # The log of this is: log(alpha) + log(abs(1 - lambda_ * tanh(beta*x)**2)) - (lambda_ -1) * log(cosh(beta*x))
        # This seems different from the paper's formula. Let's re-check the paper or use a known formula for Jacobian.

        # The log-likelihood for a transformation y = g(x) is:
        # sum(log(f_Z(y_i))) + sum(log|g'(x_i)|)
        # where f_Z is the PDF of the target distribution (std normal here).
        # log(f_Z(y_i)) = -0.5 * log(2*pi) - 0.5 * y_i^2
        # So, LL = N * (-0.5 * np.log(2 * np.pi)) - 0.5 * np.sum(y**2) + np.sum(log_abs_deriv)

        # The derivative part from the original paper (and common transformations) is:
        # log |dy/dx| = log(alpha) - lambda * log(cosh(beta * x)) for the core part of the transformation.
        # The (1 - lambda * tanh^2(beta*x)) term arises from d/dx (tanh(beta*x) / cosh(beta*x)^(lambda-1))
        # Let's use the derivative form from the paper's log-likelihood which is simpler:
        # log(alpha) - lambda_ * np.log(np.cosh(beta_ * x))
        # This is log | dy/dx | when y = alpha * integral(1/cosh(beta*x)^lambda dx)
        # The HP transformation is y = alpha * sinh(beta*x) / (beta * cosh(beta*x)^lambda)
        # Its derivative dy/dx = alpha * (cosh(beta*x)^2 - lambda * sinh(beta*x)^2) / cosh(beta*x)^(lambda+1)
        # dy/dx = alpha * (1 - lambda * tanh(beta*x)^2) / cosh(beta*x)^(lambda-1) -- This is what was in the class.

        log_abs_deriv[mask_neg] = np.log(alpha) + np.log(np.abs(1 - lambda_minus * np.tanh(beta_minus * X_neg)**2)) - \
                                 (lambda_minus -1) * np.log(cosh_beta_x_neg) # lambda_minus-1 because cosh is in denominator of derivative formula

    # Replace NaN or inf in log_abs_deriv (e.g., if cosh is huge or (1-tanh^2*lambda) is zero)
    log_abs_deriv = np.nan_to_num(log_abs_deriv, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)


    log_likelihood_val = N * (-0.5 * np.log(2 * np.pi)) - 0.5 * np.sum(y**2) + np.sum(log_abs_deriv)
    
    # Handle cases where log_likelihood is NaN or +/- infinity
    if np.isnan(log_likelihood_val) or np.isinf(log_likelihood_val):
        return -np.inf # Return a very small number if calculation fails

    return log_likelihood_val
