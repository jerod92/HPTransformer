import torch
import torch.nn as nn

class HyperbolicTransformTorch(nn.Module):
    """PyTorch module for applying a pre-trained Hyperbolic Transformation.

    This module applies a two-sided hyperbolic transformation to input tensors
    using a pre-defined set of parameters. These parameters are typically
    obtained by fitting the `HPTransformer` from `hp_transformer.py` on a dataset.

    The transformation parameters (`alpha`, `beta_plus`, `lambda_plus`,
    `beta_minus`, `lambda_minus`, `median`) are registered as buffers. This means
    they are part of the module's state, will be moved to the correct device
    (e.g., GPU) along with the module, but are not considered trainable
    parameters by PyTorch's optimization algorithms.

    The core transformation for one side (e.g., positive, x_centered > 0) is:
    y = alpha * sinh(beta * x_centered) / (cosh(beta * x_centered)^lambda * beta)

    Attributes
    ----------
    alpha_val : torch.Tensor
        The stored alpha scaling parameter.
    beta_plus_val : torch.Tensor
        The stored beta parameter for the positive side.
    lambda_plus_val : torch.Tensor
        The stored lambda parameter for the positive side.
    beta_minus_val : torch.Tensor
        The stored beta parameter for the negative side.
    lambda_minus_val : torch.Tensor
        The stored lambda parameter for the negative side.
    median_val : torch.Tensor
        The stored median value for centering data.
    """
    def __init__(self, alpha: float, beta_plus: float, lambda_plus: float, 
                 beta_minus: float, lambda_minus: float, median: float):
        """Initializes the HyperbolicTransformTorch module.

        Parameters
        ----------
        alpha : float
            The alpha scaling parameter of the transformation.
        beta_plus : float
            The beta parameter for data points greater than or equal to the median.
        lambda_plus : float
            The lambda parameter for data points greater than or equal to the median.
        beta_minus : float
            The beta parameter for data points less than the median.
        lambda_minus : float
            The lambda parameter for data points less than the median.
        median : float
            The median of the original data, used for centering the input tensor `X`.
        """
        super().__init__()
        
        # Register parameters as buffers. These are fixed during the module's lifetime
        # unless manually changed. They are moved with the module (e.g., .to(device)).
        self.register_buffer('alpha_val', torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer('beta_plus_val', torch.as_tensor(beta_plus, dtype=torch.float32))
        self.register_buffer('lambda_plus_val', torch.as_tensor(lambda_plus, dtype=torch.float32))
        self.register_buffer('beta_minus_val', torch.as_tensor(beta_minus, dtype=torch.float32))
        self.register_buffer('lambda_minus_val', torch.as_tensor(lambda_minus, dtype=torch.float32))
        self.register_buffer('median_val', torch.as_tensor(median, dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Applies the hyperbolic transformation to the input tensor.

        The transformation is applied element-wise. If `X` is not already a
        PyTorch tensor, it will be converted to one with `dtype=torch.float32`.

        Parameters
        ----------
        X : torch.Tensor or array-like
            Input data to be transformed. Can be of any shape.

        Returns
        -------
        torch.Tensor
            The transformed tensor, with the same shape and on the same device as input `X`
            (after potential initial conversion and move to module's device).
        """
        if not isinstance(X, torch.Tensor):
            # Convert to tensor, ensure it's float32, and move to the same device as the module's parameters.
            X = torch.as_tensor(X, dtype=torch.float32, device=self.alpha_val.device)
        elif X.device != self.alpha_val.device:
            # If X is already a tensor but on a different device, move it.
            X = X.to(self.alpha_val.device)


        X_centered = X - self.median_val
        
        # Initialize result tensor
        result = torch.zeros_like(X_centered, dtype=torch.float32)
        
        # Positive side
        positive_mask = X_centered >= 0
        if torch.any(positive_mask):
            x_pos = X_centered[positive_mask]
            # Avoid division by zero if beta_plus_val is very small
            if torch.abs(self.beta_plus_val) > 1e-9:
                cosh_beta_x_pos = torch.cosh(self.beta_plus_val * x_pos)
                result[positive_mask] = self.alpha_val * torch.sinh(self.beta_plus_val * x_pos) / \
                                        (cosh_beta_x_pos.pow(self.lambda_plus_val) * self.beta_plus_val)
            else: # Approximation for small beta * x (sinh(ax)/a -> x)
                 result[positive_mask] = self.alpha_val * x_pos


        # Negative side
        negative_mask = X_centered < 0
        if torch.any(negative_mask):
            x_neg = X_centered[negative_mask]
            # Avoid division by zero if beta_minus_val is very small
            if torch.abs(self.beta_minus_val) > 1e-9:
                cosh_beta_x_neg = torch.cosh(self.beta_minus_val * x_neg)
                result[negative_mask] = self.alpha_val * torch.sinh(self.beta_minus_val * x_neg) / \
                                        (cosh_beta_x_neg.pow(self.lambda_minus_val) * self.beta_minus_val)
            else: # Approximation for small beta * x
                result[negative_mask] = self.alpha_val * x_neg
                
        return result

if __name__ == '__main__':
    import numpy as np
    # For demonstrating parameter fitting, we might use HPTransformer
    # from hp_transformer import HPTransformer # Assuming it's in the same directory or installed

    print("--- HyperbolicTransformTorch Example ---")

    # Scenario 1: Using pre-defined parameters
    print("\nScenario 1: Using pre-defined parameters")
    alpha_s1 = 1.5
    beta_p_s1 = 0.8
    lambda_p_s1 = 0.7
    beta_n_s1 = 0.6
    lambda_n_s1 = 0.9
    median_s1 = 5.0

    # Create the transformation module
    torch_transformer_s1 = HyperbolicTransformTorch(
        alpha=alpha_s1,
        beta_plus=beta_p_s1, lambda_plus=lambda_p_s1,
        beta_minus=beta_n_s1, lambda_minus=lambda_n_s1,
        median=median_s1
    )

    # Example input tensor (can also be a NumPy array or list)
    data_s1 = torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0], dtype=torch.float32)
    
    # Transform the data
    transformed_s1 = torch_transformer_s1(data_s1)
    
    print(f"Original Data (Scenario 1): {data_s1.numpy()}")
    print(f"Parameters: alpha={alpha_s1}, beta_p={beta_p_s1}, lambda_p={lambda_p_s1}, "
          f"beta_n={beta_n_s1}, lambda_n={lambda_n_s1}, median={median_s1}")
    print(f"Transformed Data (Scenario 1): {np.round(transformed_s1.detach().cpu().numpy(), 3)}")

    # Scenario 2: Simulating parameters from a fitted HPTransformer
    # This part is conceptual if HPTransformer is not directly used here to avoid complex dependency in example.
    # print("\nScenario 2: Using parameters conceptually from a fitted HPTransformer")
    # sample_data_for_fitting = np.random.lognormal(mean=1, sigma=0.8, size=200)
    # hp_sklearn_transformer = HPTransformer()
    # hp_sklearn_transformer.fit(sample_data_for_fitting)
    # fitted_params = hp_sklearn_transformer.params_
    # fitted_median = hp_sklearn_transformer.median_
    # print(f"Fitted sklearn params: {fitted_params}, median: {fitted_median}")
    
    # For demonstration, let's use some example "fitted" parameters:
    alpha_s2 = 1.21
    beta_p_s2 = 0.65
    lambda_p_s2 = 0.88
    beta_n_s2 = 0.53
    lambda_n_s2 = 0.76
    median_s2 = np.median(np.random.lognormal(mean=1, sigma=0.8, size=200)) # Example median

    torch_transformer_s2 = HyperbolicTransformTorch(
        alpha=alpha_s2,
        beta_plus=beta_p_s2, lambda_plus=lambda_p_s2,
        beta_minus=beta_n_s2, lambda_minus=lambda_n_s2,
        median=median_s2
    )
    data_s2 = torch.linspace(-10, 20, 7, dtype=torch.float32)
    transformed_s2 = torch_transformer_s2(data_s2)
    print(f"\nOriginal Data (Scenario 2): {np.round(data_s2.numpy(), 2)}")
    print(f"Parameters (Scenario 2): alpha={alpha_s2:.2f}, beta_p={beta_p_s2:.2f}, lambda_p={lambda_p_s2:.2f}, "
          f"beta_n={beta_n_s2:.2f}, lambda_n={lambda_n_s2:.2f}, median={median_s2:.2f}")
    print(f"Transformed Data (Scenario 2): {np.round(transformed_s2.detach().cpu().numpy(), 3)}")


    # Scenario 3: GPU execution (if available)
    if torch.cuda.is_available():
        print("\nScenario 3: Running on GPU")
        device = torch.device("cuda")
        
        # Use parameters from Scenario 1 for GPU example
        torch_transformer_gpu = HyperbolicTransformTorch(
            alpha=alpha_s1, beta_plus=beta_p_s1, lambda_plus=lambda_p_s1,
            beta_minus=beta_n_s1, lambda_minus=lambda_n_s1, median=median_s1
        ).to(device) # Move module to GPU
        
        data_gpu = data_s1.to(device) # Move data to GPU
        
        transformed_gpu = torch_transformer_gpu(data_gpu)
        
        print(f"Original Data (GPU): {data_gpu.cpu().numpy()}") # Move back for printing if needed
        print(f"Transformed Data (GPU): {np.round(transformed_gpu.detach().cpu().numpy(), 3)}")
        print(f"Alpha buffer on device: {torch_transformer_gpu.alpha_val.device}")
        print(f"Output tensor on device: {transformed_gpu.device}")
    else:
        print("\nScenario 3: CUDA not available, skipping GPU example.")
    
    print("-" * 30)
