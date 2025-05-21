import unittest
import numpy as np
import torch
from numpy.testing import assert_allclose

# Module to be tested
from hyperbolic_torch import HyperbolicTransformTorch

# For comparison and parameter generation (optional, can use fixed params)
from hp_transformer import HPTransformer 
from hyperbolic_np import hp_transform_np

# Define a fixed set of realistic parameters for consistent testing
# These could be obtained from a previously fitted HPTransformer instance
FIXED_PARAMS = {
    "alpha": 1.5,
    "beta_plus": 0.8,
    "lambda_plus": 0.7,
    "beta_minus": 0.6,
    "lambda_minus": 0.9,
    "median": 10.0
}

# Helper function to get a HPTorch module with fixed params
def get_torch_module(params=None):
    if params is None:
        params = FIXED_PARAMS
    return HyperbolicTransformTorch(
        alpha=params["alpha"],
        beta_plus=params["beta_plus"],
        lambda_plus=params["lambda_plus"],
        beta_minus=params["beta_minus"],
        lambda_minus=params["lambda_minus"],
        median=params["median"]
    )

class TestHyperbolicTransformTorch(unittest.TestCase):

    def test_instantiation_and_buffer_types(self):
        module = get_torch_module()
        self.assertIsInstance(module, HyperbolicTransformTorch)
        
        self.assertIsInstance(module.alpha_val, torch.Tensor)
        self.assertEqual(module.alpha_val.dtype, torch.float32)
        self.assertAlmostEqual(module.alpha_val.item(), FIXED_PARAMS["alpha"])

        self.assertIsInstance(module.beta_plus_val, torch.Tensor)
        self.assertEqual(module.beta_plus_val.dtype, torch.float32)
        self.assertAlmostEqual(module.beta_plus_val.item(), FIXED_PARAMS["beta_plus"])

        self.assertIsInstance(module.lambda_plus_val, torch.Tensor)
        self.assertEqual(module.lambda_plus_val.dtype, torch.float32)
        self.assertAlmostEqual(module.lambda_plus_val.item(), FIXED_PARAMS["lambda_plus"])

        self.assertIsInstance(module.beta_minus_val, torch.Tensor)
        self.assertEqual(module.beta_minus_val.dtype, torch.float32)
        self.assertAlmostEqual(module.beta_minus_val.item(), FIXED_PARAMS["beta_minus"])
        
        self.assertIsInstance(module.lambda_minus_val, torch.Tensor)
        self.assertEqual(module.lambda_minus_val.dtype, torch.float32)
        self.assertAlmostEqual(module.lambda_minus_val.item(), FIXED_PARAMS["lambda_minus"])

        self.assertIsInstance(module.median_val, torch.Tensor)
        self.assertEqual(module.median_val.dtype, torch.float32)
        self.assertAlmostEqual(module.median_val.item(), FIXED_PARAMS["median"])

    def _test_forward_pass_consistency(self, data_np):
        module = get_torch_module()
        
        # Numpy reference
        expected_np = hp_transform_np(
            data_np,
            FIXED_PARAMS["alpha"],
            FIXED_PARAMS["beta_plus"],
            FIXED_PARAMS["lambda_plus"],
            FIXED_PARAMS["beta_minus"],
            FIXED_PARAMS["lambda_minus"],
            FIXED_PARAMS["median"]
        )
        
        # PyTorch module forward pass
        data_torch = torch.as_tensor(data_np, dtype=torch.float32)
        output_torch = module(data_torch)
        
        self.assertIsInstance(output_torch, torch.Tensor)
        self.assertEqual(output_torch.shape, data_torch.shape)
        assert_allclose(output_torch.detach().cpu().numpy(), expected_np, rtol=1e-5, atol=1e-6)

    def test_forward_pass_1d_mixed_data(self):
        data = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float32) # Median is 10
        self._test_forward_pass_consistency(data)

    def test_forward_pass_1d_positive_data(self):
        data = np.array([11.0, 15.0, 20.0, 25.0], dtype=np.float32) # All > median
        self._test_forward_pass_consistency(data)

    def test_forward_pass_1d_negative_data(self):
        data = np.array([-5.0, 0.0, 2.0, 8.0], dtype=np.float32) # All < median
        self._test_forward_pass_consistency(data)
        
    def test_forward_pass_2d_mixed_data(self):
        data = np.array([[-5.0, 0.0], [5.0, 10.0], [15.0, 20.0]], dtype=np.float32)
        self._test_forward_pass_consistency(data)

    def test_forward_pass_input_type_conversion(self):
        module = get_torch_module()
        data_np = np.array([5.0, 10.0, 15.0], dtype=np.float64) # Different dtype
        # Module should convert this to float32 tensor
        output_torch = module(data_np) 
        self.assertEqual(output_torch.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping GPU test")
    def test_gpu_execution(self):
        module = get_torch_module()
        module.to('cuda') # Move module to GPU
        
        # Check if buffers are on GPU
        self.assertEqual(module.alpha_val.device.type, 'cuda')

        data_np = np.array([-5.0, 10.0, 25.0], dtype=np.float32)
        data_gpu = torch.as_tensor(data_np, dtype=torch.float32).to('cuda')
        
        output_gpu = module(data_gpu)
        self.assertEqual(output_gpu.device.type, 'cuda')
        
        # Compare with CPU execution for consistency
        module_cpu = get_torch_module()
        data_cpu = torch.as_tensor(data_np, dtype=torch.float32)
        output_cpu = module_cpu(data_cpu)
        
        assert_allclose(output_gpu.detach().cpu().numpy(), output_cpu.detach().cpu().numpy(), rtol=1e-5, atol=1e-6)

    def test_forward_empty_input(self):
        module = get_torch_module()
        data_empty_np = np.array([], dtype=np.float32)
        data_empty_torch = torch.as_tensor(data_empty_np)
        
        output_torch = module(data_empty_torch)
        self.assertEqual(output_torch.numel(), 0)
        self.assertEqual(output_torch.shape, data_empty_torch.shape)

        # Also test with 2D empty tensor (e.g. shape (0, 2))
        data_empty_2d_torch = torch.empty((0,2), dtype=torch.float32)
        output_2d_torch = module(data_empty_2d_torch)
        self.assertEqual(output_2d_torch.numel(), 0)
        self.assertEqual(output_2d_torch.shape, data_empty_2d_torch.shape)


    def test_forward_beta_near_zero(self):
        # Test positive side with beta_plus near zero
        params_beta_p_nz = FIXED_PARAMS.copy()
        params_beta_p_nz["beta_plus"] = 1e-10
        module_p_nz = get_torch_module(params_beta_p_nz)
        
        data_np = np.array([12.0, 15.0], dtype=np.float32) # X > median (10.0)
        # Expected: alpha * (X - median)
        expected_p_nz = params_beta_p_nz["alpha"] * (data_np - params_beta_p_nz["median"])
        
        output_torch_p_nz = module_p_nz(torch.as_tensor(data_np))
        assert_allclose(output_torch_p_nz.detach().cpu().numpy(), expected_p_nz, rtol=1e-5)

        # Test negative side with beta_minus near zero
        params_beta_n_nz = FIXED_PARAMS.copy()
        params_beta_n_nz["beta_minus"] = 1e-10
        module_n_nz = get_torch_module(params_beta_n_nz)

        data_np_neg = np.array([2.0, 5.0], dtype=np.float32) # X < median (10.0)
        # Expected: alpha * (X - median)
        expected_n_nz = params_beta_n_nz["alpha"] * (data_np_neg - params_beta_n_nz["median"])
        
        output_torch_n_nz = module_n_nz(torch.as_tensor(data_np_neg))
        assert_allclose(output_torch_n_nz.detach().cpu().numpy(), expected_n_nz, rtol=1e-5)

    def test_forward_with_fitted_hp_transformer_params(self):
        # 1. Fit HPTransformer on some data
        source_data = np.random.lognormal(0, 1, size=200) + 1.0 # Ensure positive
        hp_sklearn = HPTransformer()
        hp_sklearn.fit(source_data)

        # 2. Get parameters from fitted HPTransformer
        alpha, beta_m, beta_p, lambda_m, lambda_p = hp_sklearn.params
        median = hp_sklearn.median
        
        # 3. Instantiate HyperbolicTransformTorch with these parameters
        hpt_torch = HyperbolicTransformTorch(
            alpha=alpha, beta_plus=beta_p, lambda_plus=lambda_p,
            beta_minus=beta_m, lambda_minus=lambda_m, median=median
        )

        # 4. Test data
        test_data_np = np.array([-5.0, 0.0, median-1.0, median, median+1.0, 20.0, 30.0], dtype=np.float32)
        test_data_torch = torch.from_numpy(test_data_np)

        # 5. Get output from both
        expected_transform_np = hp_transform_np(test_data_np, alpha, beta_p, lambda_p, beta_m, lambda_m, median)
        torch_output = hpt_torch(test_data_torch).detach().cpu().numpy()

        assert_allclose(torch_output, expected_transform_np, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
