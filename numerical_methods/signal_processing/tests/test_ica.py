"""Test cases for Independent Component Analysis."""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from numerical_methods.signal_processing.ica import center, whiten, ica, create_signals


class TestIndependentComponentAnalysis(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        # Test Case 1 from problem description
        X1 = np.array([
            [-4., -1.25837414, -4.2834508, 4.22567322, 1.43150983, -6.28790332],
            [-4., -3.22918707, -6.3417254, 6.31283661, 3.31575491, -8.14395166],
            [-8., -0.48756122, -6.62517619, 6.53850983, 0.74726474, -10.43185497]
        ])

        # Test Case 2 from problem description
        X2 = np.array([
            [-4., -4.10199583, -0.70436724, -2.02846889, 2.84962972, -1.19342653, 5.76905316, -6.28790332],
            [-4., -6.47956934, 1.79067353, -4.29994873, 4.71052915, -2.73957041, 7.31309801, -8.14395166],
            [-8., -6.58156517, -2.91369371, -2.32841762, 3.56015887, 0.06700306, 9.08215117, -10.43185497]
        ])

        # Test Case 3 from problem description
        X3 = np.array([
            [-4.00000000e+00, 6.08976682e+00, -1.80018426e-01, 2.52000394e+00, -8.19025595e-01, 2.06616123e+00, -4.27972909e+00, -4.34384652e+00, -1.14726131e-01, -6.28790332e+00],
            [-4.00000000e+00, 7.60043896e+00, -1.97889810e+00, 4.92666864e+00, -3.18729058e+00, 3.81085839e+00, -5.80653121e+00, -6.28303437e+00, 1.38708138e+00, -8.14395166e+00],
            [-8.00000000e+00, 9.69020578e+00, 1.84108347e+00, 3.44667258e+00, -6.31617101e-03, 1.87701963e+00, -6.08626030e+00, -6.62688090e+00, -2.72764475e+00, -1.04318550e+01]
        ])

        self.test_matrices = [X1, X2, X3]

    def test_subproblem_01_center(self):
        """Test Subproblem 1: Centering and Standardization with specific inputs"""
        for i, X in enumerate(self.test_matrices):
            with self.subTest(test_case=i+1):
                # 1. Test standard behavior (divide_sd=True)
                D = center(X, divide_sd=True)

                # Check means are 0
                means = np.mean(D, axis=1)
                self.assertTrue(np.allclose(means, 0, atol=1e-7), f"Case {i+1}: Means should be 0, got {means}")

                # Check standard deviations are 1
                stds = np.std(D, axis=1)
                self.assertTrue(np.allclose(stds, 1, atol=1e-7), f"Case {i+1}: Stds should be 1, got {stds}")

                # Verify shape is preserved
                self.assertEqual(D.shape, X.shape)

                # 2. Test centering only (divide_sd=False)
                D_no_sd = center(X, divide_sd=False)
                means_no_sd = np.mean(D_no_sd, axis=1)
                self.assertTrue(np.allclose(means_no_sd, 0, atol=1e-7), f"Case {i+1} (no_sd): Means should be 0")

    def test_subproblem_02_whiten(self):
        """Test Subproblem 2: Whitening with specific inputs"""
        for i, X in enumerate(self.test_matrices):
            with self.subTest(test_case=i+1):
                Z = whiten(X)

                # Check covariance is Identity
                cov = np.cov(Z)
                identity = np.eye(cov.shape[0])

                self.assertTrue(np.allclose(cov, identity, atol=1e-7),
                                f"Case {i+1}: Covariance should be Identity.\nGot:\n{cov}")

                # Check shape is preserved
                self.assertEqual(Z.shape, X.shape)

    def test_subproblem_03_ica(self):
        """Test Subproblem 3: ICA with specific inputs and synthetic data"""

        # 1. Test running ICA on specific inputs (smoke test)
        for i, X in enumerate(self.test_matrices):
            with self.subTest(test_case=f"Explicit Input {i+1}"):
                np.random.seed(0)
                S_hat = ica(X, cycles=200, tol=1e-5)
                # Ensure output shape is correct
                self.assertEqual(S_hat.shape, X.shape)

        # 2. Test ICA signal recovery on synthetic data (Test Case 4)
        with self.subTest(test_case="Synthetic Data"):
            X, S_original = create_signals(N=2000)

            # Run ICA
            np.random.seed(0)
            S_hat = ica(X, cycles=200, tol=1e-5)

            self.assertEqual(S_hat.shape, S_original.shape)

            # Check correlation to verify recovery (ignoring order and sign)
            n_sources = S_original.shape[0]
            correlations = np.zeros((n_sources, n_sources))

            for i in range(n_sources):
                for j in range(n_sources):
                    corr = np.corrcoef(S_hat[i], S_original[j])[0, 1]
                    correlations[i, j] = np.abs(corr)

            # Best match for each original source
            max_corrs = np.max(correlations, axis=0)

            print(f"\nMax correlations per source (Synthetic Test): {max_corrs}")

            # We expect high correlation for all sources
            self.assertTrue(np.all(max_corrs > 0.9),
                            f"ICA failed to recover sources. Max correlations: {max_corrs}")


if __name__ == "__main__":
    unittest.main()
