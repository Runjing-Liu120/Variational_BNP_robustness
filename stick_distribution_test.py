import numpy as np
import numpy.testing as np_test
from stick_distribution_lib import SticksVariationaDistribution
import unittest

class TestParameters(unittest.TestCase):
    def test_sticks(self):
        np.random.seed(42)
        tau_1 = np.array([2., 3., 4.])
        tau_2 = np.array([3., 4., 5.])
        k_max = len(tau_1)

        sticks = SticksVariationaDistribution(k_max)
        sticks.set(tau_1, tau_2)
        tau_1_test, tau_2_test = sticks.get()
        np_test.assert_array_almost_equal(tau_1_test, tau_1)
        np_test.assert_array_almost_equal(tau_2_test, tau_2)

        draws = sticks.draw(100000)
        stick_means = np.mean(draws, 1)
        np_test.assert_allclose(stick_means, sticks.e(), 1e-3)

        stick_log_means = np.mean(np.log(draws), 1)
        np_test.assert_allclose(stick_log_means, sticks.e_log(), 1e-3)

        # Test the lower bounds.
        for k in range(k_max):
            y_prob, y_log_prob = sticks.get_mn_bound_q(k)
            self.assertAlmostEqual(np.sum(y_prob), 1.0)
            np_test.assert_array_almost_equal(y_log_prob, np.log(y_prob))

            lb = sticks.e_log_1_m_nu_prod(k)
            draws_prod = np.prod(draws[:k + 1, :], 0)
            true_e = np.mean(np.log(1 - draws_prod))
            self.assertTrue(lb <= true_e)


if __name__ == '__main__':
    unittest.main()
