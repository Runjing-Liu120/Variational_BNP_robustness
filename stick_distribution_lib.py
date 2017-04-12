import numpy as np
from numpy import random
import scipy as sp
from scipy import special
from scipy import misc
from copy import deepcopy

class SticksVariationaDistribution(object):
    def __init__(self, k_max):
        self.k_max = k_max
        self.tau_1 = np.ones(k_max)
        self.tau_2 = np.ones(k_max)

    def set(self, tau_1, tau_2):
        assert len(tau_1) == len(tau_2), "tau_1 and tau_2 must be the same length"
        self.tau_1 = tau_1
        self.tau_2 = tau_2

        # When setting, cache expensive calculations that will be used over
        # and over again.
        self.tau_1_digamma = sp.special.digamma(self.tau_1)
        self.tau_2_digamma = sp.special.digamma(self.tau_2)
        self.tau_12_digamma = sp.special.digamma(self.tau_1 + self.tau_2)

        # These are the terms that appear in expressions indexed by y.
        # The empty sum -- the first element of tau_1_digamma_cumsum -- is zero.
        tau_1_digamma_cumsum = \
            np.insert(np.cumsum(self.tau_1_digamma[:-1]), 0, 0)
        tau_12_digamma_cumsum = np.cumsum(self.tau_12_digamma)
        self.y_log_prob_prop = \
            self.tau_2_digamma + tau_1_digamma_cumsum - tau_12_digamma_cumsum

    def get(self):
        return self.tau_1, self.tau_2

    def e_log(self):
        return self.tau_1_digamma - self.tau_12_digamma

    def e(self):
        return self.tau_1 / (self.tau_1 + self.tau_2)

    # The multinomial for the lower bound of E_q[log(1 - \prod_{m=1}^k \nu_m )]
    # except that, here, k is zero-indexed.
    def get_mn_bound_q(self, k):
        assert k < len(self.tau_1)
        y_log_prob = self.y_log_prob_prop[:k + 1]
        y_log_prob = y_log_prob - sp.misc.logsumexp(y_log_prob)
        y_prob = np.exp(y_log_prob)

        return y_prob, y_log_prob

    # The lower bound of E_q[log(1 - \prod_{m=1}^k \nu_m )]
    # except that, here, k is zero-indexed.
    def e_log_1_m_nu_prod(self, k):
        assert k < len(self.tau_1)
        y_prob, y_log_prob = self.get_mn_bound_q(k)
        y_entropy = -1.0 * np.sum(np.dot(y_prob, y_log_prob))
        return y_entropy + np.sum(np.dot(y_prob, self.y_log_prob_prop[:k + 1]))

    # Draw an array of betas with draws in columns and k in rows.
    def draw(self, num_draws):
        k_max = len(self.tau_1)
        draws = np.full((k_max, num_draws), float('nan'))
        for k in range(k_max):
            draws[k, :] = np.random.beta(self.tau_1[k], self.tau_2[k], num_draws)
        return draws
