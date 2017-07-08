import unittest

import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import scipy as osp

from copy import deepcopy

import DP_normal_mixture_lib as dp

import sys
sys.path.append('../../LinearResponseVariationalBayes.py')

from VariationalBayes.ParameterDictionary import ModelParamsDict
from VariationalBayes.Parameters import ScalarParam, VectorParam, ArrayParam
from VariationalBayes.MultinomialParams import SimplexParam
from VariationalBayes.DirichletParams import DirichletParamArray
from VariationalBayes.MatrixParameters import PosDefMatrixParam, PosDefMatrixParamVector
from VariationalBayes.SparseObjectives import SparseObjective, Objective

np.random.seed(12312)

# DP parameters
x_dim = 2
k_approx = 5
num_obs = 100

# prior parameters
alpha = 1.2 # DP parameter
mu_prior = np.zeros(x_dim)
mu_prior_info = 1.0 * np.eye(x_dim)

info_x = 1.0 * np.eye(x_dim)

# draw data
x = dp.draw_data(alpha, mu_prior, mu_prior_info, info_x, \
                            x_dim, k_approx, num_obs)[0]

# set up vb model
global_params = ModelParamsDict('global')
global_params.push_param(
    PosDefMatrixParamVector(name='info', length=k_approx, matrix_size=x_dim)) # variational variances
global_params.push_param(
    ArrayParam(name='mu', shape=(k_approx, x_dim))) # variational means
global_params.push_param(
    DirichletParamArray(name='v_sticks', shape=(k_approx - 1, 2)))

local_params = ModelParamsDict('local')
local_params.push_param(
    SimplexParam(name='e_z', shape=(num_obs, k_approx)))

vb_params = ModelParamsDict('vb_params model')
vb_params.push_param(global_params)
vb_params.push_param(local_params)

# initialze
vb_params.set_free(np.random.random(vb_params.free_size()))

# get moments
e_log_v = vb_params['global']['v_sticks'].e_log()[:,0] # E[log v]
e_log_1mv = vb_params['global']['v_sticks'].e_log()[:,1] # E[log 1 - v]
e_z = vb_params['local']['e_z'].get()
mu = vb_params['global']['mu'].get()
info = vb_params['global']['info'].get()
tau = vb_params['global']['v_sticks'].alpha.get()

# draw variational samples
num_samples = 10**5
v_samples, pi_samples, mu_samples, z_samples = \
            dp.variational_samples(mu, info, tau, e_z, num_samples)


class TestElbo(unittest.TestCase):
    def assert_rel_close(self, x, y, rel_error = 0.02):
        err = np.abs((x - y) / x)
        self.assertTrue(err < rel_error)

    def test_dp_prior(self):
        dp_prior_computed = dp.dp_prior(alpha, e_log_1mv) \
                - e_log_1mv.shape[0] * osp.special.betaln(1, alpha)
        dp_prior_sampled = np.sum(np.mean(\
                osp.stats.beta.logpdf(v_samples, 1, alpha) , axis = 0))
        #print(dp_prior_computed)
        #print(dp_prior_sampled)

        self.assert_rel_close(dp_prior_computed, dp_prior_sampled)

    def test_normal_prior(self):
        normal_prior_computed = dp.normal_prior(mu, info, mu_prior, mu_prior_info)
        normal_prior_sampled = \
                np.mean([- 0.5 * np.trace(\
                    np.dot(np.dot(mu_samples[i,:,:] - mu_prior, mu_prior_info), \
                    (mu_samples[i,:,:] - mu_prior).T))
                    for i in range(mu_samples.shape[0])])
        # print(normal_prior_computed)
        # print(normal_prior_sampled)
        self.assert_rel_close(normal_prior_computed, normal_prior_sampled)

    def test_z_lh(self):
        z_lh_sampled = np.mean([np.sum(osp.stats.multinomial.logpmf(\
                    z_samples[i, : , :], n = 1, p = pi_samples[i, :])) for\
                    i in range(np.shape(z_samples)[0])])
        z_lh_computed = dp.loglik_ind(e_z, e_log_v, e_log_1mv)
        # print(z_lh_computed)
        # print(z_lh_sampled)

        self.assert_rel_close(z_lh_computed, z_lh_sampled)

    def test_data_lh(self):
        data_lh_samples = np.zeros(num_samples)
        for i in range(num_samples):
            x_center = x - np.dot(z_samples[i,:,:], mu_samples[i,:,:])
            data_lh_samples[i] = - 0.5 * np.trace(\
                    np.dot(np.dot(x_center, info_x), x_center.T))
        data_lh_sampled = np.mean(data_lh_samples)
        data_lh_computed = dp.loglik_obs(e_z, mu, info, x, info_x) \
                        - 0.5 * np.trace(np.dot(np.dot(x, info_x), x.T))
        # print(data_lh_sampled)
        # print(data_lh_computed)
        self.assert_rel_close(data_lh_sampled, data_lh_computed)


class TestCaviUpdate(unittest.TestCase):
    def test_z_update(self):

        # our manual update
        test_z_update = dp.z_update(mu, info, x, info_x, e_log_v, e_log_1mv)

        # autograd update
        get_auto_z_update = grad(dp.e_loglik_full, 6)
        auto_z_update = get_auto_z_update(
                x, mu, info, tau, e_log_v, e_log_1mv, e_z,
                mu_prior, mu_prior_info, info_x, alpha)
        log_const = sp.misc.logsumexp(auto_z_update, axis = 1)
        auto_z_update = np.exp(auto_z_update - log_const[:, None])

        # print(auto_z_update[0:5, :])
        # print(test_z_update[0:5, :])

        self.assertTrue(\
                np.sum(np.abs(auto_z_update - test_z_update)) <= 10**(-8))


if __name__ == '__main__':
    unittest.main()
