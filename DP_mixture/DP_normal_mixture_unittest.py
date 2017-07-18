import unittest

import autograd.numpy as np
import autograd.scipy as sp
from autograd import grad

import scipy as osp

from copy import deepcopy

import DP_normal_mixture_lib as dp
import DP_normal_mixture_opt_lib as dp_opt
import DP_functional_perturbation_lib as fun_pert

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
prior_mu = np.random.random(x_dim)
prior_info = np.random.random((x_dim, x_dim))
prior_info = np.dot(prior_info.T, prior_info)

info_x = np.random.random((x_dim, x_dim)) # 1.0 * np.eye(x_dim)
info_x = np.dot(info_x.T, info_x)

# draw data
x = dp.draw_data(info_x, x_dim, k_approx, num_obs)[0]


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
e_log_v, e_log_1mv, e_z, mu, mu2, info, tau = dp.get_vb_params(vb_params)

# draw variational samples
num_samples = 10**5
v_samples, pi_samples, mu_samples, z_samples = \
            dp.variational_samples(mu, info, tau, e_z, num_samples)


class TestElbo(unittest.TestCase):
    def assert_rel_close(self, x, y, std_error, deviations = 3):
        err = np.abs(x - y)
        self.assertTrue(err < std_error * deviations)

    def test_dp_prior(self):
        dp_prior_computed = dp.dp_prior(alpha, e_log_1mv) \
                - e_log_1mv.shape[0] * osp.special.betaln(1, alpha)
        dp_prior_samples_mean = np.sum(np.mean(\
                osp.stats.beta.logpdf(v_samples, 1, alpha) , axis = 0))

        print(dp_prior_computed)
        print(dp_prior_samples_mean)

        self.assertTrue(np.abs(dp_prior_computed - dp_prior_samples_mean) < 0.01)

    def test_normal_prior(self):
        normal_prior_computed = dp.normal_prior(mu, mu2, prior_mu, prior_info)\
            - 0.5 * np.dot(np.dot(prior_mu, prior_info), prior_mu) * k_approx
        normal_prior_samples = \
                [- 0.5 * np.trace(\
                    np.dot(np.dot(mu_samples[i,:,:] - prior_mu, prior_info), \
                    (mu_samples[i,:,:] - prior_mu).T))
                    for i in range(mu_samples.shape[0])]

        normal_prior_samples_mean = np.mean(normal_prior_samples)
        normal_prior_samples_std = np.std(normal_prior_samples)

        print(normal_prior_samples_mean)
        print(normal_prior_samples_std / np.sqrt(num_samples))
        print(normal_prior_computed)

        self.assert_rel_close(normal_prior_computed, normal_prior_samples_mean, \
                    normal_prior_samples_std / np.sqrt(num_samples))


    def test_z_lh(self):
        z_lh_samples = np.array([np.sum(osp.stats.multinomial.logpmf(\
                    z_samples[i, : , :], n = 1, p = pi_samples[i, :])) for\
                    i in range(np.shape(z_samples)[0])])
        z_lh_samples_mean = np.mean(z_lh_samples)
        z_lh_samples_std = np.std(z_lh_samples)

        z_lh_computed = dp.loglik_ind(e_z, e_log_v, e_log_1mv)

        self.assert_rel_close(z_lh_computed, z_lh_samples_mean, \
                    z_lh_samples_std / np.sqrt(num_samples))

    def test_data_lh(self):

        data_lh_samples = np.zeros(num_samples)
        for i in range(num_samples):
            x_center = x - np.dot(z_samples[i,:,:], mu_samples[i,:,:])
            data_lh_samples[i] = - 0.5 * np.trace(\
                    np.dot(np.dot(x_center, info_x), x_center.T))
        data_lh_samples_mean = np.mean(data_lh_samples)
        data_lh_samples_std = np.std(data_lh_samples)

        data_lh_computed = dp.loglik_obs(e_z, mu, mu2, x, info_x) \
                        - 0.5 * np.trace(np.dot(np.dot(x, info_x), x.T))

        print(data_lh_samples_mean)
        print(data_lh_samples_std / np.sqrt(num_samples))
        print(data_lh_computed)

        self.assert_rel_close(data_lh_samples_mean, data_lh_computed,
            data_lh_samples_std / np.sqrt(num_samples))


class TestCaviUpdates(unittest.TestCase):
    def test_z_update(self):

        # our manual update
        test_z_update = dp.z_update(mu, mu2, x, info_x, e_log_v, e_log_1mv)

        # autograd update
        get_auto_z_update = grad(dp.e_loglik_full, 6)
        auto_z_update = get_auto_z_update(
                x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                prior_mu, prior_info, info_x, alpha)
        log_const = sp.misc.logsumexp(auto_z_update, axis = 1)
        auto_z_update = np.exp(auto_z_update - log_const[:, None])

        # print(auto_z_update[0:5, :])
        # print(test_z_update[0:5, :])

        self.assertTrue(\
                np.sum(np.abs(auto_z_update - test_z_update)) <= 10**(-8))

    def test_tau_update(self):
        # our manual update
        test_tau_update = dp_opt.tau_update(e_z, alpha)

        get_tau1_update = grad(dp.e_loglik_full, 4)
        get_tau2_update = grad(dp.e_loglik_full, 5)

        auto_tau1_update = get_tau1_update(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                            prior_mu, prior_info, info_x, alpha) + 1
        auto_tau2_update = get_tau2_update(x, mu, mu2, tau, e_log_v, e_log_1mv, e_z,
                            prior_mu, prior_info, info_x, alpha) + 1

        self.assertTrue(\
                np.sum(np.abs(test_tau_update[:,0] - auto_tau1_update)) <= 10**(-8))
        self.assertTrue(\
                np.sum(np.abs(test_tau_update[:,1] - auto_tau2_update)) <= 10**(-8))


class TestFunctionalPerturbation(unittest.TestCase):
    def u(self, x):
        return(3.0 * x)

    def test_integration(self):
        test = fun_pert.dp_prior_perturbed\
                            (tau, alpha, self.u, n_grid = 10**6)

        truth = 0
        for k in range(k_approx - 1):
            integrand = lambda x : osp.stats.beta.pdf(x, tau[k,0], tau[k,1]) * \
                    np.log(self.u(x) + osp.stats.beta.pdf(x, 1, alpha))

            truth += osp.integrate.quad(integrand, 0, 1)[0]

        self.assertTrue(np.abs(test - truth) <= 10**(-6))

    def test_moment(self):
        # test against moment (without pertubation)
        true_moment = dp.dp_prior(alpha, e_log_1mv) -\
                        e_log_1mv.shape[0] * osp.special.betaln(1, alpha)

        true_moment2 = (alpha - 1) * (osp.special.digamma(tau[:,1]) - \
                        osp.special.digamma(tau[:,0] + tau[:,1])) -\
                        osp.special.betaln(1, alpha)

        test_moment = fun_pert.dp_prior_perturbed\
                            (tau, alpha, n_grid = 10**6)
                            # here, recall that the default perturbation is 0

        print(np.abs(true_moment - test_moment))
        self.assertTrue(np.abs(true_moment - test_moment) <= 10**(-6))

    def test_elbo(self):
        # without perturbation, check against old elbo computations
        test_elbo = fun_pert.compute_elbo_perturbed(\
                x, mu, mu2, info, tau, e_log_v, e_log_1mv, e_z,
                prior_mu, prior_info, info_x, alpha, n_grid = 10**6)

        true_elbo = dp.compute_elbo(x, mu, mu2, info, tau, e_log_v, e_log_1mv,
                    e_z, prior_mu, prior_info, info_x, alpha) - \
                    (k_approx - 1) * osp.special.betaln(1, alpha)

        print(np.abs(test_elbo - true_elbo))
        self.assertTrue(np.abs(test_elbo - true_elbo) <= 10**(-6))

if __name__ == '__main__':
    unittest.main()
