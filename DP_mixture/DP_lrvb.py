import autograd.numpy as np
import autograd

from copy import deepcopy

import DP_functional_perturbation_lib as fun_pert

# This class examines the sensitivty to priors on alpha
class DPAlphaSensitivity(object):
    def __init__(self, model, optimal_global_free_params):
        self.model = deepcopy(model)
        self.alpha = model.alpha

        self.optimal_global_free_params = optimal_global_free_params

        # get the necessary derivatives
        self.get_kl_hessian = autograd.hessian(self.kl, argnum = 0)

        self.get_kl_jac = autograd.jacobian(self.kl, argnum = 0)
        self.get_alpha_jac = autograd.jacobian(self.get_kl_jac, argnum = 1)

        print('evaluating hessian ...')
        self.kl_hessian = self.get_kl_hessian(self.optimal_global_free_params, \
                                            self.alpha)
        print('ok')

    def kl(self, global_free_params, alpha):
        self.model.vb_params['global'].set_free(global_free_params)
        self.model.alpha = alpha

        return self.model.kl_optimize_z()

    def get_param_sensitivity(self, interesting_moments):
        # interesting moments should be a function that takes in the global
        # free parameters, and returns the posterior moments you want to examine

        get_moment_jac = autograd.jacobian(interesting_moments)
        moment_jac = get_moment_jac(self.optimal_global_free_params)

        sensitivity_operator = np.linalg.solve(self.kl_hessian, moment_jac.T)

        alpha_jac = self.get_alpha_jac(self.optimal_global_free_params,\
                                        self.alpha)

        return np.dot(sensitivity_operator.T, -1 * alpha_jac)

    def influence_function(self, theta, k, interesting_moments):

        get_moment_jac = autograd.jacobian(interesting_moments)
        moment_jac = get_moment_jac(self.optimal_global_free_params)

        sensitivity_operator = np.linalg.solve(self.kl_hessian, moment_jac.T)

        log_q_pi_k_jac = autograd.jacobian(self.get_log_q_pi_k, 0)

        g_bar = np.dot(sensitivity_operator.T, \
                log_q_pi_k_jac(self.optimal_global_free_params, theta, k).T)

        log_prior_density = lambda x : \
                fun_pert.log_q_pi(x, np.array([[1, self.alpha]]))

        ratio = np.exp(\
                self.get_log_q_pi_k(self.optimal_global_free_params, theta, k) \
                        - log_prior_density(theta))

        return ratio * g_bar

    def get_log_q_pi_k(self, global_free_params, theta, k):
        self.model.vb_params['global'].set_free(global_free_params)
        tau = self.model.vb_params['global']['v_sticks'].alpha.get()
        return fun_pert.log_q_pi(theta, np.array([tau[k,:]]))
