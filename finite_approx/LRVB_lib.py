import autograd.numpy as np
import autograd.scipy as sp
from autograd.scipy import special
from autograd import grad, hessian, hessian_vector_product, hessian, jacobian

from copy import deepcopy

from scipy import optimize

import finite_approx.valez_finite_VI_lib as vi
import finite_approx.generic_optimization_lib as packing


class OptimzationTrace(object):
    def __init__(self, verbose=True):
        self.reset()
        self.print_every = 10
        self.verbose = verbose

    def reset(self):
        self.stepnum = 0
        self.objective = []
        self.params = []

    def update(self, params, objective):
        self.params.append(params)
        self.objective.append(objective)
        if self.verbose and self.stepnum % self.print_every == 0:
            print('Step {} objective: {}'.format(self.stepnum, objective))
        self.stepnum += 1


class DataSet(object):
    def __init__(self, x, k_approx, alpha, sigma_eps, sigma_a):
        self.x = x
        self.k_approx = k_approx
        self.alpha = alpha
        self.data_shape = {'D': x.shape[1], 'N': x.shape[0] , 'K':k_approx}
        self.sigmas = {'eps': sigma_eps, 'A': sigma_a}

        self.hyper_params = packing.pack_hyperparameters\
                        (self.alpha, self.sigmas['A'], self.sigmas['eps'])

        self.get_kl_grad =  grad(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hvp = hessian_vector_product(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hessian = hessian(
            lambda params: self.wrapped_kl(params, tracing=False))

        # It turns out to be much faster to take the gradient wrt the
        # small vector first.
        self.get_wrapped_kl_hyperparams_hyperparamgrad = \
            grad(self.wrapped_kl_hyperparams, argnum=1)
        self.get_kl_sens_hess = \
            jacobian(self.get_wrapped_kl_hyperparams_hyperparamgrad, argnum=0)

        self.moments_jac = jacobian(self.get_moments_vector)

        self.trace = OptimzationTrace()

    def unpack_params(self, params):
         return packing.unpack_params(
            params, self.data_shape['K'], self.data_shape['D'],
            self.data_shape['N'])

    def cavi_updates(self, tau, nu, phi_mu, phi_var):
        vi.cavi_updates(tau, nu, phi_mu, phi_var, \
                        self.x, self.alpha, self.sigmas)

    def wrapped_kl(self, params, tracing=True):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        elbo = vi.compute_elbo(tau, nu, phi_mu, phi_var, \
                               self.x, self.sigmas, self.alpha)
        if tracing:
            self.trace.update(params, -1 * elbo)
        return -1 * elbo

    def wrapped_kl_hyperparams(self, params, hyper_params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        alpha, sigma_a, sigma_eps = packing.unpack_hyperparameters(hyper_params)
        sigmas = {'eps': sigma_eps, 'A': sigma_a}
        elbo = vi.compute_elbo(tau, nu, phi_mu, phi_var, self.x, sigmas, alpha)
        return -1 * elbo

    def get_prediction(self, params):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        return np.matmul(nu, phi_mu.T)

    def run_cavi(self, tau, nu, phi_mu, phi_var, max_iter=200, tol=1e-6):
        params = packing.flatten_params(tau, nu, phi_mu, phi_var)

        self.trace.reset()
        diff = np.float('inf')
        while diff > tol and self.trace.stepnum < max_iter:
            self.cavi_updates(tau, nu, phi_mu, phi_var)
            new_params = packing.flatten_params(tau, nu, phi_mu, phi_var)
            diff = np.max(np.abs(new_params - params))
            self.trace.update(params, diff)
            if not np.isfinite(diff):
                print('Error: non-finite parameter difference.')
                break
            params = new_params

        if self.trace.stepnum >= max_iter:
            print('Warning: CAVI reached max_iter.')

        print('Done with CAVI.')
        return tau, nu, phi_mu, phi_var

    def run_newton_tr(self, params_init, maxiter=200, gtol=1e-6):
        self.trace.reset()
        self.tr_opt = optimize.minimize(
            lambda params: self.wrapped_kl(params, tracing=True),
            params_init, method='trust-ncg',
            jac=self.get_kl_grad,
            hessp=self.get_kl_hvp,
            tol=1e-6, options={'maxiter': maxiter, 'disp': True, 'gtol': gtol })

        print('Done with Newton trust region.')
        return self.tr_opt

    def get_moments(self, params):
        # Return moments of interest.
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        e_log_pi, e_log_pi2, e_mu, phi_moment2, nu_moment = \
            vi.get_moments(tau, nu, phi_mu, phi_var)
        return e_log_pi, e_mu

    def get_moments_vector(self, params):
        e_log_pi, e_mu = self.get_moments(params)
        return packing.pack_moments(e_log_pi, e_mu)

    ## begin LRVB computations
    def set_jacobians(self, params):
        self.moment_jac_set = self.moments_jac(params)
        self.kl_hess_set = self.get_kl_hessian(params)
        self.par_hp_hess_set = self.get_kl_sens_hess(params, self.hyper_params)
        self.kl_hess_inv_set = np.linalg.inv(self.kl_hess_set)

    def local_prior_sensitivity(self):
        try:
            sensitivity_operator = \
                        -1 * np.dot(self.kl_hess_inv_set, self.par_hp_hess_set.T)
            return np.matmul(self.moment_jac_set, sensitivity_operator)
        except AttributeError: # if the jacobians are not set yet, set them
            if hasattr(self, 'tr_opt'):
                self.set_jacobians(self.tr_opt.x)
                sensitivity_operator = \
                            -1 * np.dot(self.kl_hess_inv_set, self.par_hp_hess_set.T)
                return np.matmul(self.moment_jac_set, sensitivity_operator)
            else:
                raise ValueError(\
                    'Please run newton trust region to find an optima first')

    def influence_function_pi(self, theta, k):
        if not(hasattr(self, 'moment_jac_set')) or not(hasattr(self, 'kl_hess_inv_set')):
            # set jacobians if necessary
            if hasattr(self, 'tr_opt'):
                self.set_jacobians(self.tr_opt.x, self.hyper_params)
                print('hi')
            else:
                raise ValueError(\
                    'Please run newton trust region to find an optima first')

        log_q_pi_k_jac = jacobian(self.get_log_q_pi_k, 0)

        term1 = np.dot(self.moment_jac_set, self.kl_hess_inv_set)
        term2 = np.exp(self.get_log_q_pi_k(self.tr_opt.x, theta, k) \
                            - log_p0_pi(theta, self.alpha, self.k_approx))
        term3 = log_q_pi_k_jac(self.tr_opt.x, theta, k)

        return np.dot(term1, term2*term3)

    def get_log_q_pi_k(self, params, pi_k, k):
        tau, phi_mu, phi_var, nu = self.unpack_params(params)
        return log_q_pi(pi_k, np.array([tau[k,:]]))



    #def get_variational_log_lh(self, params, a, z, pi):
    #    tau, phi_mu, phi_var, nu = self.unpack_params(params)
    #    return log_q_a(a, phi_mu.T, phi_var) + log_q_z(z, nu) + log_q_pi(pi, tau)


##############
# same as above, but using VB library
# that is, vb_model and hyper_params are instances of the ModelParamsDict class

class DataSetII(object):
    def __init__(self, x, vb_model, hyper_params):
        self.vb_model = deepcopy(vb_model)
        self.hyper_params = deepcopy(hyper_params)
        self.x = x

        self.alpha = hyper_params['alpha'].get()
        self.sigmas = {'A': hyper_params['var_a'].get(),
                            'eps': hyper_params['var_eps'].get()}
        self.k_approx = np.shape(vb_model['phi'].e())[0]

        self.get_kl_grad =  grad(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hvp = hessian_vector_product(
            lambda params: self.wrapped_kl(params, tracing=False))
        self.get_kl_hessian = hessian(
            lambda params: self.wrapped_kl(params, tracing=False))

        # It turns out to be much faster to take the gradient wrt the
        # small vector first.
        self.get_wrapped_kl_hyperparams_hyperparamgrad = \
            grad(self.wrapped_kl_hyperparams, argnum=1)
        self.get_kl_sens_hess = \
            jacobian(self.get_wrapped_kl_hyperparams_hyperparamgrad, argnum=0)

        self.moments_jac = jacobian(self.get_moments_vector)

        self.trace = OptimzationTrace()

    def unpack_params(self, vb_model):
        phi_mu = vb_model['phi'].mean.get()
        phi_var = 1 / vb_model['phi'].info.get()
        nu = vb_model['nu'].get()
        tau = vb_model['pi'].alpha.get()

        return tau, phi_mu.T, phi_var, nu

    def cavi_updates(self, tau, nu, phi_mu, phi_var):
        vi.cavi_updates(tau, nu, phi_mu, phi_var, \
                        self.x, self.alpha, self.sigmas)

    def wrapped_kl(self, free_vb_params, tracing=True):
        self.vb_model.set_free(free_vb_params)
        elbo = vi.compute_elboII(self.x, self.vb_model, self.hyper_params)
        if tracing:
            self.trace.update(free_vb_params, -1 * elbo)
        return -1 * elbo

    def wrapped_kl_hyperparams(self, free_vb_params, free_hyper_params):
        self.vb_model.set_free(free_vb_params)
        self.hyper_params.set_free(free_hyper_params)
        elbo = vi.compute_elboII(self.x, self.vb_model, self.hyper_params)
        return -1 * elbo

    def get_prediction(self, vb_model):
        phi_mu = self.vb_model['phi'].e()
        nu = self.vb_model['nu'].get()
        return np.matmul(nu, phi_mu)

    def run_cavi(self, tau, nu, phi_mu, phi_var, max_iter=200, tol=1e-6):
        params = packing.flatten_params(tau, nu, phi_mu, phi_var)

        self.trace.reset()
        diff = np.float('inf')
        while diff > tol and self.trace.stepnum < max_iter:
            self.cavi_updates(tau, nu, phi_mu, phi_var)
            new_params = packing.flatten_params(tau, nu, phi_mu, phi_var)
            diff = np.max(np.abs(new_params - params))
            self.trace.update(params, diff)
            if not np.isfinite(diff):
                print('Error: non-finite parameter difference.')
                break
            params = new_params

        if self.trace.stepnum >= max_iter:
            print('Warning: CAVI reached max_iter.')

        print('Done with CAVI.')
        return tau, nu, phi_mu, phi_var


    def run_newton_tr(self, params_init, maxiter=200, gtol=1e-6):
        self.trace.reset()
        self.tr_opt = optimize.minimize(
            lambda params: self.wrapped_kl(params, tracing=True),
            params_init, method='trust-ncg',
            jac=self.get_kl_grad,
            hessp=self.get_kl_hvp,
            tol=1e-6, options={'maxiter': maxiter, 'disp': True, 'gtol': gtol })

        print('Done with Newton trust region.')
        return self.tr_opt

    def get_moments(self, free_vb_params):
        # Return moments of interest.
        self.vb_model.set_free(free_vb_params)
        e_log_pi1, e_log_pi2, phi_moment1, phi_moment2, nu_moment =\
                        vi.get_moments_VB(self.vb_model)
        return e_log_pi1, phi_moment1

    def get_moments_vector(self, free_vb_params):
        e_log_pi, e_mu = self.get_moments(free_vb_params)
        return packing.pack_moments(e_log_pi, e_mu)

    ## begin LRVB computations
    def set_jacobians(self, free_vb_params, free_hyper_params):
        self.moment_jac_set = self.moments_jac(free_vb_params)
        self.kl_hess_set = self.get_kl_hessian(free_vb_params)
        self.par_hp_hess_set = \
                self.get_kl_sens_hess(free_vb_params, free_hyper_params)
        self.kl_hess_inv_set = np.linalg.inv(self.kl_hess_set)

    def local_prior_sensitivity(self):
        try:
            sensitivity_operator = \
                    -1 * np.dot(self.kl_hess_inv_set, self.par_hp_hess_set.T)
            return np.matmul(self.moment_jac_set, sensitivity_operator)
        except AttributeError: # if the jacobians are not set yet, set them
            if hasattr(self, 'tr_opt'):
                print(self.hyper_params.get_vector())
                self.set_jacobians(self.tr_opt.x, self.hyper_params.get_free())
                sensitivity_operator = \
                    -1 * np.dot(self.kl_hess_inv_set, self.par_hp_hess_set.T)
                return np.matmul(self.moment_jac_set, sensitivity_operator)
            else:
                raise ValueError(\
                    'Please run newton trust region to find an optima first')

    def influence_function_pi(self, theta, k):
        if not(hasattr(self, 'moment_jac_set')) or not(hasattr(self, 'kl_hess_inv_set')):
            # set jacobians if necessary
            if hasattr(self, 'tr_opt'):
                self.set_jacobians(self.tr_opt.x, self.hyper_params.get_free())
            else:
                raise ValueError(\
                    'Please run newton trust region to find an optima first')

        log_q_pi_k_jac = jacobian(self.get_log_q_pi_k, 0)

        term1 = np.dot(self.moment_jac_set, self.kl_hess_inv_set)
        term2 = np.exp(self.get_log_q_pi_k(self.tr_opt.x, theta, k) \
                            - log_p0_pi(theta, self.alpha, self.k_approx))
        term3 = log_q_pi_k_jac(self.tr_opt.x, theta, k)

        return np.dot(term1, term2*term3)

    def get_log_q_pi_k(self, free_vb_params, pi_k, k):
        self.vb_model.set_free(free_vb_params)
        tau = self.vb_model['pi'].alpha.get()[k,:]
        return log_q_pi(pi_k, tau[None, :])


###################
# Variational likelihoods

# normal means
def log_q_a(a, phi_mu, phi_var):
    # note the shapes below: phi_mu should be the "correct",
    # not the transposed shape
    k_approx = np.shape(phi_mu)[0]
    x_d = np.shape(phi_mu)[1]
    assert np.shape(phi_mu)[0] == len(phi_var), 'shape of phi_var and phi_mu do not match'
    assert np.shape(a) == np.shape(phi_mu), 'shape of A and phi_mu do not match'

    log_denom = (x_d/2) * np.sum(np.log(2 * np.pi * phi_var))
    return -0.5 * np.sum((a.T - phi_mu.T) ** 2 / phi_var) - log_denom

# bernoulli responsibilities
def log_q_z(z, nu):
    return np.sum(np.log(nu**z) + np.log((1 - nu)**(1-z)))

# pi stick lengths
def log_q_pi(pi, tau):
    log_beta = sp.special.gammaln(tau[:,0]) + sp.special.gammaln(tau[:,1]) \
                    - sp.special.gammaln(tau[:,0] + tau[:,1])

    return np.sum(-log_beta + (tau[:,0] - 1.0) * np.log(pi) \
                    + (tau[:,1] - 1) * np.log(1.0 - pi))



####################
# Prior likelihoods
def log_p0_a(a, prior_var):
    prior_mean = np.zeros(np.shape(a))
    prior_var = np.full(np.shape(a)[0], prior_var)
    return log_q_a(a, prior_mean, prior_var)

# bernoulli responsibilities
def log_p0_z(z, pi):
    nu = np.tile(pi, (np.shape(z)[0], 1))
    return log_q_z(z, nu)

# pi stick lengths
def log_p0_pi(pi, alpha, k_approx):
    # tau = np.array([np.full(k_approx, 1), np.full(k_approx, alpha/k_approx)]).T
    tau = np.array([[1, alpha/k_approx]])
    return log_q_pi(pi, tau)

def log_p0_all(a, z, pi, alpha, k_approx, sigma_a):
    return log_p0_a(a, sigma_a) + log_p0_z(z, pi) \
            + log_p0_pi(pi, alpha, k_approx)

# delete the below: the above should take care of these cases
# just double check the indices/broadcasting works as desired above
##################
# Marginal distributions for pi
# here, pi_k refers to a scalar, not a vector
# and tau_k is two dimensional, corresponding to the two beta parameters
def log_q_pi_k(pi_k, tau_k):
    log_beta = sp.special.gammaln(tau_k[0]) + sp.special.gammaln(tau_k[1]) \
                    - sp.special.gammaln(tau_k[0] + tau_k[1])

    return -log_beta + (tau_k[0] - 1) * np.log(pi_k) \
                    + (tau_k[1] - 1) * np.log(1 - pi_k)

def log_p0_pi_k(pi_k, alpha, k_approx):
    tau_k = [1, alpha/k_approx]
    return log_q_pi_k(pi_k, tau_k)
