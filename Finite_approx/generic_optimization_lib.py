import autograd.numpy as np
import autograd.scipy as sp

def pack_tau(tau):
    return np.log(tau).flatten()

def unpack_tau(tau_packed, k_approx):
    return np.exp(tau_packed).reshape((k_approx))

def pack_phi_mu(phi_mu):
    return phi_mu.flatten()

def unpack_phi_mu(phi_mu_packed, k_approx, x_dim):
    return phi_mu_packed.reshape((x_dim, k_approx))

def pack_phi_var(phi_var):
    return np.log(phi_var.flatten())

def unpack_phi_var(phi_var_packed):
    return np.exp(phi_var_packed)

def pack_nu(nu):
    return sp.special.logit(nu).flatten()

def unpack_nu(nu_packed, num_samples, k_approx):
    return sp.special.expit(nu_packed).reshape(num_samples, k_approx)

def pack_params(tau, phi_mu, phi_var, nu):
    return np.hstack([ pack_tau(tau), pack_phi_mu(phi_mu),
                       pack_phi_var(phi_var), pack_nu(nu) ])

def unpack_params(params, k_approx, x_dim, num_samples):
    offset = 0

    tau_size = k_approx * 2
    phi_mu_size = k_approx * x_dim
    phi_var_size = k_approx
    nu_size = num_samples * k_approx

    assert len(params) == tau_size + phi_mu_size + phi_var_size + nu_size

    tau = unpack_tau(params[offset:(offset + tau_size)], k_approx, x_dim)
    offset += tau_size

    phi_mu = unpack_phi_mu(params[offset:(offset + phi_mu_size)], k_approx, x_dim)
    offset += phi_mu_size

    phi_var = unpack_phi_var(params[offset:(offset + phi_var_size)])
    offset += phi_var_size

    nu = unpack_nu(params[offset:(offset + nu_size)], num_samples, k_approx)

    return tau, phi_mu, phi_var, nu


def pack_hyperparameters(alpha, sigma_a, sigma_eps):
    return np.array([ alpha, sigma_a, sigma_eps ])


def unpack_hyperparameters(hyper_params):
    alpha = hyper_params[0]
    sigma_a = hyper_params[1]
    sigma_eps = hyper_params[2]
    return alpha, sigma_a, sigma_eps


# Stack params in a vector without constraining them.
def flatten_params(tau, nu, phi_mu, phi_var):
    return np.hstack([ tau.flatten(), nu.flatten(),
                       phi_mu.flatten(), phi_var.flatten() ])


def unflatten_params(params, k_approx, x_dim, num_samples):
    tau_size = k_approx * 2
    phi_mu_size = k_approx * x_dim
    phi_var_size = k_approx
    nu_size = num_samples * k_approx

    assert len(params) == tau_size + phi_mu_size + phi_var_size + nu_size

    offset = 0

    tau = params[offset:(offset + tau_size)].reshape(k_approx, 2)
    offset += tau_size

    nu = params[offset:(offset + nu_size)].reshape(num_samples, k_approx)
    offset += nu_size

    phi_mu = params[offset:(offset + phi_mu_size)].reshape(x_dim, k_approx)
    offset += phi_mu_size

    phi_var = params[offset:(offset + phi_var_size)]
    offset += phi_var_size

    return tau, phi_mu, phi_var, nu


def pack_moments(e_log_pi, e_mu):
    return np.hstack([ e_log_pi, e_mu.flatten() ])


def unpack_moments(param, k_approx, x_dim):
    pi_size = k_approx
    mu_size = k_approx * x_dim
    assert len(param) == pi_size + mu_size
    offset = 0

    e_log_pi = param[offset:(offset + pi_size)]
    offset += pi_size

    e_mu = param[offset:(offset + mu_size)].reshape(x_dim, k_approx)

    return e_log_pi, e_mu
