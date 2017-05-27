import autograd.numpy as np
import autograd.scipy as sp


def pack_tau(tau):
    return np.log(tau).flatten()

def unpack_tau(tau_packed, K_approx, D):
    return np.exp(tau_packed).reshape((K_approx, D))

def pack_phi_mu(phi_mu):
    return phi_mu.flatten()

def unpack_phi_mu(phi_mu_packed, K_approx, D):
    return phi_mu_packed.reshape((D, K_approx))

def pack_phi_var(phi_var):
    return np.log(phi_var.flatten())

def unpack_phi_var(phi_var_packed):
    return np.exp(phi_var_packed)

def pack_nu(nu):
    return sp.special.logit(nu).flatten()

def unpack_nu(nu_packed, Num_samples, K_approx):
    return sp.special.expit(nu_packed).reshape(Num_samples, K_approx)

def pack_params(tau, phi_mu, phi_var, nu):
    return np.hstack([ pack_tau(tau), pack_phi_mu(phi_mu),
                       pack_phi_var(phi_var), pack_nu(nu) ])

def unpack_params(params, K_approx, D, Num_samples):
    offset = 0

    tau_size = K_approx * D
    phi_mu_size = K_approx * D
    phi_var_size = K_approx
    nu_size = Num_samples * K_approx

    assert len(params) == tau_size + phi_mu_size + phi_var_size + nu_size

    tau = unpack_tau(params[offset:(offset + tau_size)], K_approx, D)
    offset += tau_size

    phi_mu = unpack_phi_mu(params[offset:(offset + phi_mu_size)], K_approx, D)
    offset += phi_mu_size

    phi_var = unpack_phi_var(params[offset:(offset + phi_var_size)])
    offset += phi_var_size

    nu = unpack_nu(params[offset:(offset + nu_size)], Num_samples, K_approx)

    return tau, phi_mu, phi_var, nu
