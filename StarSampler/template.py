



import numpy as np
from scipy import integrate

grav_const = 4.30091e-6  # kpc·(km/s)^2 / M_sun

class Model(object):
    def __init__(self, **model_param):
        '''
        Model for Osipkov-Merritt DF f(r, v_r, v_t) in a gNFW + Plummer system.
        model_param must include:
          rho0  : DM density normalization [M_sun/kpc^3]
          rs    : DM scale radius [kpc]
          gamma : DM inner slope
          r_star: Plummer scale radius [kpc]
          r_a   : anisotropy radius [kpc]
        '''
        # unpack parameters
        self.rho0   = model_param['rho0']
        self.rs     = model_param['rs']
        self.gamma  = model_param['gamma']
        self.r_star = model_param['r_star']
        self.r_a    = model_param['r_a']
        # Sampling is in (r, v_r, v_t)
        self.nX = 1
        self.nV = 2
        # radial range: 0 to 5*r_star
        self.Xlim = [0.0, 5.0 * self.r_star]
        # velocity range: 0 to sqrt(2*|phi(0)|)
        v_max = np.sqrt(2.0 * abs(self._phi(0.0)))
        self.Vlim = [0.0, v_max]
        self.sampler_input = [self.nX, self.nV, self.Xlim, self.Vlim]
        # Placeholder: could precompute lookup tables here

    def _rho_dm(self, r):
        # generalized NFW
        x = r / self.rs
        return self.rho0 * x**(-self.gamma) * (1.0 + x)**(-(3.0 - self.gamma))

    def _phi(self, r):
        '''Compute gravitational potential phi(r) (relative, positive depth).'''
        # phi(r) = G * (1/r * int_0^r 4pi s^2 rho(s) ds + int_r^infty 4pi s rho(s) ds)
        def integrand1(s): return 4.0 * np.pi * s**2 * self._rho_dm(s)
        def integrand2(s): return 4.0 * np.pi * s   * self._rho_dm(s)
        # inner term
        inner = integrate.quad(integrand1, 0.0, r, limit=100)[0] / r if r>0 else 0.0
        # outer term: integrate to some cutoff, e.g. 50*rs
        cutoff = 50.0 * self.rs
        outer = integrate.quad(integrand2, r, cutoff, limit=100)[0]
        return grav_const * (inner + outer)

    def DF(self, X, V):
        '''Return unnormalized DF f(r, v_r, v_t).'''
        r = np.asarray(X[0]) if isinstance(X, list) else np.asarray(X)
        vr = np.asarray(V[0])
        vt = np.asarray(V[1])
        # relative energy E = phi(r) - (vr^2 + vt^2)/2
        phi_r = np.vectorize(self._phi)(r)
        E = phi_r - 0.5 * (vr**2 + vt**2)
        # Q = E - L^2/(2 r_a^2) = E - (r vt)^2/(2 r_a^2)
        L2 = (r * vt)**2
        Q = E - L2 / (2.0 * self.r_a**2)
        # DF f(Q): Placeholder implementation
        # In practice, compute f(Q) via Abel transform (App. A1)
        # Here we simply zero-out unphysical Q.<br>
        f = np.zeros_like(Q)
        mask = Q > 0
        f[mask] = Q[mask]**1.5  # toy power-law placeholder
        return f






