import numpy as np
import scipy.integrate

def amplitude_sq_R(Pk, R, volume, maxiter=100):
    """
    Pk: power spectrum as function of k (h/Mpc), e.g. interpolation.
    R: scale (Mpc/h) at which to calculate amplitude of fluctuations \sigma_R.
    volume: physical volume of box (Mpc^3 h^{-3}), used for normalizing integral
    """
    kSwitch = 2*np.pi/R

    integrant = lambda k: Pk(k) * top_hat_3D_Fourier(k, R) * 4.0*np.pi*k**2
    integrantInv = lambda k: Pk(1./k) * top_hat_3D_Fourier(1./k, R) * 4.0*np.pi*(1./k)**4

    s1 = scipy.integrate.quadrature(integrant, 0, kSwitch, maxiter=maxiter)[0]
    s2 = scipy.integrate.quadrature(integrantInv, 1e-30, 1/kSwitch,
                                    maxiter=maxiter)[0]

    # The volume/(2pi)^3 is a normalization of the convolution (the integral)
    # that is used to normalize the power spectrum. (s1+s2)*volume/(2pi)^3 is
    # the sigma0^2 as calculated from the given power spectrum.
    return volume / (2*np.pi)**3 * (s1 + s2)


def normalize(Pk, Rth, sigma0, volume, maxiter=100):
    """Normalize the power spectrum for the periodic field in box of
    /volume/ that it will be used on, on a top-hat scale Rth with sigma0.
    /volume/ must have units h^-3 Mpc^3."""
    
    return sigma0**2 / amplitude_sq_R(Pk, Rth, volume, maxiter=maxiter)


# def moment(self, order, Rg, volume, maxiter=100):
#     """Calculate the spectral moment of order /order/ of the power spectrum
#     by convolving with a Gaussian window of radius /Rg/ over the power
#     spectrum."""
#     amp = self.amplitude # we want to convolve the normalized power spectrum
#     kSwitch = 2*np.pi/Rg
#     s1 = integrate(self.moment_integrant, 0, kSwitch, \
#          args = (order, Rg), maxiter=maxiter)[0]
#     s2 = integrate(self.moment_integrantInv, 1e-30, 1/kSwitch, \
#          args = (order, Rg), maxiter=maxiter)[0]
#     return np.sqrt( amp * (s1+s2) * volume / (2*np.pi)**3 )

# def integrant(Pk, k, R):
#     """Integrand used to determine power spectrum amplitude"""
#     return Pk(k) * top_hat_3D_Fourier(k, R) * 4.0*np.pi*k**2
    
# def integrantInv(Pk, k, R):
#     """Inverse of integrand used to determine power spectrum amplitude"""
#     k = 1.0/k
#     return Pk(k) * top_hat_3D_Fourier(k, R) * 4.0*np.pi*k**4

# def moment_integrant(Pk, k, order, Rg):
#     """Integrand used to determine power spectrum amplitude"""
#     return Pk(k) * gaussian_Fourier(k, Rg) * 4.0*np.pi*k**2 * k**(2*order)
    
# def moment_integrantInv(Pk, k, order, Rg):
#     """Inverse of integrand used to determine power spectrum amplitude"""
#     k = 1.0/k
#     return Pk(k) * gaussian_Fourier(k, Rg) * 4.0*np.pi*k**4 * k**(2*order)

def top_hat_3D_Fourier(k, R):
    """Top-hat window function in 3D Fourier space."""
    kw = k*R
    return 9.0*(np.sin(kw)-kw*np.cos(kw))**2/kw**6

# def gaussian_Fourier(k, Rg):
#     return np.exp( -k*k*Rg*Rg )
