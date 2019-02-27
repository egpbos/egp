import numpy as np
import egp.powerSpectrum as PS

logk = np.linspace(-3,3,601)
k = 10**logk

dk = 1.

cosmology = {
'omegaM': 0.268,
'omegaB': 0.044,
'omegaL': 0.732,
'h': 0.704,
'trans': PS.trans7,
'primn': 0.947,
'rth': 8./0.7,
'sigma0': 0.776
}

power = PS.powerspectrum(k,dk,cosmology)
logpower = np.log10(power)

pl.plot(logk, logpower, '-')
pl.show()

# Zie verder testEHpowerspectrum.py