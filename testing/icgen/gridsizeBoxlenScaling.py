# check hoe de statistieken schalen met gridsize en boxlen
# deze test is nog uitgevoerd op egpICgen 0.3

import egpICgen.gaussianField as GF
import egpICgen.powerSpectrum as PS

cosmology1 = {
'omegaM': 1,
'omegaL': 0,
'h': 0.7,
'trans': PS.trans1,
'primn': -1,
'rth': 8./0.7,
'sigma0': 1
}

out00 = GF.fieldgen(cosmology1, 100./0.7, 64, seed=2, returnFourier=True)
out01 = GF.fieldgen(cosmology1, 100./0.7, 128, seed=2, returnFourier=True)
out10 = GF.fieldgen(cosmology1, 200./0.7, 64, seed=2, returnFourier=True)
out11 = GF.fieldgen(cosmology1, 200./0.7, 128, seed=2, returnFourier=True)

# out00 en out11 zouden dezelfde statistiek moeten hebben als de normalisatie
# goed is.
# out00 en out01 zouden gelijk moeten schalen als out10 en out11 als de
# berekening onafhankelijk van boxlen is.

np.abs(out00[0]).mean()
np.abs(out01[0]).mean()
np.abs(out10[0]).mean()
np.abs(out11[0]).mean()

# aan beide bovenstaande voorwaarden is voldaan