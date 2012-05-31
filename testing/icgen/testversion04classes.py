from egp.icgen import GaussianRandomField as GRF, Cosmology, CosmoPowerSpectrum as CosmoPS, DisplacementField

cosmo = Cosmology('wmap3')
cosmo.trans = 8

ps = CosmoPS(cosmo)

rho = GRF(ps, 100/0.7, 64, seed=102)

psi = DisplacementField(rho)

pl.figure(0)
pl.imshow(rho.t[30], interpolation='nearest')
pl.figure(1)
pl.imshow(psi.x.t[30], interpolation='nearest')
pl.figure(2)
pl.imshow(psi.y.t[30], interpolation='nearest')
pl.figure(3)
pl.imshow(psi.z.t[30], interpolation='nearest')
pl.show()

# werkt allemaal prima lijkt het
