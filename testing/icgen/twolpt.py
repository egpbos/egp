from egp import icgen

cosmo = icgen.Cosmology('wmap7')
cosmo.trans = 8

power = icgen.CosmoPowerSpectrum(cosmo)

boxlen = 100.
gridsize = 128

seed = 20
grf = icgen.GaussianRandomField(power, boxlen, gridsize, seed)

psi = icgen.DisplacementField(grf)

psi2 = icgen.DisplacementField2ndOrder(psi)

xz, vz = icgen.zeldovich(0, psi, cosmo)
x2, v2 = icgen.two_LPT_ICs(0, psi, psi2, cosmo)

from mayavi import mlab

mlab.figure(1)
mlab.points3d(xz[0], xz[1], xz[2], mode='point')

mlab.figure(2)
mlab.points3d(x2[0], x2[1], x2[2], mode='point')
