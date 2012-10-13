from egp import io
ic = io.GadgetData('/Users/users/pbos/dataserver/sims/ICs/ic_icon_100Mpc_256_6_2522572538.dat')
z0 = io.GadgetData('/Users/users/pbos/dataserver/sims/run106/snap_008')

from egp import icgen
boxlen = 100000.0
gridsize = 256
dx = boxlen/gridsize
q = np.mgrid[dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx]

psi = ic.pos.T.reshape(3,256,256,256) - q
psi -= (psi > 0.8*boxlen)*boxlen

redshift_old = 63.
from egp.icgen import Cosmology
cosmo = Cosmology('wmap7')
omegaM = cosmo.omegaM
omegaL = cosmo.omegaL
omegaR = cosmo.omegaR
D_old = icgen.grow(redshift_old, omegaR, omegaM, omegaL)
D_new = icgen.grow(0, omegaR, omegaM, omegaL)

psi *= D_new/D_old

zeld0 = (q + psi)%boxlen

diff = z0.pos.T.reshape(3,256,256,256) - zeld0
diff -= (diff > 0.8*boxlen)*boxlen
diff += (diff < -0.8*boxlen)*boxlen

np.abs(diff).mean()
np.abs(diff).std()
np.abs(diff).max()
pl.hist(np.log10(np.abs(diff.flatten())))

#~ In [69]: np.abs(diff).mean()
#~ Out[69]: 696.81233468397556
#~ 
#~ In [70]: np.abs(diff).std()
#~ Out[70]: 687.13049339896281
#~ 
#~ In [71]: np.abs(diff).max()
#~ Out[71]: 30267.140369052067
