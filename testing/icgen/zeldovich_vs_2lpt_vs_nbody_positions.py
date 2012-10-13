# Deze gadget run is volgens mij gemaakt met:
# result = np.array([ 19.74546442,  38.41062136,  67.70247069]) # Zel'dovich result
# Dit gebruiken we hieronder dus.

from egp import io, icgen, toolbox
from iconstrain5 import constrain_field

#~ ic = io.GadgetData('/Users/users/pbos/dataserver/sims/ICs/ic_icon_100Mpc_256_6_2522572538.dat')
z0 = io.GadgetData('/Users/users/pbos/dataserver/sims/run106/snap_008')

cosmo = icgen.Cosmology('wmap7')
cosmo.trans = 8
boxlen = 100.0
gridsize = 256
dx = boxlen/gridsize
q = np.mgrid[dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx]

seed = 2522572538

ps = icgen.CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

result = np.array([ 19.74546442,  38.41062136,  67.70247069])

rhoc = toolbox.critical_density(cosmo) # M_sun Mpc^-3 h^2
particle_mass = cosmo.omegaM * rhoc * boxlen**3 / gridsize**3 / 1e10 # 10^10 M_sun h^-1

rhoU_out = icgen.GaussianRandomField(ps, boxlen, gridsize, seed=seed)

# constrained peak mass stuff:
path = "/Users/users/pbos/code/egp/testing/icgen/" # kapteyn
#path = "/Users/patrick/astro/code/egp/testing/icgen/" # macbook
cluster_table_file = open(path+"MCXC+xyz_SCLx4.1.csv")
cluster_table = icgen.csvreader(cluster_table_file)
cluster_table.next() # skip header
clusters = []
for cluster in cluster_table:
    clusters.append(cluster)

cluster_table_file.close()
mass0 = np.array([x[12] for x in clusters], dtype='float64')[0] # 10^14 Msun

irhoC = constrain_field(result, mass0, boxlen, rhoU_out, ps, cosmo)

ipsiC = icgen.DisplacementField(irhoC)
ipsi2C = icgen.DisplacementField2ndOrder(ipsiC)
del irhoC

xz, vz = icgen.zeldovich(0, ipsiC, cosmo)
x2, v2 = icgen.two_LPT_ICs(0, ipsiC, ipsi2C, cosmo) # Mpc, not h^-1!
xN = z0.pos.T.reshape(3,256,256,256)/1000

dz2 = xz - x2
dz2 -= (dz2 > 0.8*boxlen)*boxlen
dz2 += (dz2 < -0.8*boxlen)*boxlen
dNz = xN - xz
dNz -= (dNz > 0.8*boxlen)*boxlen
dNz += (dNz < -0.8*boxlen)*boxlen
dN2 = xN - x2
dN2 -= (dN2 > 0.8*boxlen)*boxlen
dN2 += (dN2 < -0.8*boxlen)*boxlen

np.abs(dz2).mean()
np.abs(dNz).mean()
np.abs(dN2).mean()

np.abs(dz2).max()
np.abs(dNz).max()
np.abs(dN2).max()

np.abs(dz2).min()
np.abs(dNz).min()
np.abs(dN2).min()

np.abs(dz2).std()
np.abs(dNz).std()
np.abs(dN2).std()

#~ diff -= (diff > 0.8*boxlen)*boxlen
#~ diff += (diff < -0.8*boxlen)*boxlen

#~ np.abs(diff).mean()
#~ np.abs(diff).std()
#~ np.abs(diff).max()
#~ pl.hist(np.log10(np.abs(diff.flatten())))

#~ In [69]: np.abs(diff).mean()
#~ Out[69]: 696.81233468397556
#~ 
#~ In [70]: np.abs(diff).std()
#~ Out[70]: 687.13049339896281
#~ 
#~ In [71]: np.abs(diff).max()
#~ Out[71]: 30267.140369052067
