#!/usr/bin/env python
# encoding: utf-8

from egp.icgen import *
from matplotlib import pyplot as pl
from mayavi import mlab

# ---- BASICS (cosmology, box, etc) ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8

boxlen = 100 # Mpc h^-1
gridsize = 64

dk = 2*np.pi/boxlen*cosmo.h # here you do need to put the actual physical size!
kmax = gridsize*dk
halfgrid = gridsize/2

ps = CosmoPowerSpectrum(cosmo)
ps.normalize((boxlen/cosmo.h)**3)

# ---- FIELDS & CONSTRAINTS ----
# Unconstrained field
rhoU = GaussianRandomField(ps, boxlen, gridsize, seed=None)

# Constrained fields, 2 Jul 2012:
path = "/Users/users/pbos/code/egp/testing/icgen/" # kapteyn
#path = "/Users/patrick/astro/code/egp/testing/icgen/" # macbook

cluster_table_file = open(path+"MCXC+xyz_SCLx4.1.csv")
cluster_table = csvreader(cluster_table_file)

cluster_table.next() # skip header
clusters = []
for cluster in cluster_table:
    clusters.append(cluster)

cluster_table_file.close()

#constraints = constraints_from_csv(path+"constraint_cluster_peak_test.csv", ps, boxlen)

#radec = np.array([np.array(x[0:2]) for x in clusters]) # deg
subpos = np.array([np.array(x[-3:], dtype='float64') for x in clusters]) # Mpc
subpos = subpos*0.7 # Mpc h^-1 (MCXC uses H0 = 70)
#scale = np.array([x[10] for x in clusters], dtype='float64') # kpc/arcsec
#l500 = np.array([x[11] for x in clusters], dtype='float64') # 10^37 W = 10^44 erg s^-1
m500 = np.array([x[12] for x in clusters], dtype='float64') # 10^14 Msun
r500 = np.array([x[13] for x in clusters], dtype='float64') # Mpc
r500 = r500*0.7 # Mpc h^-1 (MCXC uses H0 = 70)

# find position in box of objects:
# (other possibility: define a subbox of objects plus 50 Mpc on all sides;
# largest side => cubic size of subbox; then you need the dx and subsize lines)
#dx = subpos.max(axis=0) - subpos.min(axis=0)
#subsize = dx.max().round()
center = (subpos.max(axis=0) + subpos.min(axis=0))/2
pos = subpos - center + boxlen/2
#boxlen = subsize + boxlen

hubble_constant = cosmo.h*100 * 3.24077649e-20 # s^-1
gravitational_constant = 6.67300e-11 * (3.24077649e-23)**3 / 5.02785431e-31 # Mpc^3 Msun^-1 s^-2
rhoc = 3.*hubble_constant**2/8/np.pi/gravitational_constant # critical density (Msun Mpc^-3)
# Derive a rough estimate for the peak scale, based on uniform density (~ true
# at z = \inf):
scale_mpc = (3*(m500*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
# Note that we did not take the peak height (and thus the true volume) into
# account!!! WE NEED TO LOOK INTO THIS ISSUE.
# OF MOET DAT GEWOON OOK ITERATIEF BEPAALD WORDEN?
# Ludlow & Porciani zeggen er in elk geval niets over (ze bepalen het wel, maar
# gebruiken het nergens)...

scale = ConstraintScale(scale_mpc[0])
location = ConstraintLocation(pos[0])

constraints = []

# first guess for height:
sigma0 = ps.moment(0, scale_mpc[0]/cosmo.h, (boxlen/cosmo.h)**3)
height = 10.*sigma0

constraints.append(HeightConstraint(location, scale, height))

# make it a real peak:
constraints.append(ExtremumConstraint(location, scale, 0))
constraints.append(ExtremumConstraint(location, scale, 1))
constraints.append(ExtremumConstraint(location, scale, 2))

# Do the field stuff!
rhoC1 = ConstrainedField(rhoU, constraints)

# Now, Zel'dovich it:
psiC1 = DisplacementField(rhoC1)
X, Y, Z, vx, vy, vz = zeldovich_new(0., psiC1, cosmo) # Mpc, not h^-1!

# ---- PLOTTING ----
# SyncMaster 2443 dpi:
y = 1200 #pixels
dInch = 24 # inch (diagonal)
ratio = 16./10 # it's not a 16/9 screen
yInch = dInch/np.sqrt(ratio**2+1)
dpi = y/yInch

#fig = pl.figure(figsize=(20/2.54,24/2.54), dpi=dpi)
#ax1 = fig.add_subplot(2,1,1)
#ax2 = fig.add_subplot(2,1,2)
#ax3 = fig.add_subplot(3,2,3)
#ax4 = fig.add_subplot(3,2,4)
#ax5 = fig.add_subplot(3,2,5)

#ax1.imshow(rhoU.t[halfgrid], interpolation='nearest')
#ax2.imshow(rhoC1.t[halfgrid], interpolation='nearest')
#~ ax3.imshow(rhoC2.t[halfgrid], interpolation='nearest')
#~ ax4.imshow(rhoC3.t[halfgrid], interpolation='nearest')
#~ ax5.imshow(rhoC4.t[halfgrid], interpolation='nearest')
#pl.show()

#contour = mlab.contour3d(rhoC1.t, opacity=0.3)
#contour.module_manager.scalar_lut_manager.lut.scale = 'log10'
#contour.contour.number_of_contours = 10
#mlab.draw()

#quiver = mlab.quiver3d(X,Y,Z,vx,vy,vz, opacity=0.3)

points = mlab.points3d(X*cosmo.h,Y*cosmo.h,Z*cosmo.h, mode='point', opacity=0.5)
cluster = mlab.points3d(pos[0,0], pos[0,1], pos[0,2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc[0], opacity=0.3)

#vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(rhoC1.t), vmin=0, vmax=1) # volume rendering

mlab.show()
