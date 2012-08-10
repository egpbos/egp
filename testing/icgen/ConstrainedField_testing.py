#!/usr/bin/env python
# encoding: utf-8

from egp.icgen import *
from matplotlib import pyplot as pl

# --------------------------- TESTING CODE ---------------------------

# ---- BASICS (cosmology, box, etc) ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8

boxlen = 100 # (10 Aug 2012)
gridsize = 64

dk = 2*np.pi/boxlen
kmax = gridsize*dk
halfgrid = gridsize/2

ps = CosmoPowerSpectrum(cosmo)
ps.normalize(boxlen**3)

# ---- FIELDS & CONSTRAINTS ----
# Unconstrained field
rhoU = GaussianRandomField(ps, boxlen, gridsize, seed=None)

# Constraint Locations/Scales
#~ location1 = ConstraintLocation(np.array((50,50,50))/cosmo.h) # Mpc
#~ location2 = ConstraintLocation(np.array((50,75,75))/cosmo.h) # Mpc
#~ scale1 = ConstraintScale(8/cosmo.h) # Mpc

# Spectral parameters (for easier constraint value definition, at least in case
# of gaussian random field):
#~ sigma0 = ps.moment(0, scale1.scale, boxlen**3)
#~ sigma1 = ps.moment(1, scale1.scale, boxlen**3)
#~ sigma2 = ps.moment(2, scale1.scale, boxlen**3)
#~ 
#~ sigma_min1 = ps.moment(-1, scale1.scale, boxlen**3)
#~ gamma_nu = sigma0**2 / sigma_min1 / sigma1
#~ sigma_g = 3./2 * cosmo.omegaM * (cosmo.h*100)**2 * sigma_min1
#~ sigma_g_peak = sigma_g * np.sqrt(1 - gamma_nu**2)
#~ 
#~ gamma = sigma1**2/sigma0/sigma2
#~ sigma_E = 3./2*cosmo.omegaM*(cosmo.h*100)**2 * sigma0 * np.sqrt((1-gamma**2)/15)

# Constraints ...
# ... height (positive for peak, negative for void) ...
#~ heightVal = 5*sigma0
#~ height = HeightConstraint(location1, scale1, heightVal)
#~ height2 = HeightConstraint(location2, scale1, heightVal)

# ... extrema ...
#~ extre1 = ExtremumConstraint(location1, scale1, 0)
#~ extre2 = ExtremumConstraint(location1, scale1, 1)
#~ extre3 = ExtremumConstraint(location1, scale1, 2)

# ... gravity ...
#~ g1 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 0, cosmo)
#~ g2 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 1, cosmo)
#~ g3 = GravityConstraint(location1, scale1, 3*sigma_g_peak, 2, cosmo)

# ... shape ...
#~ x_d = 5. # steepness
#~ a12 = 8. # first axial ratio
#~ a13 = 1. # second axial ratio
#~ alpha = 0. # Euler angle (w.r.t box axes)
#~ beta  = 0. # Euler angle
#~ psi   = 0. # Euler angle

# BEGIN factory voor shape:
#~ A = euler_matrix(alpha, beta, psi)
#~ lambda1 = x_d * sigma2 / (1 + a12**2 + a13**2)
#~ lambda2 = lambda1 * a12**2
#~ lambda3 = lambda1 * a13**2
#~ 
#~ d11 = - lambda1*A[0,0]*A[0,0] - lambda2*A[1,0]*A[1,0] - lambda3*A[2,0]*A[2,0]
#~ d22 = - lambda1*A[0,1]*A[0,1] - lambda2*A[1,1]*A[1,1] - lambda3*A[2,1]*A[2,1]
#~ d33 = - lambda1*A[0,2]*A[0,2] - lambda2*A[1,2]*A[1,2] - lambda3*A[2,2]*A[2,2]
#~ d12 = - lambda1*A[0,0]*A[0,1] - lambda2*A[1,0]*A[1,1] - lambda3*A[2,0]*A[2,1]
#~ d13 = - lambda1*A[0,0]*A[0,2] - lambda2*A[1,0]*A[1,2] - lambda3*A[2,0]*A[2,2]
#~ d23 = - lambda1*A[0,1]*A[0,2] - lambda2*A[1,1]*A[1,2] - lambda3*A[2,1]*A[2,2]
#~ 
#~ shape1 = []
#~ shape1.append(ShapeConstraint(location1, scale1, d11, 0, 0))
#~ shape1.append(ShapeConstraint(location1, scale1, d22, 1, 1))
#~ shape1.append(ShapeConstraint(location1, scale1, d33, 2, 2))
#~ shape1.append(ShapeConstraint(location1, scale1, d12, 0, 1))
#~ shape1.append(ShapeConstraint(location1, scale1, d13, 0, 2))
#~ shape1.append(ShapeConstraint(location1, scale1, d23, 1, 2))
# END shape factory

# ... and tidal field.
#~ epsilon = 10. # strength of the tidal field
#~ pomega  = 0.25*np.pi # parameter for relative strength along axes (0, 2*pi)
#~ alphaE = 0.5*np.pi # Euler angle (w.r.t box axes)
#~ betaE  = 0.25*np.pi # Euler angle
#~ psiE   = 0. # Euler angle

# BEGIN factory voor tidal field:
#~ T = euler_matrix(alphaE, betaE, psiE)
#~ L1 = np.cos((pomega + 2*np.pi)/3)
#~ L2 = np.cos((pomega - 2*np.pi)/3)
#~ L3 = np.cos(pomega/3)
#~ fac = epsilon*sigma_E
#~ 
#~ E11 = fac*(L1*T[0,0]*T[0,0] + L2*T[1,0]*T[1,0] + L3*T[2,0]*T[2,0])
# Note that in these values the epsilon*f_G term (whatever f_G might have been)
# is neglected due to the field being linear (see vdW&B96).
#~ E22 = fac*(L1*T[0,1]*T[0,1] + L2*T[1,1]*T[1,1] + L3*T[2,1]*T[2,1])
#~ E12 = fac*(L1*T[0,0]*T[0,1] + L2*T[1,0]*T[1,1] + L3*T[2,0]*T[2,1])
#~ E13 = fac*(L1*T[0,0]*T[0,2] + L2*T[1,0]*T[1,2] + L3*T[2,0]*T[2,2])
#~ E23 = fac*(L1*T[0,1]*T[0,2] + L2*T[1,1]*T[1,2] + L3*T[2,1]*T[2,2])
#~ 
#~ tidal1 = []
#~ tidal1.append(TidalConstraint(location1, scale1, E11, 0, 0, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E22, 1, 1, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E12, 0, 1, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E13, 0, 2, cosmo))
#~ tidal1.append(TidalConstraint(location1, scale1, E23, 1, 2, cosmo))
# END tidal field factory


# Constrained fields, 21-23 February 2012:
#~ rhoC1 = ConstrainedField(rhoU, [height,])
#~ rhoC2 = ConstrainedField(rhoU, [height, extre1, extre2, extre3])
#~ rhoC3 = ConstrainedField(rhoU, [height, height2])


# Constrained fields, 9 March 2012:
#rhoC4 = ConstrainedField(rhoU, [height, g1, g2, g3]) # gravity/velocity werkt
#rhoC4 = ConstrainedField(rhoU, [height] + shape1) # shape werkt
#~ rhoC4 = ConstrainedField(rhoU, [height] + tidal1)

# Constrained fields, 22 May 2012:
constraints1 = constraints_from_csv("/Users/users/pbos/code/egp/testing/icgen/constraints1.csv", ps, boxlen)
rhoC1 = ConstrainedField(rhoU, constraints1)

# ---- PLOTTING ----
# SyncMaster 2443 dpi:
y = 1200 #pixels
dInch = 24 # inch (diagonal)
ratio = 16./10 # it's not a 16/9 screen
yInch = dInch/np.sqrt(ratio**2+1)
dpi = y/yInch

fig = pl.figure(figsize=(20/2.54,24/2.54), dpi=dpi)
ax1 = fig.add_subplot(3,2,1)
ax2 = fig.add_subplot(3,2,2)
ax3 = fig.add_subplot(3,2,3)
ax4 = fig.add_subplot(3,2,4)
ax5 = fig.add_subplot(3,2,5)

ax1.imshow(rhoU.t[halfgrid], interpolation='nearest')
ax2.imshow(rhoC1.t[halfgrid], interpolation='nearest')
#~ ax3.imshow(rhoC2.t[halfgrid], interpolation='nearest')
#~ ax4.imshow(rhoC3.t[halfgrid], interpolation='nearest')
#~ ax5.imshow(rhoC4.t[halfgrid], interpolation='nearest')
pl.show()


# ------------ PROBABILITY FUNCTIONS ----------- -
# curvature
x = np.linspace(0,10,5001)

cpdf = cpdf_curvature(x, height.value, gamma)
#cumcpdf = np.cumsum(cpdf)/cpdf.sum()
cumcpdfBBKS = cumulative_cpdf_curvature(x,height.value, gamma)
invPDF = UnivariateSpline(cumcpdfBBKS, x, k=3, s=0)
curvature = invPDF(np.random.random())

pl.plot(x,cpdf,'-',x,cumcpdfBBKS,'-')#,x,cumcpdf,'-')
pl.show()

# shape
p,e = np.mgrid[-0.25:0.25:0.005, 0:0.5:0.005]
xmax = x[cpdf.argmax()]
cpdf_ep = cpdf_shape(e, p, xmax)
pl.imshow(cpdf_ep, interpolation='nearest')
pl.show()
