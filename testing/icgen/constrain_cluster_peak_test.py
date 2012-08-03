#!/usr/bin/env python
# encoding: utf-8

from egp.icgen import *
from matplotlib import pyplot as pl
from mayavi import mlab
import egp.toolbox

# ---- BASICS (cosmology, box, etc) ----
cosmo = Cosmology('wmap7')
cosmo.trans = 8

boxlen = 100. # Mpc h^-1
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
#~ location = ConstraintLocation(pos[0])
pos0 = np.array([20.,40.,70.])
location = ConstraintLocation(pos0)

constraints = []

# first guess for height:
sigma0 = ps.moment(0, scale_mpc[0]/cosmo.h, (boxlen/cosmo.h)**3)
height = 3.*sigma0

constraints.append(HeightConstraint(location, scale, height))

# make it a real peak:
constraints.append(ExtremumConstraint(location, scale, 0))
constraints.append(ExtremumConstraint(location, scale, 1))
constraints.append(ExtremumConstraint(location, scale, 2))

# Do the field stuff!
rhoC1 = ConstrainedField(rhoU, constraints)

# Now, Zel'dovich it:
psiC1 = DisplacementField(rhoC1)
POS, v = zeldovich_new(0., psiC1, cosmo) # Mpc, not h^-1!


# <begin find mean position of particles>

# Find the mean position of the particles that were originally in the peak (or
# at least in a sphere with radius of the peak scale), or MEDIAN position:
xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2

# determine roll needed to get peak position back to where it should be:
floor_cell = np.int32(location.location/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
roll = floor_cell - gridsize/2
# difference of roll (= integer) with real position (in cells):
diff = location.location/boxlen*gridsize - floor_cell
xgrid -= diff[0]/gridsize*boxlen
ygrid -= diff[1]/gridsize*boxlen
zgrid -= diff[2]/gridsize*boxlen

# (to be rolled) distance function (squared!):
r2grid = xgrid**2 + ygrid**2 + zgrid**2
# roll it:
r2grid = np.roll(r2grid, -roll[0], axis=0) # roll negatively, because element[0,0,0]
r2grid = np.roll(r2grid, -roll[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
r2grid = np.roll(r2grid, -roll[2], axis=2) # (due to changing around in zeldovich)

spheregrid = r2grid < scale_mpc[0]**2

# finally calculate the "new position" of the peak:
#~ POS = np.array([X,Y,Z])
mean_peak_pos = POS[:,spheregrid].mean(axis=1)*cosmo.h
median_peak_pos = np.median(POS[:,spheregrid],axis=1)*cosmo.h

# <end find mean position of particles>

# AND THEN....

# ITERATE!!!!!!!!!!

# Yeah.

# <iteration 2>
pos_prev = pos0
#pos_i = pos_prev + (pos_prev - mean_peak_pos)/2 # 2 is arbitrary; just a little more than one
pos_i = pos_prev + (pos_prev - mean_peak_pos)*2 # 1/2 was too little. This needs to be adjusted automatically -> use solving algorithm

location_i = ConstraintLocation(pos_i)

scale_mpc_i = (3*(m500*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
# Note that we did not take the peak height (and thus the true volume) into
# account!!! WE NEED TO LOOK INTO THIS ISSUE.
# OF MOET DAT GEWOON OOK ITERATIEF BEPAALD WORDEN?
# Ludlow & Porciani zeggen er in elk geval niets over (ze bepalen het wel, maar
# gebruiken het nergens)...

scale_i = ConstraintScale(scale_mpc_i[0])

constraints_i = []

# first guess for height:
sigma0_i = ps.moment(0, scale_mpc_i[0]/cosmo.h, (boxlen/cosmo.h)**3)
height_i = 3.*sigma0_i

constraints_i.append(HeightConstraint(location_i, scale_i, height_i))

# make it a real peak:
constraints_i.append(ExtremumConstraint(location_i, scale_i, 0))
constraints_i.append(ExtremumConstraint(location_i, scale_i, 1))
constraints_i.append(ExtremumConstraint(location_i, scale_i, 2))

# Do the field stuff!
rhoC1_i = ConstrainedField(rhoU, constraints_i) # N.B.: rhoU stays the same!!!

# Now, Zel'dovich it:
psiC1_i = DisplacementField(rhoC1_i)
POS_i, v_i = zeldovich_new(0., psiC1_i, cosmo) # Mpc, not h^-1!

# <begin find mean position of particles>

# Find the mean position of the particles that were originally in the peak (or
# at least in a sphere with radius of the peak scale), or MEDIAN position:
xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2

# determine roll needed to get peak position back to where it should be:
floor_cell_i = np.int32(location_i.location/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
roll_i = floor_cell_i - gridsize/2
# difference of roll (= integer) with real position (in cells):
diff_i = location_i.location/boxlen*gridsize - floor_cell_i
xgrid -= diff_i[0]/gridsize*boxlen
ygrid -= diff_i[1]/gridsize*boxlen
zgrid -= diff_i[2]/gridsize*boxlen

# (to be rolled) distance function (squared!):
r2grid = xgrid**2 + ygrid**2 + zgrid**2
# roll it:
r2grid = np.roll(r2grid, -roll[0], axis=0) # roll negatively, because element[0,0,0]
r2grid = np.roll(r2grid, -roll[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
r2grid = np.roll(r2grid, -roll[2], axis=2) # (due to changing around in zeldovich)

spheregrid_i = r2grid < scale_mpc_i[0]**2

# finally calculate the "new position" of the peak:
#~ POS_i = np.array([X_i,Y_i,Z_i])
mean_peak_pos_i = POS_i[:,spheregrid_i].mean(axis=1)*cosmo.h
median_peak_pos_i = np.median(POS_i[:,spheregrid_i],axis=1)*cosmo.h

# <end find mean position of particles>


# <end iteration 2>


# <begin iteration i>
def plot_positions(pos_i):
    location_i = ConstraintLocation(pos_i)
    
    scale_mpc_i = (3*(m500*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    # Note that we did not take the peak height (and thus the true volume) into
    # account!!! WE NEED TO LOOK INTO THIS ISSUE.
    # OF MOET DAT GEWOON OOK ITERATIEF BEPAALD WORDEN?
    # Ludlow & Porciani zeggen er in elk geval niets over (ze bepalen het wel, maar
    # gebruiken het nergens)...
    
    scale_i = ConstraintScale(scale_mpc_i[0])
    
    constraints_i = []
    
    # first guess for height:
    sigma0_i = ps.moment(0, scale_mpc_i[0]/cosmo.h, (boxlen/cosmo.h)**3)
    height_i = 3.*sigma0_i
    
    constraints_i.append(HeightConstraint(location_i, scale_i, height_i))
    
    # make it a real peak:
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 0))
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 1))
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 2))
    
    # Do the field stuff!
    rhoC1_i = ConstrainedField(rhoU, constraints_i) # N.B.: rhoU stays the same!!!
    
    # Now, Zel'dovich it:
    psiC1_i = DisplacementField(rhoC1_i)
    POS_i, v_i = zeldovich_new(0., psiC1_i, cosmo) # Mpc, not h^-1!
    
    # <begin find mean position of particles>
    
    # Find the mean position of the particles that were originally in the peak (or
    # at least in a sphere with radius of the peak scale), or MEDIAN position:
    xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2
    
    # determine roll needed to get peak position back to where it should be:
    floor_cell_i = np.int32(location_i.location/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
    roll_i = floor_cell_i - gridsize/2
    # difference of roll (= integer) with real position (in cells):
    diff_i = location_i.location/boxlen*gridsize - floor_cell_i
    xgrid -= diff_i[0]/gridsize*boxlen
    ygrid -= diff_i[1]/gridsize*boxlen
    zgrid -= diff_i[2]/gridsize*boxlen
    
    # (to be rolled) distance function (squared!):
    r2grid = xgrid**2 + ygrid**2 + zgrid**2
    # roll it:
    r2grid = np.roll(r2grid, -roll_i[0], axis=0) # roll negatively, because element[0,0,0]
    r2grid = np.roll(r2grid, -roll_i[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
    r2grid = np.roll(r2grid, -roll_i[2], axis=2) # (due to changing around in zeldovich)
    
    spheregrid_i = r2grid < scale_mpc_i[0]**2
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos_i = POS_i[:,spheregrid_i].mean(axis=1)*cosmo.h
    points = mlab.points3d(POS_i[0]*cosmo.h,POS_i[1]*cosmo.h,POS_i[2]*cosmo.h, mode='point', opacity=0.5)
    cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc[0], opacity=0.3)
    peak_points = mlab.points3d(POS_i[0][spheregrid]*cosmo.h, POS_i[1][spheregrid]*cosmo.h, POS_i[2][spheregrid]*cosmo.h, opacity=0.5, mode='point', color=(0,1,0))
    mlab.show()


def iterate_mean(pos_i):
    location_i = ConstraintLocation(pos_i)
    
    scale_mpc_i = (3*(m500*1e14)/4/np.pi/rhoc)**(1./3) # Mpc h^-1
    # Note that we did not take the peak height (and thus the true volume) into
    # account!!! WE NEED TO LOOK INTO THIS ISSUE.
    # OF MOET DAT GEWOON OOK ITERATIEF BEPAALD WORDEN?
    # Ludlow & Porciani zeggen er in elk geval niets over (ze bepalen het wel, maar
    # gebruiken het nergens)...
    
    scale_i = ConstraintScale(scale_mpc_i[0])
    
    constraints_i = []
    
    # first guess for height:
    sigma0_i = ps.moment(0, scale_mpc_i[0]/cosmo.h, (boxlen/cosmo.h)**3)
    height_i = 3.*sigma0_i
    
    constraints_i.append(HeightConstraint(location_i, scale_i, height_i))
    
    # make it a real peak:
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 0))
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 1))
    constraints_i.append(ExtremumConstraint(location_i, scale_i, 2))
    
    # Do the field stuff!
    rhoC1_i = ConstrainedField(rhoU, constraints_i) # N.B.: rhoU stays the same!!!
    
    # Now, Zel'dovich it:
    psiC1_i = DisplacementField(rhoC1_i)
    POS_i, v_i = zeldovich_new(0., psiC1_i, cosmo) # Mpc, not h^-1!
    
    # <begin find mean position of particles>
    
    # Find the mean position of the particles that were originally in the peak (or
    # at least in a sphere with radius of the peak scale), or MEDIAN position:
    xgrid, ygrid, zgrid = np.mgrid[0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize, 0:boxlen:boxlen/gridsize] + boxlen/gridsize/2 - boxlen/2
    
    # determine roll needed to get peak position back to where it should be:
    floor_cell_i = np.int32(location_i.location/boxlen*gridsize) # "closest" cell (not really of course in half of the cases...)
    roll_i = floor_cell_i - gridsize/2
    # difference of roll (= integer) with real position (in cells):
    diff_i = location_i.location/boxlen*gridsize - floor_cell_i
    xgrid -= diff_i[0]/gridsize*boxlen
    ygrid -= diff_i[1]/gridsize*boxlen
    zgrid -= diff_i[2]/gridsize*boxlen
    
    # (to be rolled) distance function (squared!):
    r2grid = xgrid**2 + ygrid**2 + zgrid**2
    # roll it:
    r2grid = np.roll(r2grid, -roll_i[0], axis=0) # roll negatively, because element[0,0,0]
    r2grid = np.roll(r2grid, -roll_i[1], axis=1) # is not x=0,0,0 but x=boxlen,boxlen,boxlen
    r2grid = np.roll(r2grid, -roll_i[2], axis=2) # (due to changing around in zeldovich)
    
    spheregrid_i = r2grid < scale_mpc_i[0]**2
    
    # finally calculate the "new position" of the peak:
    #~ POS_i = np.array([X_i,Y_i,Z_i])
    mean_peak_pos_i = POS_i[:,spheregrid_i].mean(axis=1)*cosmo.h
    return mean_peak_pos_i

def difference(pos_iter, boxlen):
    print pos_iter, "i.e.", pos_iter%boxlen, "in the box"
    pos_new = iterate_mean(pos_iter%boxlen)
    print "geeft:", pos_new
    return np.sum((pos_new - pos0)**2)

from scipy.optimize import fmin_l_bfgs_b as solve
from scipy.optimize import anneal

bound_range = 0.1*boxlen
boundaries = ((pos0[0]-bound_range, pos0[0]+bound_range), (pos0[1]-bound_range, pos0[1]+bound_range), (pos0[2]-bound_range, pos0[2]+bound_range))
lower = np.array(boundaries)[:,0]
upper = np.array(boundaries)[:,1]
result = solve(difference, pos0, args=(boxlen,), bounds = boundaries, approx_grad=True)#, epsilon=0.5)
#~ result = anneal(difference, pos0, args=(boxlen,), lower=lower, upper=upper)

plot_positions(result[0])

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

#~ points = mlab.points3d(X*cosmo.h,Y*cosmo.h,Z*cosmo.h, mode='point', opacity=0.5)
#~ cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc[0], opacity=0.3)

#vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(rhoC1.t), vmin=0, vmax=1) # volume rendering

#~ mlab.show()

# Density doen
#~ pos = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
#~ mass = rhoc * boxlen**3 / gridsize**3

#~ rho = egp.toolbox.TSC_density(pos*cosmo.h, gridsize, boxlen, mass)

#import egp.io
#egp.io.write_gadget_ic_dm('data2/pos.gadget', pos*cosmo.h, np.array([vx,vy,vz]).T, mass, 0., boxlen, cosmo.omegaM, cosmo.omegaL, cosmo.h)
# run cd ~/data2; DTFE pos.gadget pos_g -g 96 -p
#dtfe_g = np.memmap('data2/pos_g.a_den', dtype='float32')[3:].reshape((96,96,96))

#~ pl.figure(1)
#~ pl.imshow(rho[24], interpolation='nearest')
#~ pl.figure(2)
#~ pl.imshow(dtfe_g[24], interpolation='nearest')
#~ pl.show()

#~ dx = boxlen/gridsize
#~ rhoX = np.mgrid[0:boxlen:dx,0:boxlen:dx,0:boxlen:dx]
#~ rhoX = np.mgrid[dx:boxlen+dx:dx,dx:boxlen+dx:dx,dx:boxlen+dx:dx]

#~ contour = mlab.contour3d(rhoX[0], rhoX[1], rhoX[2], rho, opacity=0.3)#, extent=(0,boxlen,0,boxlen,0,boxlen))
#~ contour.module_manager.scalar_lut_manager.lut.scale = 'log10'
#~ contour.contour.number_of_contours = 10
#~ mlab.draw()
#~ cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc[0], opacity=0.3)

#~ sample = np.random.randint(0,len(X.flatten()),20000)
#~ points = mlab.points3d(X.flatten()[sample]*cosmo.h,Y.flatten()[sample]*cosmo.h,Z.flatten()[sample]*cosmo.h, opacity=0.4, scale_factor=1)#, mode='point')

#~ contour = mlab.contour3d(rhoX[0][12:24], rhoX[1][12:24], rhoX[2][12:24], rho[12:24], opacity=0.3)#, extent=(0,boxlen,0,boxlen,0,boxlen))
#~ contour.module_manager.scalar_lut_manager.lut.scale = 'log10'
#~ contour.contour.number_of_contours = 10
#~ mlab.draw()
#~ 
#~ rho_field = mlab.pipeline.scalar_field(rhoX[0], rhoX[1], rhoX[2], np.log10(rho))
#~ vol = mlab.pipeline.volume(rho_field) # volume rendering
#~ 
#~ rho_field = mlab.pipeline.scalar_field(rhoX[0], rhoX[1], rhoX[2], rho)
#~ mlab.pipeline.image_plane_widget(rho_field,
                            #~ plane_orientation='x_axes',
                            #~ slice_index=10,
                        #~ )
#~ mlab.pipeline.image_plane_widget(rho_field,
                            #~ plane_orientation='y_axes',
                            #~ slice_index=10,
                        #~ )
#~ mlab.outline()
#~ 
#~ points = mlab.points3d(X[12:24]*cosmo.h,Y[12:24]*cosmo.h,Z[12:24]*cosmo.h, opacity=0.5, mode='point')
#~ cluster = mlab.points3d(pos0[0], pos0[1], pos0[2], mode='sphere', color=(1,0,0), scale_factor=scale_mpc[0], opacity=0.3)

#~ peak_points = mlab.points3d(X[spheregrid]*cosmo.h, Y[spheregrid]*cosmo.h, Z[spheregrid]*cosmo.h, opacity=0.5, mode='point', color=(0,1,0))
#~ peak_quiver = mlab.quiver3d(X[spheregrid]*cosmo.h,Y[spheregrid]*cosmo.h,Z[spheregrid]*cosmo.h,vx[spheregrid],vy[spheregrid],vz[spheregrid], opacity=0.3)
#~ mlab.show()
