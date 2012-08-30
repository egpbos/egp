from egp.icgen import *
from time import time
from iconstrain import constrain_field
import cProfile

def zeldovich_time(redshift, psi, cosmo, print_info=False):
    """
    Use the Zel'dovich approximation to calculate positions and velocities at
    certain /redshift/, based on the DisplacementField /psi/ of e.g. a Gaussian
    random density field and Cosmology /cosmo/.
    
    Outputs a tuple of a position and velocity vector array; positions are in
    units of h^{-1} Mpc (or in fact the same units as /psi.boxlen/) and
    velocities in km/s.
    """
    psi1 = psi.x.t
    psi2 = psi.y.t
    psi3 = psi.z.t
    
    omegaM = cosmo.omegaM
    omegaL = cosmo.omegaL
    omegaR = cosmo.omegaR
    boxlen = psi.boxlen
    
    gridsize = len(psi1)
    if print_info: print "Boxlen:    ",boxlen
    dx = boxlen/gridsize
    if print_info: print "dx:        ",dx
    t0 = time()
    f = fpeebl(redshift, omegaR, omegaM, omegaL)
    t1 = time()
    if print_info: print "fpeebl:    ",f
    D = grow(redshift, omegaR, omegaM, omegaL)
    t2 = time()
    D0 = grow(0, omegaR, omegaM, omegaL) # used for normalization of D to t = 0
    t3 = time()
    if print_info: print "D+(z):     ",D
    if print_info: print "D+(0):     ",D0
    if print_info: print "D(z)/D(0): ",D/D0
    H = hubble(redshift, omegaR, omegaM, omegaL)
    t4 = time()
    if print_info: print "H(z):      ",H
    
    xfact = D/D0
    vfact = D/D0*H*f/(1+redshift)
    
    v = vfact * np.array([psi1,psi2,psi3]) # vx,vy,vz
    t5 = time()
    # lagrangian coordinates, in the center of the gridcells:
    q = np.mgrid[dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx,dx/2:boxlen+dx/2:dx]
    t6 = time()
    X = (q + xfact*(v/vfact))%boxlen
    t7 = time()
    #~ # Mirror coordinates, because somehow it doesn't match the coordinates put
    #~ # into the constrained field.
    #~ X = boxlen - X # x,y,z
    #~ v = -v
    # FIXED MIRRORING: using different FFT convention now (toolbox.rfftn etc).
    
    return np.array([t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6])

def initialize():
    cosmo = Cosmology('wmap7')
    cosmo.trans = 8
    boxlen = 100. # Mpc h^-1
    
    redshift = 63.
    
    gridsize = 64
    #~ seed = np.random.randint(0x100000000)
    seed = 2522572538
    
    pos0 = np.array([20.,40.,70.])
    mass0 = 2.5 # 10^14 M_sun
    
    t0 = time()
    ps = CosmoPowerSpectrum(cosmo)
    ps.normalize(boxlen**3)
    t1 = time()
    
    # Unconstrained field
    delU = GaussianRandomField(ps, boxlen, gridsize, seed=seed)
    t2 = time()
    
    delC = constrain_field(pos0, mass0, boxlen, delU, ps, cosmo)
    t3 = time()
    
    psiC = DisplacementField(delC)
    t4 = time()
    
    print "Normalize ps", t1-t0
    print "GRF", t2-t1
    print "CRF", t3-t2
    print "Psi", t4-t3
    return redshift, cosmo, psiC

redshift, cosmo, psiC = initialize()

def run():
    times = zeldovich_time(redshift, psiC, cosmo)
    normalized_time = times/times.min()
    print "fpeebl(redshift)", "%10.2f" % normalized_time[0], "%10.6f" % times[0]
    print "grow(redshift)  ", "%10.2f" % normalized_time[1], "%10.6f" % times[1]
    print "grow(0)         ", "%10.2f" % normalized_time[2], "%10.6f" % times[2]
    print "hubble(redshift)", "%10.2f" % normalized_time[3], "%10.6f" % times[3]
    print "v               ", "%10.2f" % normalized_time[4], "%10.6f" % times[4]
    print "q               ", "%10.2f" % normalized_time[5], "%10.6f" % times[5]
    print "X               ", "%10.2f" % normalized_time[6], "%10.6f" % times[6]
    print "total           ", "%21.6f" % times.sum()
    return normalized_time
