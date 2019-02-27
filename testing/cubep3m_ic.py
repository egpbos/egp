#/usr/bin/env python
import numpy as np
import egp.icgen
import struct
import tarfile
import os

### 0. Parameters verzamelen
## cosmological simulation parameters
gridsize = 64 # amount of particles = gridsize**3
boxlen = 200.0 # Mpc/h
redshift = 63.0

seed = 2522572538

cosmo = egp.icgen.Cosmology('wmap7', trans=8)
ps = egp.icgen.CosmoPowerSpectrum(cosmo)

omega_l = cosmo.omegaL
omega_m = cosmo.omegaM
omega_b = cosmo.omegaB
omega_ch = 0. # Chaplygin gas density
bias = cosmo.bias
power_index = cosmo.primn
# N.B.: power_index is only used in dist_init, so we only actually need
#       it for the param file. Same seems to go for the bias parameter.

wde = -1.0
w_a = 0.0
alpha_ch = 0. # Chaplygin gas parameter
A_ch = 1. # Chaplygin gas parameter

#snapshots = np.array([ 3.79778903, 2.98354398, 2.04688779, 1.00131278, 0.50730794, 0.2521318, 0.10412542, 0.])
snapshots = np.array([20., 5., 3., 2., 1.5, 1., 0.5, 0.])

## names, paths etc. parameters
run_name = 'test1.9_+0.5_no_pp'

run_path     = '/Users/users/pbos/dataserver/cubep3m/'+run_name+'/'
ic_path      = run_path + 'input/'
ic_filename  = ic_path + 'xv0.ic'
scratch_path = '/Users/users/pbos/dataserver/cubep3m/scratch/'
output_path  = run_path + 'output/'
cubepm_root  = '../' # relative to batch, where everything is run from

## (fine) grid parameters
# nodes / dimension, total nodes = nodes_dim**3
nodes_dim = 1
num_node_compiled = nodes_dim**3
# fine mesh tiles / node / dimension
tiles_node_dim = 2
# cores / node
cores = 8

# size of fine mesh tile in cells / dimension must be set as:
#  nf_tile = I*mesh_scale*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
#   -- where I is an integer; we set this integer here:
nf_tile_I      = 2
# Fine mesh force cut-off in fine mesh cells (determined by kernel)
nf_cutoff      = 16
# Fine mesh buffer size in fine mesh cells
nf_buf         = nf_cutoff + 8
# size of fine mesh tile in cells / dimension
nf_tile = nf_tile_I*gridsize*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
# number of cells / dimension of entire simulation (fine grid)
nc             = (nf_tile-2*nf_buf)*tiles_node_dim*nodes_dim

## code parameters
# e.g. with or without PP, timestep-multiplier, etc.

pp_run = False # False in tests 1.6 en 1.8
pp_range = 2
verbose = False # diagnostic info, timing, etc.
debug = False # extra debugging information
displace_from_mesh = False # random displacement at every timestep
read_displacement_seed = False # use constant seed for displace_from_mesh
# N.B.: must write ic_path/seed0.init containing enough seeds for all
#       timesteps!
pid_flag = True

# timestep parameters:

dt_scale = 1.0 # Increase/decrease to make the timesteps larger/smaller
dt_max = 1.0
ra_max = 0.05
da_max = 0.01
cfactor = 1.05

max_nts = 4000 # max number of timesteps

# density buffer fraction (1.0 == no buffer, 2.0 == 2x avg density, etc)
density_buffer = 2.0
    

### 1. Basis van de voorbereidingen voor de simulatie run
egp.toolbox.mkdir(run_path)
egp.toolbox.mkdir(output_path)

## vers source pakket neerzetten
source_tar_path = egp.icgen.__file__[:egp.icgen.__file__.rfind('/')]+"/cubep3m_clean.tar"
source_tar = tarfile.TarFile(source_tar_path)

source_tar.extractall(path = run_path)


### 2. IC-bestanden schrijven
## pos en vel bepalen
delta = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed)
psi = egp.icgen.DisplacementField(delta)
pos, vel = egp.icgen.zeldovich(redshift, psi, cosmo)

# volgorde veranderen (test 1.3):                                                  
#j = np.mgrid[0:gridsize, 0:gridsize, 0:gridsize].reshape(3,gridsize,gridsize,gridsize) 
#pos = pos.T[j[2], j[1], j[0]].T
#vel = vel.T[j[2], j[1], j[0]].T

pos = pos.reshape(3, gridsize**3).T
vel = vel.reshape(3, gridsize**3).T

## wegschrijven (dm-only)
# Position is converted to fine-grid cell units.
pos = pos / boxlen * nc #- 0.5 # -0.5 om onduidelijke redenen...
# VOOR TEST 1.9 HEBBEN WE DE -0.5 WEER WEGGEHAALD
# Velocity is also converted to internal units.
# velunit = 150/a * L/N * sqrt(omega_m) * h
# where L is the boxsize in Mpc/h and N is the fine grid size.
# The h is left out in the actual number, because Hubble-units are
# used everywhere in the code.
vel = vel / (150*(1+redshift) * boxlen / nc * np.sqrt(omega_m))

f = open(ic_filename,'wb')

# header
N = len(pos) # Number of DM particles (Npart[1], i.e. the 2nd entry)
header = struct.pack("=I", N)
f.write(header)

# pos & vel
data = np.array(np.hstack((pos, vel)), dtype='float32', order='C')
f.write(data.data)

f.close()

#~ fread = np.memmap(ic_filename, dtype='float32', mode='r')
#~ freadi = np.memmap(ic_filename, dtype='int32', mode='r')


### 3. Rest van de voorbereidingen voor de simulatie run
## source bestanden met parameters schrijven
"""
The FFLAGS parameter in the Makefile can be appended with defines:
BINARY: output formaat, lijkt het
NGP: iets met een kernel, wrs voor density estimation
NGPH: ook die kernel, maar dan voor de halofinder
LRCKCORR: geen idee, maar lijkt me nuttig, als het een CORRection is.
PPINT: ik denk de buitenste PP loop (zie paper, figuur 2, if pp=.true.)
PP_EXT: binnenste PP loop dan dus.
MPI_TIME: print timings (min/max/avg) van MPI jobs.
DIAG: print allerlei dingen, volgens mij puur voor diagnostics?
PID_FLAG: hou particle IDs bij om ICs aan snapshots te kunnen relateren.
READ_SEED: "This will always use the same random number at each time step. It surely introduces a bias, but is good for testing code."
DISP_MESH: applies random offset from initial position at each timestep om anisotropie t.g.v. het grid te minimaliseren.
DEBUG: zal wel nog meer debug info zijn. Er zijn ook andere, meer specifieke DEBUG parameters; DEBUG_LOW, DEBUG_VEL, DEBUG_CCIC, DEBUG_RHOC, DEBUG_CRHO en DEBUG_PP_EXT bijvoorbeeld.
Chaplygin: include a Chaplygin gas cosmic component.
"""
FFLAGS_append = "-DBINARY -DNGP -DLRCKCORR -DNGPH"
if pp_run:
    FFLAGS_append += " -DPPINT -DPP_EXT"
if displace_from_mesh:
    FFLAGS_append += " -DDISP_MESH"
if read_displacement_seed:
    FFLAGS_append += " -DREAD_SEED"
if verbose:
    FFLAGS_append += " -DDIAG -DMPI_TIME"
if debug:
    FFLAGS_append += " -DDEBUG -DDEBUG_LOW -DDEBUG_VEL -DDEBUG_CCIC -DDEBUG_RHOC -DDEBUG_CRHO -DDEBUG_PP_EXT"
if pid_flag:
    FFLAGS_append += " -DPID_FLAG"

makefile_path = run_path+"source_threads/Makefile"
parameterfile_path = run_path+"parameters"
cubepm_par_path = run_path+"source_threads/cubepm.par"

def fill_template_file(filename, value_dict):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    f = open(filename, 'w')
    f.write(text % value_dict)
    f.close()

fill_template_file(makefile_path, locals())
fill_template_file(parameterfile_path, locals())
fill_template_file(cubepm_par_path, locals())

## input (checkpoints)
checkpoints = open(run_path+"input/checkpoints", 'w')
projections = open(run_path+"input/projections", 'w')
halofinds   = open(run_path+"input/halofinds", 'w')
for snap in snapshots:
    checkpoints.write(str(snap)+"\n")
    projections.write(str(snap)+"\n")
    halofinds.write(str(snap)+"\n")
checkpoints.close(); projections.close(); halofinds.close()

## run-scripts schrijven
# kapteyn:
run_script_path = run_path+"batch/kapteyn.run"
# millipede:
#~ run_script_path = run_path+"batch/millipede.run"
# N.B.: MILLIPEDE SCRIPT WERKT NOG NIET!

fill_template_file(run_script_path, locals())

# create symlink to executable in batch directory
os.symlink(run_path+"source_threads/cubep3m", run_path+"batch/cubep3m")

print("Run with:\n%(run_path)sbatch/kapteyn.run" % locals())
