#/usr/bin/env python
import numpy as np
import egp.icgen
import struct
import tarfile
import os

def setup_cubep3m_run(pos, vel, cosmo, ps, boxlen, gridsize, seed, redshift, snapshots, run_name, run_path_base, ic_filename, nodes_dim, tiles_node_dim, cores, nf_tile_I = 2, nf_cutoff = 16, pid_flag=False, pp_run=True, pp_range = 2, displace_from_mesh=False, read_displacement_seed=False, verbose=False, debug=False, chaplygin=False, dt_scale = 1.0, dt_max = 1.0, ra_max = 0.05, da_max = 0.01, cfactor = 1.05, max_nts = 4000, density_buffer = 2.0):
    """
    Give pos and vel like they come out of egp.icgen.zeldovich.
    
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
    
    -- Main mesh parameters:
    nodes_dim: nodes / dimension, total nodes = nodes_dim**3
    tiles_node_dim: fine mesh tiles / node / dimension
    cores: cores / node
    
    -- Fine mesh parameters:
    nf_tile_I: The size of fine mesh tile in cells / dimension must be set as:
               nf_tile = I*mesh_scale*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
               -- where I is an integer; we set this integer using nf_tile_I.
    nf_cutoff: Fine mesh force cut-off in fine mesh cells (determined by kernel)
    
    -- General code behaviour parameters:
    pp_run
    pp_range
    verbose: diagnostic info, timing, etc.
    debug: extra debugging information
    displace_from_mesh: random displacement at every timestep
    read_displacement_seed: use constant seed for displace_from_mesh
                            N.B.: must write ic_path/seed0.init containing
                            enough seeds for all timesteps!
    pid_flag
    
    -- Time-step parameters:
    dt_scale: Increase/decrease to make the timesteps larger/smaller
    dt_max
    ra_max
    da_max
    cfactor
    max_nts: max number of timesteps
    
    -- Miscellaneous:
    density_buffer: density buffer fraction (1.0 == no buffer, 2.0 == 2x avg
                    density, etc)
    
    
    """
    ### 0. Parameters
    omega_l = cosmo.omegaL
    omega_m = cosmo.omegaM
    omega_b = cosmo.omegaB
    bias = cosmo.bias
    power_index = cosmo.primn
    wde = cosmo.w_0
    w_a = cosmo.w_a
    if chaplygin:
        omega_ch = cosmo.omega_ch
        alpha_ch = cosmo.alpha_ch
        A_ch = cosmo.A_ch
    else:
        omega_ch = 0.
        alpha_ch = 0.
        A_ch = 1.
    
    num_node_compiled = nodes_dim**3
    
    # Fine mesh buffer size in fine mesh cells
    nf_buf         = nf_cutoff + 8
    # size of fine mesh tile in cells / dimension
    nf_tile = nf_tile_I*gridsize*(nodes_dim)**2 / tiles_node_dim + 2*nf_buf
    # number of cells / dimension of entire simulation (fine grid)
    nc             = (nf_tile-2*nf_buf)*tiles_node_dim*nodes_dim
    
    ### 1. Basic preparations
    egp.toolbox.mkdir(run_path)
    egp.toolbox.mkdir(output_path)
    
    run_path     = run_path_base + run_name+'/'
    scratch_path = run_path_base + 'scratch/'
    
    ic_path      = run_path + 'input/'
    ic_filename  = ic_path + 'xv0.ic'
    output_path  = run_path + 'output/'
    cubepm_root  = '../' # relative to batch, where everything is run from

    ## vers source pakket neerzetten
    source_tar_path = egp.icgen.__file__[:egp.icgen.__file__.rfind('/')]+"/cubep3m_clean.tar"
    source_tar = tarfile.TarFile(source_tar_path)
    
    source_tar.extractall(path = run_path)
    
    ### 2. Writing IC-files
    pos = pos.reshape(3, gridsize**3).T
    vel = vel.reshape(3, gridsize**3).T
    # Position is converted to fine-grid cell units.
    pos = pos / boxlen * nc - 0.5 # -0.5 for unclear reasons, but seems to work better...
    # Velocity is also converted to internal units.
    # velunit = 150/a * L/N * sqrt(omega_m) * h
    # where L is the boxsize in Mpc/h and N is the fine grid size.
    # The h is left out in the actual number, because Hubble-units are
    # used everywhere in the code.
    vel = vel / (150*(1+redshift) * boxlen / nc * np.sqrt(omega_m))
    
    ## the actual writing (dm-only):
    f = open(ic_filename,'wb')
    # header
    N = len(pos) # Number of DM particles (Npart[1], i.e. the 2nd entry)
    header = struct.pack("=I", N)
    f.write(header)
    # pos & vel
    data = np.array(np.hstack((pos, vel)), dtype='float32', order='C')
    f.write(data.data)
    f.close()
    
    ### 3. Remaining simulation run preparations
    ## source bestanden met parameters schrijven
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
    if chaplygin:
        FFLAGS_append += " -DChaplygin"
    
    makefile_path = run_path+"source_threads/Makefile"
    parameterfile_path = run_path+"parameters"
    cubepm_par_path = run_path+"source_threads/cubepm.par"
    
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
    if location == 'kapteyn':
        run_script_path = run_path+"batch/kapteyn.run"
    elif location == 'millipede':
        run_script_path = run_path+"batch/millipede.run"
        print "N.B.: MILLIPEDE SCRIPT WERKT NOG NIET!"
    else:
        print "location parameter not recognized! Either kapteyn or millipede."
        raise SystemExit
    
    fill_template_file(run_script_path, locals())
    
    # create symlink to executable in batch directory
    os.symlink(run_path+"source_threads/cubep3m", run_path+"batch/cubep3m")
    
    print("Run with:\n%(run_script_path)s" % locals())


def fill_template_file(filename, value_dict):
    f = open(filename, 'r')
    text = f.read()
    f.close()
    f = open(filename, 'w')
    f.write(text % value_dict)
    f.close()

### 0. Parameters verzamelen
## cosmological simulation parameters
gridsize = 64 # amount of particles = gridsize**3
boxlen = 200.0 # Mpc/h
redshift = 63.0

seed = 2522572538

cosmo = egp.icgen.Cosmology('wmap7', trans=8)
ps = egp.icgen.CosmoPowerSpectrum(cosmo)

#snapshots = np.array([ 3.79778903, 2.98354398, 2.04688779, 1.00131278, 0.50730794, 0.2521318, 0.10412542, 0.])
snapshots = np.array([20., 5., 3., 2., 1.5, 1., 0.5, 0.])

## names, paths etc. parameters
run_name = 'test1.9_+0.5_no_pp'
run_path_base = '/Users/users/pbos/dataserver/cubep3m/'

## grid parameters
nodes_dim = 1
tiles_node_dim = 2
cores = 8
nf_tile_I = 2

## pos en vel bepalen
delta = egp.icgen.GaussianRandomField(ps, boxlen, gridsize, seed)
psi = egp.icgen.DisplacementField(delta)
pos, vel = egp.icgen.zeldovich(redshift, psi, cosmo)

