import os

def prepare_gadget_run(boxlen, gridsize, cosmo, ic_file, redshift_begin, run_dir_base, run_name, nproc, output_list_filename = 'outputs_main.txt', DE_file = 'wdHdGHz_LCDM_bosW7.txt', ic_format = 2, time_max = 1.0, softening_factor = 22.5*768/300000., time_limit_cpu = 864000, resubmit_on = 0, resubmit_command = '0', cpu_time_bet_restart_file = 3600, part_alloc_factor = 1.4, tree_alloc_factor = 0.8, buffer_size = 100, gadget_executable = "/net/schmidt/data/users/pbos/sw/code/gadget/gadget3Sub_512_SL6/P-Gadget3_512")
    """Arguments:
    boxlen (kpc h^-1)
    cosmo (Cosmology object)
    ic_file (path)
    redshift_begin
    run_dir_base (directory path)
    run_name (sub directory name)
    nproc (number of processors)
    output_list_filename (filename w.r.t. run_dir_base)
    DE_file (filename w.r.t. run_dir_base)
    ic_format (1 or 2)
    time_max (expansion factor a)
    softening_factor (fraction of mean interparticle distance)
    time_limit_cpu (seconds)
    resubmit_on (0 or 1)
    resubmit_command (path)
    cpu_time_bet_restart_file (seconds)
    part_alloc_factor
    tree_alloc_factor
    buffer_size (MB)
    gadget_executable (file path)
    
    Note that run_dir_base is not the directory where the simulation will be
    run; that is run_dir_base+run_name; the run_name directory will be created
    by this function.
    """
    output_dir = run_dir_base+'/'+run_name
    os.mkdir(output_dir)
    
    parameter_filename = run_dir_base+'/'+run_name+'.par'
    run_script_filename = run_dir_base+'/'+run_name+'.sh'
    
    output_list_filename = run_dir_base+'/'+output_list_filename
    DE_file = run_dir_base+'/'+DE_file

    time_begin = 1/(redshift_begin+1)

    # Softening: based on Dolag's ratios
    # default ~ 1/17.3 of the mean ipd
    softening = softening_factor*boxlen/gridsize
    softening_max_phys = softening/3
    
    # the actual parameter file:
    par_file_text = "\
\%\%\%\%\% In-/output\n\
InitCondFile  		%(ic_file)s\n\
OutputDir           %(output_dir)s\n\
OutputListFilename  %(output_list_filename)s\n\
\n\
\n\
\%\%\%\%\% Characteristics of run & Cosmology\n\
TimeBegin             %(time_begin)f\n\
TimeMax	              %(time_max)f\n\
BoxSize               %(boxlen)f\n\
\n\
Omega0	              %(cosmo.omegaM)f\n\
OmegaLambda           %(cosmo.omegaL)f\n\
OmegaBaryon           %(cosmo.omegaB)f\n\
HubbleParam           %(cosmo.h)f   ; only needed for cooling\n\
\n\
\n\
\%\%\%\%\% DE (GadgetXXL)\n\
DarkEnergyFile          %(DE_file)s\n\
\%DarkEnergyParam        -0.4\n\
VelIniScale		        1.0\n\
\n\
\n\
\%\%\%\%\% Softening lengths\n\
MinGasHsmlFractional     0.5  \% minimum csfc smoothing in terms of the gravitational softening length\n\
\n\
\% ~ 1/20 of mean ipd (Dolag: 22.5 for ipd of 300000/768)\n\
SofteningGas       %(softening)f\n\
SofteningHalo      %(softening)f\n\
SofteningDisk      0.0\n\
SofteningBulge     0.0        \n\
SofteningStars     %(softening)f\n\
SofteningBndry     0\n\
\n\
\% ~ 1/3 of the above values\n\
SofteningGasMaxPhys       %(softening_max_phys)f\n\
SofteningHaloMaxPhys      %(softening_max_phys)f\n\
SofteningDiskMaxPhys      0.0  \% corr.to EE81_soft = 70.0\n\
SofteningBulgeMaxPhys     0.0         \n\
SofteningStarsMaxPhys     %(softening_max_phys)f\n\
SofteningBndryMaxPhys     0\n\
\n\
\n\
\%\%\%\%\% Time/restart stuff\n\
TimeLimitCPU             %(time_limit_cpu)i\n\
ResubmitOn               %(resubmit_on)i\n\
ResubmitCommand          %(resubmit_command)s\n\
CpuTimeBetRestartFile    %(cpu_time_bet_restart_file)s\n\
\n\
\n\
\%\%\%\%\% Memory\n\
PartAllocFactor       %(part_alloc_factor)f\n\
TreeAllocFactor       %(tree_alloc_factor)f\n\
BufferSize            %(buffer_size)i\n\
\n\
\n\
ICFormat                   %(ic_format)i\n\
\n\
\n\
\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\n\
\%\%\%\%\%\%\%\%\%\%\%\%\% Usually don't edit below here \%\%\%\%\%\%\%\%\%\%\%\%\%\n\
\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\%\n\
\n\
\n\
\n\
\%\% In-/output parameters\n\
OutputListOn               1\n\
SnapFormat                 2\n\
NumFilesPerSnapshot        1\n\
NumFilesWrittenInParallel  1\n\
\n\
\n\
\%\% Default filenames\n\
SnapshotFileBase        snap\n\
EnergyFile        energy.txt\n\
InfoFile          info.txt\n\
TimingsFile       timings.txt\n\
CpuFile           cpu.txt\n\
RestartFile       restart\n\
\n\
\n\
\%\% Misc. options\n\
ComovingIntegrationOn 1\n\
CoolingOn 0\n\
PeriodicBoundariesOn   1\n\
\n\
\n\
\%\% Output frequency (when not using output list)\n\
TimeBetSnapshot        1.04912649189365\n\
TimeOfFirstSnapshot    0.090909091\n\
TimeBetStatistics      0.02\n\
 \n\
\n\
\%\% Accuracy of time integration\n\
TypeOfTimestepCriterion 0		\% Not used option in Gadget2 (left over from G1)\n\
ErrTolIntAccuracy       0.05	\% Accuracy of timestep criterion\n\
MaxSizeTimestep        0.1		\% Maximum allowed timestep for cosmological simulations\n\
                                \% as a fraction of the current Hubble time (i.e. dln(a))\n\
MinSizeTimestep        0		\% Whatever\n\
MaxRMSDisplacementFac  0.25		\% Something\n\
\n\
\n\
\%\% Tree algorithm and force accuracy\n\
ErrTolTheta            0.45\n\
TypeOfOpeningCriterion 1\n\
ErrTolForceAcc         0.005\n\
TreeDomainUpdateFrequency    0.025\n\
\% DomainUpdateFrequency   0.2\n\
\n\
\n\
\%\%  Parameters of SPH\n\
DesNumNgb           64\n\
MaxNumNgbDeviation  1\n\
ArtBulkViscConst    0.75\n\
InitGasTemp         166.53\n\
MinGasTemp          100.    \n\
CourantFac          0.2\n\
\n\
\n\
\%\% System of units\n\
UnitLength_in_cm         3.085678e21        ;  1.0 kpc /h\n\
UnitMass_in_g            1.989e43           ;  solar masses\n\
UnitVelocity_in_cm_per_s 1e5                ;  1 km/sec\n\
GravityConstantInternal  0\n\
\n\
\n\
\%\% Quantities for star formation and feedback\n\
StarformationOn 0\n\
CritPhysDensity     0.     \%  critical physical density for star formation in\n\
                            \%  hydrogen number density in cm^(-3)\n\
MaxSfrTimescale     1.5     \% in internal time unpar_file_textits\n\
CritOverDensity      57.7    \%  overdensity threshold value\n\
TempSupernova        1.0e8   \%  in Kelvin\n\
TempClouds           1000.0   \%  in Kelvin\n\
FactorSN             0.1\n\
FactorEVP            1000.0\n\
\n\
WindEfficiency    2.0\n\
WindFreeTravelLength 10.0\n\
WindEnergyFraction  1.0\n\
WindFreeTravelDensFac 0.5\n\
\n\
\n\
\%\% Additional things for Gadget XXL:\n\
\%ViscositySourceScaling 0.7\n\
\%ViscosityDecayLength   2.0\n\
\%ConductionEfficiency     0.33\n\
Shock_LengthScale 2.0\n\
Shock_DeltaDecayTimeMax  0.02\n\
ErrTolThetaSubfind  	 0.1\n\
DesLinkNgb 	 	32\n"
    
    with open(parameter_filename, 'w') as par_file:
        par_file.write(par_file_text)
    
    run_script_text = "\
#!/bin/bash\n\
# Gadget simulation %(run_name)s.\n\
\n\
cd %(run_dir_base)s\n\
mpiexec -np %(nproc)i %(gadget_executable) %(par_file)s\n"
    
    with open(run_script_filename, 'w') as run_script:
        run_script.write(run_script_text)