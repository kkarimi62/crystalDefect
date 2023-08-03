#!/bin/csh

#--- set natom and box size (output of KartInput.py)
set natom=`cat .natom.txt`
set lx=`cat .lx.txt`
set temp=`cat .temp.txt`

#--- set directories for lmp scripts
setenv	INCLUDE ${INC}
setenv	PathEAM ${PathEam}
setenv	DATAFILE ${DataFile}

###################################### Main Input/Output files ########################################################

setenv INI_FILE_NAME             'init_xyz.conf'   # The file name containing the intial configuration 
 
###################################### Simulation Details ##################################################

setenv NBRE_KMC_STEPS                1       # The max number of KMC steps to be executed
setenv TEMPERATURE                   ${temp} #500.0    # The simulated temperature in kelvin

setenv NUMBER_ATOMS                   ${natom} #864     # The total number of atoms 
setenv SIMULATION_BOX                ${lx} #21.47   # The size of the simulation box (x, y and z)
setenv NSPECIES                       5 #3 #5     # The number of different atom types (default: 2)
setenv ATOMIC_SYMBOLS					'Ni Co Cr Fe Mn' #"Ni Co Cr" #'Ni Co Cr Fe Mn' #"Ni Co Cr"
#setenv NTRAVAILLEUR                     3     # The number of cores associated with forces calculations per ARTnouveau search (default:1)
###################################### Restart options #####################################################

setenv RESTART_KMC                   .false.  # IF true, restart from previous run
setenv RESTART_FILE               "this_conf" # The file name used to continue a simulation from where it was last stopped
setenv RESTART_IMPORT                .false.  # Start a NEW simulation but with the current KMC event catalogue (events.uft and topos.list)
setenv NEW_CATALOGUE                 .false.  #don't change! IF true, will continue simulation but will rebuild event catalogue from scratch


#################################### Basin parameters #################################
setenv OSCILL_TREAT            BMRM       # choose between BMRM, TABU or NON
setenv BASIN_LOCAL		.false.	
setenv MIN_SIG_BARRIER                 0.1    # Max height of barrier and inv. barrier for an event to be considered inside a basin

#################################### Topology Params ##################################

setenv TOPO_RADIUS                    3.0     # radius for topology cluster > CRYST_TOPO_RADIUS 
setenv MAX_TOPO_CUTOFF                3.0     # length-cutoff used by default to link two atoms
setenv MIN_TOPO_CUTOFF                2.2     # minimal length cutoff used when looking at secondary topologies 
#setenv CRYST_TOPOID                   1079097  # topo id of the crystalline-like topologies
setenv CRYST_TOPO_RADIUS              3.0     # radius for crystal-like topologies (default: 4.0 A) 
setenv TOPOLOGY_FILE 				  'Topologies' # Store info about topologies
setenv TOPO_STAT_FILE 				  'topos.list' # Store statistics about topologies
#setenv EVENT_IGNORE_FILE 			  'Event_ignore' # File where events to ignore are stored# (default: Event_ignore)
#setenv EVENT_REMOVE_FILE 			  'Event_remove' # File where events ids to remove from catalogue are stored #(applied at #setenv MAX_NODES_GRAPH				  22 #kam
setenv UNIQUE_CRYST_STRUCT	.true. #kam

#################################### Force calculations ###############################

setenv ENERGY_CALC                LAM      # choose between EDP or SWP or SWA or LAM (Lammps)
setenv FORCE_CALC                 TTL      # choose TTL for total force calculation and PAR for partial force

setenv INPUT_LAMMPS_FILE   'in_kart.lammps' # LAMMPS input file when using ENERGY_TYPE = LAM to calculate the forces
  
setenv UNITS_CONVERSION       'metal'      # Converts the energy and distance units provided by the force code
                                          # into a desired value for ART - affects only the parameters in ART and kART
                                          # units available are real , electron and si
                                          # be very careful when choosing other units, you must change parameters of energy 
                                          # and distance (if it's not in Angstrom) in input files for LAMMPS 
 
setenv UPDATE_VER_NEIB           TTL      # choose TTL for total force calculation and PAR for partial force
setenv NEIB_CALC                 ALL      # choose ALL or VER
setenv UPDATE_TOPO               TTL       # choose TTL or PAR
setenv PAR_DISP_THRESH2       0.00001     # max displacement squared which triggers an update of the neighbor 
                                           # list when using VER lists (default: 0.00001)                                                              
                                            
################ FIRE MINIMIZATION ###############
# setenv DTMAX_FIRE           0.15        # default value
setenv DTMAX_FIRE           0.25        # default value
setenv DTMAX_FIRE_MIN       0.05        # default value=DTMAX_FIRE
# setenv MAX_ITER_FIRE        1500        # default value
# setenv NORM_CRITERIUM       0.0005      # default value
# setenv FMAX_CRITERIUM       0.0005      # default value

################ FIRE PERP. HYPERPLANE MINIMIZATION ############
# setenv MAX_ITER_FIRE_PERP    15          # default value
# setenv FOR_LEAVING_BASIN_USE_FIRE .true. # When relaxing perp for leaving harmonic basin if .true. choose FIRE algorithm else SD

################# ART PARAMETERS ######################################################
setenv SADDLE_PUSH_PARAM          0.1     # The fraction of the initial-saddle distance used to push saddle config. away from initial minimum (default: 0.1)
setenv TYPE_OF_EVENTS             local   # Initial move for events - global or local
setenv RADIUS_INITIAL_DEFORMATION 3.0     # Cutoff for local-move (in angstroems)
setenv EIGENVALUE_THRESHOLD      -1.0     # Eigenvalue threshold for leaving basin

setenv EXIT_FORCE_THRESHOLD       0.1    # Threshold for convergence at saddle point
setenv FORCE_THRESHOLD_PERP_REL   0.05    # Threshold for perpendicular relaxation

setenv FINE_EXIT_FORCE_THRESHOLD       0.05    # finner Threshold for convergence at saddle point 
setenv FINE_FORCE_THRESHOLD_PERP_REL   0.01    # finner Threshold for perpendicular relaxation

#kam setenv MIN_NUMBER_KSTEPS          2       # Min. number of ksteps before calling lanczos
setenv INCREMENT_SIZE             0.1     # Overall scale for the increment moves in activation

setenv INITIAL_STEP_SIZE          1.00    # Size of initial displacement, in A
setenv BASIN_FACTOR               3.00
setenv MIN_NUMBER_KSTEPS          3       # Min. number of ksteps before calling lanczos
setenv MAX_PERP_MOVES_BASIN       6       # Maximum number of perpendicular steps leaving basin
setenv MAX_PERP_MOVES_ACTIV       20       # Maximum number of perpendicular steps during activation
setenv MAX_ITER_BASIN             40      # Maximum number of iteraction for leaving the basin (kter)



setenv MAX_ITER_ACTIVATION        30      # Maximum number of iteraction during activation (iter)

setenv NUMBER_LANCZOS_VECTORS     15      # Number of vectors included in lanczos procedure
setenv LANCZOS_STEP               1e-3    #1e-2    # Size of the step for the numerical derivative (def: 0.001)




# CHECK_LANCZOS_STAB overwrite LANCZOS_STEP and loop for 1e-1 to 1e-7 to check stability and then stop the code. 
# setenv CHECK_LANCZOS_STAB  .true.         # Check lanczos stability over 200 steps, each iteration uses previous lanczos vector
setenv NBRE_POINTS_LANCZOS        2       # number of extra points for numerical derivative in lanczos default 1 supported 1 or 2 for now.


#################################### GENERIC events parameters ########################
setenv SEARCH_FREQUENCY      10           # Minimum number of attempts to find a GENERIC event per new topology encountered
setenv THRES_INCREASE_FREQ   25           # Number of failed attempts encountered because increasing the EIGEN_THRESH
setenv TYPE_EVENT_UPDATE     SPEC         # choose between SPEC or GENE 
setenv USE_LOG_SEARCH       .true.       # Search frequency is multiplied by logarithmic increasing function (default .true.)

setenv CHECK_INI_SAD_CONNECTIVITY .true. # When GENERIC saddle is found, pushes the system towards the initial minimum
                                          #  and minimizes.
                                          # If minimized config. not the same as the initial one, the saddle is rejected.

##################################### Specific event parameters ########################

#setenv MAX_HISTO                 20       # The number of histogram bins (default: 100)
#setenv REFINE_PUSH               0.1       # The ratio of the saddle-final distance pushed before minimizing a SPEC event
#setenv OUTPUT_SPECIFIC           .false.  # IF true, will create a SPEC_EVENT_DIR and output two txt files for each SPEC event found
#setenv SPEC_EVENTS_DIR  'SPEC_EVENTS_DIR' # Name of the SPEC_EVENT_DIR directory
#setenv MAX_SPEC_EV_NBRE          40       # The min number of spec events per atom in memory (if more, array is doubled, tripled, etc.

#setenv REFINE_ATTEMPTS            2       # The number of attempts to refine a generic event (default: 2)
#setenv MIN_REFINE_BARRIER                 # minimal barrier to be refined, all barriers lower are cloned (default: 0.25d0 * Bconst*temperature)
#setenv BARRIER_CHANGE_THRESH              # Maximum acceptable change in barrier (in %) for refined events comp. to original GEN (default: not used)
#setenv BARRIER_CHANGE_MIN                 # Minimum barrier needed to use BARRIER_CHANGE_THRESH in eV (default: 0.1 eV)


############### Printing details ######################################################################

setenv ALLCONF_WITH_SADDLE           .true.
setenv PRINT_DETAILS                 .true.  # Prints the details of activation and minimization 
setenv MINSAD_DETAILS                .true.  # Prints the details of activation and minimization 
setenv USE_TXT_EVENTFILE             .true.
setenv STATISTICS                    .true.   # Write statistics about force and event calculation  
setenv OUTPUT_CONFIG_EVENTS          .true.   # IF true, will create a txt file with the list of all the topologies and events after each KMC step
setenv OUTPUT_SPECIFIC   	     .true.
#setenv OUTPUT_NEB_GEN_EVENT        .true.    # Can be useful

 

############### Run the simulation ######################################################################
unlimit stacksize
ln -s ../../lmpScripts/${INPUT_LAMMPS_FILE} . #--- lmp script 
/home/kamran.karimi1/Project/git/kart/src/KMCART_exec
