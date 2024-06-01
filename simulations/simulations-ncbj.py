def makeOAR( EXEC_DIR, node, core, time ):
        someFile = open( 'oarScript.sh', 'w' )
        print >> someFile, '#!/bin/bash\n'
        print >> someFile, 'EXEC_DIR=%s\n' %( EXEC_DIR )
        print >> someFile, 'MEAM_library_DIR=%s\n' %( MEAM_library_DIR )
    #	print >> someFile, 'module load mpich/3.2.1-gnu\n'
        print >> someFile, 'source /mnt/opt/spack-0.17/share/spack/setup-env.sh\nspack load openmpi@4.0.5 %gcc@9.3.0\nspack load openblas@0.3.18%gcc@9.3.0\nspack load python@3.8.12%gcc@8.3.0\n\n',
        print >> someFile, 'export LD_LIBRARY_PATH=/mnt/opt/tools/cc7/lapack/3.5.0-x86_64-gcc46/lib:${LD_LIBRARY_PATH}\n'
        #--- run python script 
        for script,var,indx, execc in zip(Pipeline,Variables,range(100),EXEC):
            if execc[:4] == 'lmp_': #_mpi' or EXEC == 'lmp_serial':
                print >> someFile, "time srun $EXEC_DIR/%s < %s -echo screen -var OUT_PATH \'%s\' -var PathEam %s -var INC \'%s\' %s\n"%(execc,script, OUT_PATH, '${MEAM_library_DIR}', SCRPT_DIR, var)
            elif execc == 'py':
                print >> someFile, "python3 %s %s\n"%(script, var)
            elif execc == 'kmc':
                print >> someFile, "export PathEam=${MEAM_library_DIR}\nexport INC=%s\nexport %s\n"%(SCRPT_DIR,var)
                print >> someFile, "source %s \n"%('kmc_bash.sh')
                print >> someFile, "/usr/bin/time -f \'%%e\' srun %s\n"%(kmc_exec)

        someFile.close()										  


if __name__ == '__main__':
        import os
        import numpy as np

        nruns    = 8 #3 #16 #64
        #
        nThreads = 4 #16
        nNode	 = 1
        #
        jobname  = {
                    5:'ni/void_2d_training', 
                    7:'ni/dislocation14th', 
                    8:'ni/irradiation/cascade3rd', 
            
                    4:'ni/interstitial/results/md',      #--- interstitial
                   41:'ni/interstitial/results/kmc', 
                   42:'ni/interstitial/benchmark2nd/kmc', 
            
                    5:'ni/multipleVacs/results/md/vac0', #--- multiple vacancies
                   51:'ni/multipleVacs/results/kmc/vac0', 
            
                    6:'ni/pure/results/md',              #--- single vacancy 
                   61:'ni/pure/results/kmc', 

                    7:'ni/void/results/md',         #--- void 
                   71:'ni/void/results/kmc', 

                    8:'ni/ellipse/results/md', #--- elliptical void
                   81:'ni/ellipse/results/kmc', 

                    9:'ni/ext_dislocation/results/md', #--- elliptical void
                   91:'ni/ext_dislocation/results/kmc3rd', 
                   }[42]
        sourcePath = os.getcwd() +\
                    {	
                        0:'/junk',
                        1:'/../postprocess/NiCoCrNatom1K',
                        4:'/ni/interstitial/results/md', 
                        5:'/ni/multipleVacs/results/md/vac0',
                        6:'/ni/pure/results/md',
                        7:'/ni/void/results/md',
                        8:'/ni/ellipse/results/md',
                        9:'/ni/ext_dislocation/results/md',
                    }[4] #--- must be different than sourcePath. set it to 'junk' if no path
            #
        sourceFiles = { 0:False,
                        1:['Equilibrated_300.dat'],
                        2:['data.txt','ScriptGroup.txt'],
                        3:['data.txt'], 
                        4:['lammps_data.dat'],
                        5:['lammps_data.dat'], #--- only one partition! for multiple ones, use 'submit.py'
                        6:['lammps_data.dat'], 
                        7:['lammps_data.dat'], 
                        8:['lammps_data.dat'], 
                        9:['lammps_data.dat'], 
                     }[4] #--- to be copied from the above directory. set it to '0' if no file
        #
        EXEC_DIR = '/mnt/home/kkarimi/Project/git/lammps-27May2021/src' #--- path for executable file
        kmc_exec = '/mnt/home/kkarimi/Project/git/kart-master/src/KMCART_exec'
        #
        MEAM_library_DIR=  EXEC_DIR+'/../potentials'
        #
        SCRPT_DIR = os.getcwd()+'/lmpScripts' 
        #
        SCRATCH = None
        OUT_PATH = '.'
        if SCRATCH:
            OUT_PATH = '/scratch/${SLURM_JOB_ID}'
        #--- py script must have a key of type str!
        LmpScript = {	                0:'in.PrepTemp0',
                        1:'relax.in', 
                        2:'relaxWalls.in', 
                        7:'in.Thermalization', 
                        71:'in.Thermalization', 
                        4:'in.vsgc', 
                        5:'in.minimization', 
                        51:'in.minimization', 
                        6:'in.shearDispTemp', 
                        8:'in.shearLoadTemp',
                        9:'in.elastic',
                        10:'in.elasticSoftWall',
                         11:'in.pka-simulation',
                         12:'in.interstitial',
                         13:'in.defects',
                         14:'in.vac',
                         15:'in.void',                     
                         16:'in.ellipse',                     
                         17:'in.shockley',                     
                        'p0':'partition.py', #--- python file
                        'p1':'WriteDump.py',
                        'p2':'DislocateEdge2nd.py',
                        'p3':'kartInput.py',
                        'p4':'takeOneOut.py',
                        'p5':'bash-to-csh.py',
                        'p6':'addVoid.py',
                        'p7':'addSubGroups.py',
                         'p8':'preprocess_data.py',
                         'p9':'addAtom.py',
                        1.0:'kmc.sh', #--- bash script
                        2.0:'kmcUniqueCRYST.sh', #--- bash script
                    } 
        #
        def SetVariables():
            Variable = {
                    0:' -var natoms 100000 -var cutoff 3.52 -var ParseData 0 -var ntype 3 -var DumpFile dumpInit.xyz -var WriteData data_init.txt',
                    6:' -var T 300 -var DataFile Equilibrated_300.dat',
                    4:' -var T 600.0 -var t_sw 20.0 -var DataFile Equilibrated_600.dat -var nevery 100 -var ParseData 1 -var WriteData swapped_600.dat', 
                    5:' -var buff 0.0 -var nevery 1000 -var ParseData 0 -var lx 3 -var ly 3 -var lz 3 -var ntype 2 -var cutoff 3.54  -var DumpFile dumpMin.xyz -var WriteData lammps_data.dat -var seed0 %s -var seed1 %s -var seed2 %s -var seed3 %s'%tuple(np.random.randint(1001,9999,size=4)), 
                    51:' -var buff 0.0 -var nevery 100 -var ParseData 1 -var DataFile lammps_data.dat -var DumpFile dumpMin.xyz -var WriteData lammps_data.dat', 
                    7:' -var buff 0.0 -var T 2000.0 -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile lammps_data.dat -var DumpFile dumpThermalized.xyz -var WriteData lammps_data_thermalized.dat -var rnd %s'%np.random.randint(1001,9999),
                    71:' -var buff 0.0 -var T 0.1 -var P 0.0 -var nevery 100 -var ParseData 1 -var DataFile swapped_600.dat -var DumpFile dumpThermalized2.xyz -var WriteData Equilibrated_0.dat',
                    8:' -var buff 0.0 -var T 0.1 -var sigm 1.0 -var sigmdt 0.0001 -var ndump 100 -var ParseData 1 -var DataFile Equilibrated_0.dat -var DumpFile dumpSheared.xyz',
                    9:' -var natoms 1000 -var cutoff 3.52 -var ParseData 1',
                    10:' -var ParseData 1 -var DataFile swapped_600.dat',
                    11:' -var DataFile lammps_data.dat -var epka 5 ',
                    13:' -var buff 0.0 -var nevery 1000 -var ntype 2 -var cutoff 3.54  -var DumpFile dumpMin.xyz -var WriteData lammps_data.dat -var seed0 %s -var seed1 %s -var seed2 %s -var seed3 %s'%tuple(np.random.randint(1001,9999,size=4)), 
                    12:' -var buff 0.0 -var nvac 1 -var T 2000.0 -var P 0.0 -var time 10000.0 -var nevery 10000 -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat -var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s'%tuple(np.random.randint(1001,9999,size=4)),
                    14:' -var buff 0.0 -var nvac 1 -var T 2000.0 -var P 0.0 -var time 10000.0 -var nevery 10000 -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat -var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s'%tuple(np.random.randint(1001,9999,size=4)),
                    15:' -var buff 0.0 -var nvac 1 -var T 2000.0 -var P 0.0 -var time 100000.0 -var nevery 10000 -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat -var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s'%tuple(np.random.randint(1001,9999,size=4)),
                    16:' -var buff 0.0 -var T 2000.0 -var P 0.0 -var time 1000.0 -var nevery 100 -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat -var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s'%tuple(np.random.randint(1001,9999,size=4)),
                    17:' -var buff 0.0 -var T 2000.0 -var P 0.0 -var time 10.0 -var nevery 100 -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat -var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s'%tuple(np.random.randint(1001,9999,size=4)),
                    'p0':' swapped_600.dat 10.0 %s'%(os.getcwd()+'/../postprocess'),
                    'p1':' swapped_600.dat ElasticConst.txt DumpFileModu.xyz %s'%(os.getcwd()+'/../postprocess'),
                    'p2':' %s 3.52 75.0 52.0 8.0 lammps_data.dat 4 2 1.0 0.0'%(os.getcwd()+'/lmpScripts'),
                    'p3':' lammps_data.dat init_xyz.conf %s 2000.0'%(os.getcwd()+'/lmpScripts'),
                    'p4':' lammps_data.dat lammps_data.dat %s 1 1 2.0'%(os.getcwd()+'/lmpScripts'),
#                    'p4':' data_pure.dat dataVoidVac.dat %s 1 1 48.0'%(os.getcwd()+'/lmpScripts'),
                    'p5':' ',
                    'p6':' lammps_data.dat lammps_data.dat %s 4.0 3'%(os.getcwd()+'/lmpScripts'),
                    'p7':' %s lammps_data.dat HCP 2'%(os.getcwd()+'/lmpScripts'),
                    'p8':' Atoms_dyn_Frank_Loop.dat lammps_data.dat %s'%(os.getcwd()+'/lmpScripts'),
                    'p9':' %s lammps_data.dat lammps_data.dat 1'%(os.getcwd()+'/lmpScripts'),
                     1.0:'DataFile=lammps_data.dat',
                     2.0:'DataFile=data_minimized.txt',
                    } 
            return Variable
        #--- different scripts in a pipeline
        indices = {
                    0:[5,'p4',7,'p3',1.0], #--- minimize, add vacancy, thermalize, kart input, invoke kart
                    1:[9],     #--- elastic constants
                    2:[0,'p0',10,'p1'],	   #--- local elastic constants (zero temp)
                    3:[5,7,4,'p0',10,'p1'],	   #--- local elastic constants (annealed)
                    4:['p2',5,7,4,71,8], #--- put disc. by atomsk, minimize, thermalize, anneal, thermalize, and shear
                    8:[5,7,4,51,'p4','p3',1.0], #--- minimize, thermalize, anneal, minimize, add vacancy, kart input, invoke kart
                    10:['p3','p5',1.0], #--- restart from 9: change Restart options in kmc.sh
                    5:[5,7], #--- minimize, thermalize
                    6:[5,'p3',2.0], #--- minimize, kart input, invoke kart
                    7:[5,'p4','p3',1.0], #--- minimize, add vacancy, kart input, invoke kart
                    10:[5,'p4',7,'p3','p5',1.0], #--- minimize, add vacancy, thermalize, kart input, kart.sh to bash shell ,invoke kart
                    12:[ 'p5',1.0], #--- restart 11: set restart = True in kmc.sh
                    13:[5, 'p4', 51, 'p3','p5',1.0], #--- min.,add vacancy,min.,kmc input,kart.sh to bash ,invoke kart
                    14:['p2', 51, 'p4', 51, 'p7', 'p3','p5',1.0], #--- put disc, min, add vacancy, min, add subgroup, kmc input,kart.sh to bash ,invoke kart
                    15:['p8',51,'p3','p5',1.0], #--- irradiation: preprocess, min, kmc input,kart.sh to bash ,invoke kart
                    16:[5,11], #--- irradiation: create & min, cascade
                    17:[51,'p3','p5',1.0], #--- irradiation: min, kmc input,kart.sh to bash ,invoke kart
                    91:[5,'p9',51,'p3','p5',1.0], #--- min., add interestitial, min., kart input, .sh to bash ,invoke kart
                    92:[12,'p3','p5',1.0], #--- min., add interestitial, min., kart input, kart.sh to bash shell ,invoke kart
                    93:[13,'p3','p5',1.0], #--- min., add defects, min., kart input, kart.sh to bash shell ,invoke kart

                 444:[12],           #--- interstitial: md
                  44:['p3','p5',1.0],#--- interstitial: kmc input,.sh_to_bash,invoke kart


                 333:['p2', 51, 'p4', 51, 'p7', 7 ], #--- md: put disc, min, add vacancy, min, add subgroup, thermalize
                  33:['p3','p5',1.0],                #--- dislocation: kmc input,.sh_to_bash,invoke kart

                 222:[16],           #--- elliptical void: md
                  22:['p3','p5',1.0],#--- elliptical void: kmc input,.sh_to_bash,invoke kart

                 111:[15],           #--- void: md
                  11:['p3','p5',1.0],#--- void: kmc input,.sh_to_bash,invoke kart

                 121:[14],           #--- vacancy: md
                  12:['p3','p5',1.0],#--- vacancy: kmc input,.sh_to_bash,invoke kart

                    94:[5,7,'p4',7], #--- minimize, thermalize, add vacancy, thermalize
                    9:[5,'p4',51,'p3','p5',1.0], #--- minimize, add vacancy, minimize, kart input, kart.sh to bash shell ,invoke kart
                  }[44]
        Pipeline = list(map(lambda x:LmpScript[x],indices))
        #
        EXEC_lmp = ['lmp_g++_openmpi'][0]
        durtn = ['23:59:59','47:59:59','167:59:59'][ 1 ]
        mem = '16gb' #'22gb'
        partition = ['INTEL_PHI','INTEL_CASCADE','INTEL_SKYLAKE','INTEL_IVY','INTEL_HASWELL'][1]
        #--
        DeleteExistingFolder = True

        EXEC = list(map(lambda x:np.array([EXEC_lmp,'py','kmc'])[[ type(x) == type(0), type(x) == type(''), type(x) == type(1.0) ]][0], indices))	
        if DeleteExistingFolder:
            os.system( 'rm -rf %s' % jobname ) #--- rm existing
        os.system( 'rm jobID.txt' )
        # --- loop for submitting multiple jobs
        counter = 0
        for irun in xrange( nruns ):
            Variable = SetVariables()
            Variables = list(map(lambda x:Variable[x], indices))
            print ' i = %s' % counter
            writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
            os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
            if irun == 0: #--- cp to directory
                path=os.getcwd() + '/%s' % ( jobname)
                os.system( 'ln -s %s/%s %s' % ( EXEC_DIR, EXEC_lmp, path ) ) # --- create folder & mv oar script & cp executable
            #---
            for script,indx in zip(Pipeline,range(100)):
#                os.system( 'ln -s %s/%s %s' %( SCRPT_DIR, script, writPath) ) #--- lammps script: periodic x, pxx, vy, load
                os.system( 'cp %s/%s %s' %( SCRPT_DIR, script, writPath) ) #--- lammps script: periodic x, pxx, vy, load
            if sourceFiles: 
                for sf in sourceFiles:
                    os.system( 'cp %s/Run%s/%s %s' %(sourcePath, irun, sf, writPath) ) #--- lammps script: periodic x, pxx, vy, load
            #---
            makeOAR( path, 1, nThreads, durtn) # --- make oar script
            os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s' % ( writPath) ) # --- create folder & mv oar scrip & cp executable
            jobname0 = jobname.replace('/','_')
            os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                --chdir %s --ntasks-per-node=%s --nodes=%s %s/oarScript.sh >> jobID.txt'\
                           % ( partition, mem, durtn, jobname0, counter, jobname0, counter, jobname0, counter \
                               , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 
            counter += 1

        print('jobname=',jobname)
        os.system( 'mv jobID.txt %s' % ( os.getcwd() + '/%s' % ( jobname ) ) )
