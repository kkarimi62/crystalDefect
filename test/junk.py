import numpy as np
if 1:
        MEAM_library_DIR = '/mnt/home/kkarimi/Project/git/lammps-27May2021/src/../potentials'
        INC              = '/mnt/home/kkarimi/Project/git/crystalDefect/simulations/lmpScripts'

        args             = "-screen none -var OUT_PATH . -var PathEam %s -var INC %s -var buff 0.0\
                            -var nevery 1000  -var T 2000.0 -var time 1.0\
                            -var DumpFile dumpMin.xyz -var WriteData lammps_data.dat "%(MEAM_library_DIR,INC)+\
                           "-var rnd %s -var rnd1 %s"%tuple(np.random.randint(1001,9999,size=2))
        print(args)
