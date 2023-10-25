if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 32, 10 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    PHI  = dict(zip(range(6),[1e-5,1e-4,1e-3,1e-2]))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(PHI)
    #---
    count = 0
    for key in PHI:
                alpha = PHI[key]
            #---	
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t\'5\':\'neuralNet/ni/regularization/alpha%s\',\n" % (key) #--- change job name
        #---	densities
                inums = lnums[ 1 ] - 1
#                 string[ inums ] = "\t\'3\':\'/../simulations/NiCoCrNatom1KTemp%sK\',\n"%(temp)
                string[ inums ] = "\tconfParser.set(\'neural net\',\'alpha\','[%s]')\n"%(alpha)
        #
                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
