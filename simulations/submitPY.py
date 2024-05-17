if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
#    lnums = [ 39, 137   ]
    lnums = [ 40, 58   ]
    #
    string=open('simulations-ncbj.py').readlines() #--- python script
    #---
    PHI = dict(zip(range(3),[1,2,4]))
    nphi = len(PHI)
    #---
    #---
    #--- 
    count = 0
    keyss= list(PHI.keys())
    keyss.sort()
    for iphi in keyss:
                temp = PHI[iphi]
            #---	
            #---	densities
                inums = lnums[ 0 ] - 1
#                string[ inums ] = "\t5:\'ni/multipleVacs/results/md/vac%s\',\n"%(iphi) #--- change job name
                string[ inums ] = "\t51:\'ni/multipleVacs/results/kmc/vac%s\',\n"%(iphi) #--- change job name
            #---
                inums = lnums[ 1 ] - 1
#                string[ inums ] = "\t14:\' -var buff 0.0 -var nvac %s -var T 2000 -var time 10000.0 -var rnd %%s -var rnd1 %%s -var rnd2 %%s -var rnd3 %%s -var P 0.0 -var nevery 10000  -var DumpFile dumpThermalized.xyz -var WriteData lammps_data.dat\'%%tuple(np.random.randint(1001,9999,size=4)),\n"%(temp)
                string[ inums ] = "\t5:\'/ni/multipleVacs/results/md/vac%s\',\n"%(iphi) #--- change job name
                #---

                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python2 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
