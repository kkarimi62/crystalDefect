if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 37 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    number_hidden_layers  = dict(zip(range(3),[1,2,4]))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(number_hidden_layers)
    #---
    count = 0
    for key_n in number_hidden_layers:
            number_hidden_layer = number_hidden_layers[key_n]

#---	
            inums = lnums[ 0 ] - 1
            string[ inums ] = "    path_for_simulation=\'ni/multipleVacs/results/kmc/vac%s\',\n" % (key_n) #--- change job name
    #---	densities
            #
#            inums = lnums[ 1 ] - 1
#            string[ inums ] = "    confParser.set(\'gnn\',\'num_layers\',\'%s\')\n"%(number_hidden_layer)
            #
            #
#            inums = lnums[ 2 ] - 1
#            string[ inums ] = "    confParser.set(\'gnn\',\'c_hidden\',\'%s\')\n"%(nsize)
            #
            sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
            os.system( 'python junk%s.py'%count )
            os.system( 'rm junk%s.py'%count )
            count += 1
