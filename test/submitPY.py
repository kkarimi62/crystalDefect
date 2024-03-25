if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 37, 12,13 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    number_hidden_layers  = dict(zip(range(4),[1,2,4,8]))
    hidden_layer_size     = dict(zip(range(4),[2,4,8,16,32,64,128]))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(number_hidden_layers)
    #---
    count = 0
    for key_n in number_hidden_layers:
        number_hidden_layer = number_hidden_layers[key_n]
        for key_h in hidden_layer_size:
            nsize = hidden_layer_size[key_h]

#---	
            inums = lnums[ 0 ] - 1
            string[ inums ] = "\t\'5\':\'neuralNet/ni/pure/test/layer%s/layer_size%s\',\n" % (key_n,key_h) #--- change job name
    #---	densities
            #
            inums = lnums[ 1 ] - 1
            string[ inums ] = "    confParser.set(\'gnn\',\'num_layers\',\'%s\')\n"%(number_hidden_layer)
            #
            #
            inums = lnums[ 2 ] - 1
            string[ inums ] = "    confParser.set(\'gnn\',\'c_hidden\',\'%s\')\n"%(nsize)
            #
            sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
            os.system( 'python junk%s.py'%count )
            os.system( 'rm junk%s.py'%count )
            count += 1
