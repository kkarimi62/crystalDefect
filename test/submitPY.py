if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 32, 9, 10 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    number_hidden_layers  = dict(zip(range(4),[1,2,3]))
    n_channels  = dict(zip(range(4),[2,4,8,16]))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(number_hidden_layers)
    #---
    count = 0
    for key_n in number_hidden_layers:
        number_hidden_layer = number_hidden_layers[key_n]
        for key_c in n_channels:
            n_channel = n_channels[key_c]

        #---	
            inums = lnums[ 0 ] - 1
            string[ inums ] = "\t\'5\':\'neuralNet/ni/keras/20x20/cnn/layer%s/channel%s\',\n" % (key_n,key_c) #--- change job name
    #---	densities
            inums = lnums[ 1 ] - 1
            string[ inums ] = "    confParser.set(\'neural net\',\'n_channels\',\'%s\')\n"%(n_channel)
            #
            inums = lnums[ 2 ] - 1
            string[ inums ] = "    confParser.set(\'neural net\',\'number_hidden_layers\',\'%s\')\n"%(number_hidden_layer)
            #
            sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
            os.system( 'python3 junk%s.py'%count )
            os.system( 'rm junk%s.py'%count )
            count += 1
