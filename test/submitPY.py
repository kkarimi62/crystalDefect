if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 33, 8, 9, 10, 11 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    number_hidden_layers  = dict(zip(range(4),[1,2,3]))
    hidden_layer_size     = dict(zip(range(4),[1]))
    n_channels            = dict(zip(range(4),[8,16,32,64]))
    activations           = dict(zip(range(20),['linear']))
#     activations = dict(zip(range(20),['linear','sigmoid','relu', 'softmax','softplus','softsign','tanh','selu','elu','exponential']))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(number_hidden_layers)
    #---
    count = 0
    for key_n in number_hidden_layers:
        number_hidden_layer = number_hidden_layers[key_n]
        for key_c in n_channels:
            n_channel = n_channels[key_c]
            for key_a in activations:
                activation = activations[key_a]
                for key_h in hidden_layer_size:
                    nsize = hidden_layer_size[key_h]

        #---	
                    inums = lnums[ 0 ] - 1
                    string[ inums ] = "\t\'5\':\'neuralNet/20x20/cnn/classifier/layer%s/channel%s/activation%s/layer_size%s\',\n" % (key_n,key_c,key_a,key_h) #--- change job name
            #---	densities
                    inums = lnums[ 1 ] - 1
                    string[ inums ] = "    confParser.set(\'neural net\',\'n_channels\',\'%s\')\n"%(n_channel)
                    #
                    inums = lnums[ 2 ] - 1
                    string[ inums ] = "    confParser.set(\'neural net\',\'number_hidden_layers\',\'%s\')\n"%(number_hidden_layer)
                    #
                    inums = lnums[ 3 ] - 1
                    string[ inums ] = "    confParser.set(\'neural net\',\'activation\',\"\'%s\'\")\n"%(activation)
                    #
                    inums = lnums[ 4 ] - 1
                    string[ inums ] = "    confParser.set(\'neural net\',\'hidden_layer_size\',\'%s\')\n"%(nsize)
                    #
                    sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                    os.system( 'python3 junk%s.py'%count )
                    os.system( 'rm junk%s.py'%count )
                    count += 1
