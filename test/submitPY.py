if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 32, 8, 9, 10 ]
    script = 'postproc.py test_ncbj_slurm.py'.split()[1]
    number_hidden_layers  = dict(zip(range(4),[1,2,3,4]))
    hidden_layer_size     = dict(zip(range(4),[10,100,1000,10000]))
#     n_channels  = dict(zip(range(4),[8,16,32,64]))
#     activations = dict(zip(range(20),['linear','sigmoid','relu','softmax','softplus','softsign','tanh','selu','elu','exponential']))
    activations = dict(zip(range(20),['linear']))

    string=open(script).readlines() #--- python script
    #---

    nphi = len(number_hidden_layers)
    #---
    count = 0
    for key_n in number_hidden_layers:
        number_hidden_layer = number_hidden_layers[key_n]
        for key_c in hidden_layer_size:
            n_channel = hidden_layer_size[key_c]
            for key_a in activations:
                activation = activations[key_a]

        #---	
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t\'5\':\'neuralNet/ni/keras/20x20/ann/classifier/layer%s/channel%s/activation%s\',\n" % (key_n,key_c,key_a) #--- change job name
        #---	densities
                inums = lnums[ 1 ] - 1
                string[ inums ] = "    confParser.set(\'neural net\',\'hidden_layer_size\',\'%s\')\n"%(n_channel)
                #
                inums = lnums[ 2 ] - 1
                string[ inums ] = "    confParser.set(\'neural net\',\'number_hidden_layers\',\'%s\')\n"%(number_hidden_layer)
                #
                inums = lnums[ 3 ] - 1
                string[ inums ] = "    confParser.set(\'neural net\',\'activation\',\"\'%s\'\")"%(activation)
                #
                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
