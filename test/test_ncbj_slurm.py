from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv,argvv):
    confParser = configparser.ConfigParser()
    confParser.read('configuration.ini')
    confParser.set('input files','lib_path','%s %s'%(os.getcwd()+'/../../HeaDef/postprocess'),os.getcwd())
    confParser.set('input files','input_path',argv)
    confParser.set('neural net','input_path',argvv)
    confParser.set('neural net','n_channels','1')
    confParser.set('neural net','number_hidden_layers','1')
    confParser.set('neural net','activation',"'linear'")
    #--- write
    confParser.write(open('configuration.ini','w'))	
    #--- set environment variables

    someFile = open( 'oarScript.sh', 'w' )
    print('#!/bin/bash\n',file=someFile)
    print('EXEC_DIR=%s\n source /mnt/opt/spack-0.17/share/spack/setup-env.sh\n\nspack load python@3.8.12%%gcc@8.3.0\n\n'%( EXEC_DIR ),file=someFile)
    if convert_to_py:
        print('time ipython3 py_script.py\n',file=someFile)
    else:
        print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
    someFile.close()										  
#
if __name__ == '__main__':
    import os
#
    runs	 = range(3) #32)
    nNode    = 1
    nThreads = 1
    jobname  = {
                '4':'descriptors/ni/20x20', 
                '5':'neuralNet/ni/keras/20x20/cnn/classification2nd', 
                '6':'neuralNet/ni/mlmc', 
                }['5']
    DeleteExistingFolder = True
    readPath = os.getcwd() + {
                                '4':'/../simulations/niNatom1KTemp1000K3rd',
                                '5':'/descriptors/ni/20x20',
                                '6':'/neuralNet/ni/keras/20x20/cnn/classification', 
                            }['5'] #--- source
    EXEC_DIR = '.'     #--- path for executable file
    durtn = '23:59:59'
    mem = '64gb'
    partition = ['INTEL_PHI','INTEL_CASCADE','INTEL_SKYLAKE','INTEL_IVY','INTEL_HASWELL'][1]
    argv = "%s"%(readPath) #--- don't change! 
    PYFILdic = { 
        0:'buildDescriptors.ipynb',
        1:'neuralNetwork.ipynb',
        2:'mlmc.ipynb',
        }
    keyno = 1
    convert_to_py = True
#---
#---
    PYFIL = PYFILdic[ keyno ]
    if convert_to_py:
        os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
        PYFIL = 'py_script.py'
    #--- update argV
    #---
    if DeleteExistingFolder:
        print('rm -f %s'%jobname)
        os.system( 'rm -rf %s' % jobname ) # --- rm existing
    # --- loop for submitting multiple jobs
    for counter in runs:
        print(' i = %s' % counter)
        writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
        os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#		os.system( 'cp utility.py LammpsPostProcess2nd.py OvitosCna.py %s' % ( writPath ) ) #--- cp python module
        makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter, argv) # --- make oar script
        os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp configuration.ini %s;cp %s/%s %s' % ( writPath, writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
        os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                 --chdir %s --ntasks-per-node=%s --nodes=%s %s/oarScript.sh >> jobID.txt'\
                            % ( partition, mem, durtn, jobname.split('/')[0], counter, jobname.split('/')[0], counter, jobname.split('/')[0], counter \
                                , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 

