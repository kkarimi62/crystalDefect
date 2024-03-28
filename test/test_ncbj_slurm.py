#from backports 
import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv,argvv):
    confParser           = configparser.ConfigParser()
    confParser.read('configuration.ini')
    confParser.set('input files','lib_path', '%s %s'%(os.getcwd()+'/../../HeaDef/postprocess',os.getcwd()))
    confParser.set('input files','input_path',argv)
    confParser.set('ml mc','input_path',argv)
    confParser.set('neural net','input_path',argvv)
    confParser.set('gnn','input_path',argvv)
    #
#    confParser.set('gnn','num_layers','8')
#    confParser.set('gnn','c_hidden','16')
    #--- write
    confParser.write(open('configuration_file.ini','w'))	
    #--- set environment variables

    someFile             = open( 'oarScript.sh', 'w' )
    print('#!/bin/bash\n',file=someFile)
    print('EXEC_DIR=%s\n\n'%( EXEC_DIR ),file=someFile)
#    print('source /mnt/opt/spack-0.17/share/spack/setup-env.sh\n\nspack load python@3.8.12%%gcc@8.3.0\n\n',file=someFile)
    print('source activate pytorch_gpu4th',file=someFile)
    if convert_to_py:
        print('time ipython3 py_script.py\n',file=someFile)
    else:
        print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
    someFile.close()										  

if __name__ == '__main__':
    import os
#
    runs	             = range( 1 )
    nNode                = 1
    nThreads             = 1
    jobname              = {
                            '4':'descriptors/ni/pure/shape14x14x14_increasedCutoff',#'descriptors/ni/pure/test', 
                            '5':'neuralNet/ni/pure/shape14x14x14_increasedCutoff',#shape10x10x10_overfitting_trainSize',#'neuralNet/ni/pure/test', 
                            '6':'mlmc/ni/interestitials/test2nd', 
                            }['4']
    DeleteExistingFolder = True
    readPath             = os.getcwd() + {
                                            '4':'/../simulations/ni/pure/increasedCutoff', #test2nd',
                                            '5':'/descriptors/ni/pure/shape14x14x14_increasedCutoff', #'/descriptors/ni/pure/test',
                                            '6':'/neuralNet/ni/interestitials/test2nd', 
                                        }['4'] #--- source
    PYFILdic             = { 
                            0:'buildDescriptors.ipynb',
                            1:'neuralNetwork.ipynb',
                            2:'mlmc.ipynb',
                            }
    keyno                = 0
    EXEC_DIR             = '.'     #--- path for executable file
    durtn                = '23:59:59'
    mem                  = '8gb'
    partition            = ['INTEL_PHI','INTEL_CASCADE','INTEL_SKYLAKE','INTEL_IVY','INTEL_HASWELL','GPU_K80'][ 1 ]
    argv                 = "%s"%(readPath) #--- don't change! 
    convert_to_py        = True
#---
#---
    additional_args      = ''
    if partition         == 'GPU_K80':
        additional_args  = '--gres=gpu:tesla:2'
    PYFIL                = PYFILdic[ keyno ]
    if convert_to_py:
        os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
        PYFIL            = 'py_script.py'
    #--- update argV
    #---
    if DeleteExistingFolder:
        print('rm %s'%jobname)
        os.system( 'rm -rf %s' % jobname ) # --- rm existing
    # --- loop for submitting multiple jobs
    for counter in runs:
        print(' i = %s' % counter)
        writPath         = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
        os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
        for keys in PYFILdic:
            os.system( 'ln -s %s/%s %s/%s' % ( os.getcwd(),PYFILdic[keys],writPath,PYFILdic[keys] ) ) #--- cp python modules
        makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter, argv) # --- make oar script
        os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; mv configuration_file.ini %s/configuration.ini;cp %s/%s %s' % ( writPath, writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
        jobname0         = jobname.replace('/','_')
        os.system( 'sbatch --partition=%s --mem=%s --time=%s %s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                 --chdir %s --ntasks-per-node=%s --nodes=%s --export=slurm_path=%s %s/oarScript.sh >> jobID.txt'\
                            % ( partition, mem, durtn, additional_args, jobname0, counter, jobname0, counter, jobname0, counter \
                                , writPath, nThreads, nNode, writPath, writPath ) ) # --- runs oarScript.sh! 
#         os.system('echo $slurm_path')
    print('job name=%s'%jobname)

