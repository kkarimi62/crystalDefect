# from backports import configparser
# def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
#     #--- parse conf. file
#     confParser = configparser.ConfigParser()
#     confParser.read('configuration.ini')
#     #--- set parameters
# #	confParser.set('parameters','itime0','5')
#     confParser.set('input files','lib_path',os.getcwd()+'/../../HeaDef/postprocess')
#     confParser.set('input files','input_path',argv)
#     #--- write
#     confParser.write(open('configuration.ini','w'))	
#     #--- set environment variables

#     someFile = open( 'oarScript.sh', 'w' )
#     print('#!/bin/bash\n',file=someFile)
#     print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)    
#     print('module load python/anaconda3-2018.12\nsource /global/software/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh\nconda activate gnnEnv2nd\n\n',file=someFile)
# #	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
#     if convert_to_py:
#         print('ipython3 py_script.py\n',file=someFile)
#     else:
#         print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
#     someFile.close()										  
# #
# if __name__ == '__main__':
#     import os
# #
#     runs	 = range(3) #8)
#     jobname  = {
#                 '3':'niNatom1KTemp1000K', 
#                 }['3']
#     DeleteExistingFolder = True
#     readPath = os.getcwd() + {
#                                 '3':'/../simulations/niNatom1KTemp1000K',
#                             }['3'] #--- source
#     EXEC_DIR = '.'     #--- path for executable file
#     durtn = '23:59:59'
#     mem = '256gb'
#     partition = ['parallel','cpu2019','bigmem','single'][2] 
#     argv = "%s"%(readPath) #--- don't change! 
#     PYFILdic = { 
#         1:'test.ipynb',
#         }
#     keyno = 1 #change!!!!!!!!!
#     convert_to_py = True
# #---
# #---
#     PYFIL = PYFILdic[ keyno ] 
#     #--- update argV
#     if convert_to_py:
#         os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
#         PYFIL = 'py_script.py'
#     #---
#     if DeleteExistingFolder:
#         os.system( 'rm -rf %s' % jobname ) # --- rm existing
#     # --- loop for submitting multiple jobs
#     for counter in runs:
#         print(' i = %s' % counter)
#         writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
#         os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#         os.system( 'cp configuration.ini %s' % ( writPath ) ) #--- cp python module
#         makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
#         os.system( 'chmod +x oarScript.sh; cp oarScript.sh configuration.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
#         jobname0 = jobname.split('/')[0] #--- remove slash
#         os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
#                             --chdir %s -c %s -n %s %s/oarScript.sh'\
#                            % ( partition, mem, durtn, jobname0, counter, jobname0, counter, jobname0, counter \
#                                , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!


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
    confParser.set('gnn','num_layers','8')
    confParser.set('gnn','c_hidden','16')
    #--- write
    confParser.write(open('configuration.ini','w'))	
    #--- set environment variables

    someFile             = open( 'oarScript.sh', 'w' )
    print('#!/bin/bash\n',file=someFile)
    print('EXEC_DIR=%s\n\n'%( EXEC_DIR ),file=someFile)
    print('module load python/anaconda3-2019.10-tensorflowgpu',file=someFile)
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
                            '4':'descriptors/ni/pure/test2nd',#'descriptors/ni/pure/test', 
                            '5':'neuralNet/ni/pure/test3rd',#'neuralNet/ni/pure/test', 
                            '6':'mlmc/ni/interestitials/test2nd', 
                            }['5']
    DeleteExistingFolder = True
    readPath             = os.getcwd() + {
                                            '4':'/../simulations/ni/pure/test2nd',
                                            '5':'/descriptors/ni/pure/test2nd',#'/descriptors/ni/pure/test',
                                            '6':'/neuralNet/ni/interestitials/test2nd', 
                                        }['5'] #--- source
    PYFILdic             = { 
                            0:'buildDescriptors.ipynb',
                            1:'neuralNetwork.ipynb',
                            2:'mlmc.ipynb',
                            }
    keyno                = 1
    EXEC_DIR             = '.'     #--- path for executable file
    durtn                = '23:59:59'
    mem                  = '16gb'
    partition            = ['parallel','cpu2019','bigmem','single', 'gpu-v100 --gres=gpu:1'][-1] 
    argv                 = "%s"%(readPath) #--- don't change! 
    convert_to_py        = True
#---
#---
    additional_args      = ''
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
        os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp configuration.ini %s;cp %s/%s %s' % ( writPath, writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
        jobname0         = jobname.replace('/','_')
        os.system( 'sbatch --partition=%s --mem=%s --time=%s %s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                 --chdir %s --ntasks-per-node=%s --nodes=%s --export=slurm_path=%s %s/oarScript.sh >> jobID.txt'\
                            % ( partition, mem, durtn, additional_args, jobname0, counter, jobname0, counter, jobname0, counter \
                                , writPath, nThreads, nNode, writPath, writPath ) ) # --- runs oarScript.sh! 
#         os.system('echo $slurm_path')
    print('job name=%s'%jobname)

