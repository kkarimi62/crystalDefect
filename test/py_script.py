#!/usr/bin/env python
# coding: utf-8

# In[21]:


import import_ipynb
import configparser
import numpy as np
import sys
import time
import pandas as pd
import os
import pdb
#--- tensorflow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #--- rm warnings
import tensorflow as tf
from tensorflow import keras
#tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(1)
#
from sklearn.preprocessing import StandardScaler

#pdb.set_trace()
#--- user modules
confParser = configparser.ConfigParser() #--- parse conf. file
confParser.read('configuration.ini')
list(map(lambda x:sys.path.append(x), confParser['input files']['lib_path'].split()))
import LammpsPostProcess as lp
import utility as utl
import buildDescriptors as bd
import imp
imp.reload(utl)
imp.reload(lp)
imp.reload(bd)


# # MachineLeranedMC

# In[10]:


class MachineLeranedMC( bd.ParseConfiguration,
                        bd.EnergyBarrier,
                      ):
    '''
    Performs Machine Learned Monte Carlo Swaps
    '''
    
    def __init__(self,
                 confParser, 
                 verbose = False
                ):
        
#         bd.ParseConfiguration.__init__(self, confParser, verbose = verbose )
        self.verbose     =  verbose 
        self.confParser  =  confParser
        
        self.save_output = 'saved_output'
        get_ipython().system('rm -r $self.save_output; mkdir $self.save_output')
        
    def Parse(self,fp):
        t0 = time.time()
        self.lmpData = lp.ReadDumpFile( '%s'%(fp) ) 
        self.lmpData.GetCords( ncount = sys.maxsize)
        if self.verbose:
            print('elapsed time=%s s'%(time.time()-t0))
            print('time steps:',self.lmpData.coord_atoms_broken.keys())
            display(self.lmpData.coord_atoms_broken[0].head())

    
    def Initialize( self ):
        
        self.lmpData0 = self.lmpData.coord_atoms_broken[0].copy()
        
        natom         = len( self.lmpData.coord_atoms_broken[0] )
        ndime         = 3
        self.disp     = np.zeros( natom * ndime ).reshape((natom,ndime))
        self.tdisp    = np.zeros( natom * ndime ).reshape((natom,ndime))
 
        self.mc_time  = 0.0
                
        self.box      = lp.Box(BoxBounds=self.lmpData.BoxBounds[0],AddMissing=np.array([0,0,0]))

    def GetDescriptors( self ):
        '''
        Compute structural descriptors 
        '''
        
        bd.EnergyBarrier.__init__( self,
                                  None,#'%s/EVENTS_DIR'%self.confParser['input files']['input_path'],
                                  None,#'%s/EVLIST_DIR'%self.confParser['input files']['input_path'],
                                  self.lmpData,
                                  None,# self.lmpDisp,
                                   verbose    = self.verbose,
                                   nconf      = 2, #--- only two events
                                   confParser = self.confParser,
                                   species    = ['Ni'], #'Ni Co Cr'.split()
                                   r_cut      = eval(self.confParser['EnergyBarrier']['r_cut']),
                                   dr         = eval(self.confParser['EnergyBarrier']['dr']),
                                   scale      = eval(self.confParser['EnergyBarrier']['scale']),
                                   n_max      = 8,
                                   l_max      = 6,

                      )
        
        self.perAtomData = self.lmpDataa
        self.SetDescriptors(
                      #soap = False,
                      #acsf = True,   
                      gr = True,
                     )
        
    def GetDefects( self, fp ):
        '''
        Classify Defects
        '''
        
        #--- load ml model
        model = keras.models.load_model(fp)

        #---------------
        #--- zscore X
        #---------------        
        X      = np.c_[self.descriptors ]
        scaler = StandardScaler()
        scaler.fit(X)
        X      = scaler.transform( X )

        #--- predict classes
        predict_x = model.predict( X ) 
        self.predict_classes = np.argmax( predict_x, axis=1 )
    
    
    def GetDisp( self, fp ):
        '''
        Predict Displacements
        '''

        #--- load ml model
        model = keras.models.load_model(fp)
        
        #---------------
        #--- zscore X
        #---------------        
        filtr  = self.predict_classes == 1
        X      = np.c_[self.descriptors[ filtr ] ]
        scaler = StandardScaler()
        scaler.fit(X)
        X      = scaler.transform( X )

        #--- reshape X
        shape      =  (self.shape[0],self.shape[1],self.shape[2],1) #--- rows, cols, thickness, channels
        n          =  X.shape[ 0 ]
        X_reshaped =  X.reshape((n,shape[0],shape[1],shape[2],1))

        self.predict_disp = model.predict( X_reshaped )
        
        #--- energy barriers
        Energy = 1.0
        self.predict_energy = np.ones(n)*Energy
        
        #--- center atoms
        self.atomIDs     = self.lmpDataa[ filtr ].id
        self.atomIndices = self.lmpDataa[ filtr ].index

    
    def BuildCatalog( self ):
        rate_constant_prefactor = 1.0
        kbt = 1.0
        rates = rate_constant_prefactor * np.exp(-self.predict_energy/kbt)
        
        self.catalog = pd.DataFrame( np.c_[self.atomIDs, self.atomIndices, self.predict_energy, rates, self.predict_disp ],
                                     columns = 'AtomId AtomIndex barrier true_rate dx dy dz'.split(),
                                   )
    def MCsampling( self ):
        normalized_rates = np.cumsum( self.catalog.true_rate ) / self.catalog.true_rate.sum()
        n                = len( normalized_rates )
        x                = np.random.random()
        self.event_indx  = np.arange( n )[ x < normalized_rates ][ 0 ]
        
        #--- advance time
        inv_rate         = 1.0 / self.catalog.iloc[ self.event_indx ].true_rate
        self.mc_time    += np.random.exponential( scale = inv_rate )

    def UpdateDisp( self ):
        self.disp[ : ]           = 0.0
        atomIndex                = self.catalog.iloc[ self.event_indx ].AtomIndex.astype( int )
        self.disp[ atomIndex ]   = self.predict_disp[ self.event_indx ]
        self.tdisp[ atomIndex ] += self.predict_disp[ self.event_indx ]
        
    def UpdateCords( self ):
        coords = np.c_[ self.lmpData.coord_atoms_broken[ 0 ]['x y z'.split()] ]
        coords += self.disp
        
        #--- wrap coords ???
        
        
        self.lmpData.coord_atoms_broken[0]['x y z'.split()] = coords

    def Print( self, fout, itime ):
        df = self.lmpData.coord_atoms_broken[ 0 ]
        atomm = lp.Atoms(**df.to_dict(orient='series'),dx=self.disp[:,0],dy=self.disp[:,1],dz=self.disp[:,2])
#         pdb.set_trace()
        #
        wd = lp.WriteDumpFile(atomm, self.box )
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            wd.Write(fp,itime = itime,
                     attrs=['id', 'type', 'x', 'y', 'z','dx','dy','dz'],
                     fmt='%i %i %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')


# ## main()

# In[17]:




mc_steps = 10

mlmc  =  MachineLeranedMC(confParser,
#                           verbose = True
                         )
    
#--- parse atom positions
mlmc.Parse('%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['dump_file']))
    
#--- initialization
mlmc.Initialize()
#mlmc.Print( 'coords.xyz', itime = 0 )
    
for mc_istep in range( mc_steps ):
    #--- build descriptors
    mlmc.GetDescriptors()

    #--- identify defects
    mlmc.GetDefects('%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['classifier_load']))

    #--- predict diffusion paths 
    mlmc.GetDisp('%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_load']))

    #--- build catalog
    mlmc.BuildCatalog()

    #--- mc sampling
    mlmc.MCsampling()

    #--- update disp
    mlmc.UpdateDisp()
   
    #--- save output
    mlmc.Print( 'coords.xyz', itime = mc_istep )
   
    #--- update coord
    mlmc.UpdateCords()
    
    


# In[ ]:





# In[ ]:


mlmc.lmpData


# In[ ]:


mlmc.catalog


# In[ ]:


mlmc.mc_time


# In[ ]:


dir(mlmc)

