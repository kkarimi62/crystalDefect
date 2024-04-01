#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#MachineLeranedMC" data-toc-modified-id="MachineLeranedMC-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>MachineLeranedMC</a></span><ul class="toc-item"><li><span><a href="#main()" data-toc-modified-id="main()-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>main()</a></span></li></ul></li></ul></div>

# In[1]:


import pickle
import import_ipynb
import configparser
import numpy as np
import sys
import time
import pandas as pd
import os
import pdb
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader

#--- user modules
confParser = configparser.ConfigParser() #--- parse conf. file
confParser.read('configuration.ini')
list(map(lambda x:sys.path.append(x), confParser['input files']['lib_path'].split()))
import LammpsPostProcess as lp
import utility as utl
import buildDescriptors as bd
from   neuralNetwork import GraphNet
import __main__
setattr(__main__, "GraphNet", GraphNet)
import imp
imp.reload(utl)
imp.reload(lp)
imp.reload(bd)

if eval(confParser['flags']['RemoteMachine']):
    import lammps

#--- increase width
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# # MachineLeranedMC

# In[ ]:


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
        
        #--- assign units
        temperature                  = eval(self.confParser[ 'ml mc' ][ 'temperature' ] ) #--- kelvin
        self.rate_constant_prefactor = 1.0e+13 #s^-1
        self.kbt                     = 8.617e-05 #eV K-1
        self.kbt                    *= temperature
        
    def Parse(self,fp):
        '''
        Parse lammps dump file
        '''
        t0           = time.time()
        self.lmpData = lp.ReadDumpFile( '%s'%(fp) ) 
        self.lmpData.GetCords( ncount = sys.maxsize)
        if self.verbose:
            print('elapsed time=%s s'%(time.time()-t0))
            print('time steps:',self.lmpData.coord_atoms_broken.keys())
            display(self.lmpData.coord_atoms_broken[0].head())

    
    def Initialize( self ):
        '''
        Initialize variables
        '''
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
                                   species    = confParser['input files']['species'].split(),
                                   r_cut      = eval(self.confParser['descriptors']['r_cut']),
                                   dr         = eval(self.confParser['descriptors']['dr']),
                                   scale      = eval(self.confParser['descriptors']['scale']),
                                   n_max      = 8,
                                   l_max      = 6,
                      )
        
        self.perAtomData = self.lmpDataa
        self.SetDescriptors(
                      #soap = False,
                      #acsf = True,   
                      gr = True,
                     )
        
    def GetDefects( self, fp, scaler ):
        '''
        Classify Defects
        '''
        
        #--- load ml model
        model                = keras.models.load_model(fp)

        #---------------
        #--- zscore X
        #---------------        
        loaded_scaler        = pickle.load( open(scaler, 'rb' ) )
        X                    = loaded_scaler.transform( np.c_[self.descriptors ] )

        #--- predict classes
        predict_x            = model.predict( X ) 
        self.predict_classes = np.argmax( predict_x, axis=1 )
    
    def DiscretizeTransitionPath( self ):
         #--- hard-coded values
        self.umax = 2.5 #--- to be modified
        self.du   = 0.5
        xlin = np.arange(-self.umax,+self.umax,self.du)
        ylin = np.arange(-self.umax,+self.umax,self.du)
        zlin = np.arange(-self.umax,+self.umax,self.du)
        self.nbinx = len(xlin)-1
        self.nbiny = len(ylin)-1
        self.nbinz = len(zlin)-1
        self.bins = (xlin, ylin, zlin)
        self.ux, self.uy, self.uz = np.meshgrid( self.bins[1][:-1], self.bins[0][:-1], self.bins[2][:-1] )

        
    def GetDispsFromBinaryMaps( self, atomIndex, binaryMap ):
        binaryMapReshaped = binaryMap.reshape((self.nbinx, self.nbiny, self.nbinz ))
        filtr = binaryMapReshaped == 1
        disps = np.c_[self.uy[filtr],self.ux[filtr],self.uz[filtr]]
        nrows = disps.shape[ 0 ]
        assert nrows > 0, 'no diffusion path!'
        return np.c_[np.ones(nrows)*atomIndex,disps]
    
    def GetDisp( self, fp ):
        '''
        Predict Displacements
        '''
        
        #--- load ml model
        model = torch.load( fp, map_location=torch.device('cpu') )


#        pdb.set_trace() 
#        model               = keras.models.load_model(fp)
        #---------------
        #--- zscore X
        #---------------        
 #       loaded_scaler       = pickle.load( open( scaler, 'rb' ) )
 #       filtr               = self.predict_classes > 0
#        X                   = loaded_scaler.transform( np.c_[self.descriptors[ filtr ] ] )
        input_data    = [torch.from_numpy( np.c_[self.lmpData.coord_atoms_broken[0]['x y z'.split()]] ).float()]  
        input_data_tensor              = torch.stack(input_data)


        gnn = GraphNet(
                     c_in       = 3,#eval(confParser['gnn']['c_in']),
                     c_hidden   = 32,#eval(confParser['gnn']['c_hidden']),
                     c_out      = 3,#eval(confParser['gnn']['c_out']),
                     num_layers = 3,#eval(confParser['gnn']['num_layers']),
                )  # Move model to GPU

        #--- instance of gnn: verify adj_matrices ?
        adj_matrices      = torch.stack(gnn.compute_adjacency_matrices(input_data_tensor, rcut=3.0)) 
            
        edge_index = adj_matrices[0].nonzero().t()  # Edge indices
                    
        data = Data(x=input_data_tensor[ 0 ], edge_index=edge_index)


        #--- reshape X
#        shape               =  (self.shape[0],self.shape[1],self.shape[2],1) #--- rows, cols, thickness, channels
#        n                   =  X.shape[ 0 ]
#        X_reshaped          =  X.reshape((n,shape[0],shape[1],shape[2],1))
        
                
        predict_disp    = model(data.x, data.edge_index)
        #--- reshape correctly
        natom = predict_disp.shape[ 0 ]
        ndime = 3
        self.nmode = int( predict_disp.shape[ 1 ] / ndime )
        self.predict_disp = predict_disp.reshape(self.nmode, natom, ndime).detach().numpy()

        #---
#        atomIndices         = self.lmpDataa[ filtr ].index
#        self.predict_disp     = np.concatenate([list(map(lambda x: self.GetDispsFromBinaryMaps( x[0],x[1] ) , zip(atomIndices,binary_predictions) ))])
#        self.predict_disp     = self.predict_disp.reshape((self.predict_disp.shape[0],self.predict_disp.shape[2]))
        

    def GetBarrier( self, fp, scaler ):
        '''
        Predict Displacements
        '''
        
        #--- load ml model
        model               = keras.models.load_model(fp)


        #--- setup input
        atomIndices    = self.predict_disp[ :, 0 ].astype( int )

        pixel_maps_input = np.c_[self.descriptors[ atomIndices ] ]
        vectors_input    = self.predict_disp[ :, 1: ]
        X                = np.c_[pixel_maps_input,vectors_input]

        
        #---------------
        #--- zscore X
        #---------------        
        loaded_scaler       = pickle.load( open( scaler, 'rb' ) )
        X                   = loaded_scaler.transform( X )


        #--- reshape X
        shape               =  (self.shape[0],self.shape[1],self.shape[2],1) #--- rows, cols, thickness, channels: pixel map
        shape_vector_input  = vectors_input.shape[ 1 ]

        mdime               = X.shape[ 1 ]
        X_pixels            = X[:,0:mdime-shape_vector_input]
        X_vector            = X[:,mdime-shape_vector_input:mdime]
        n                   =  X.shape[ 0 ]
        X_pixels            =  X_pixels.reshape((n,shape[0],shape[1],shape[2],1))

        
        self.predict_energy =  model.predict( [X_pixels,X_vector] )

        

    def ReturnCenterAtom( self, disp_array ):
        disp_sq = disp_array**2
        atom_indx = (disp_sq[:,0] + disp_sq[:,1] + disp_sq[:,2]).argmax()
        return np.concatenate([[atom_indx], disp_array[ atom_indx ]])
    
    def BuildCatalog( self): #, filtr ):
        #--- center atoms

        # Reshape the array to (natom, nmodes, 3)

        disp_max_per_mode = np.array( list(map(lambda x: self.ReturnCenterAtom(self.predict_disp[x,:,:]), range( self.nmode ))))

        atomIndices = disp_max_per_mode[:,0].astype(int)
        atomIDs        = self.lmpData.coord_atoms_broken[0].iloc[atomIndices].id
        atomTypes      = self.lmpData.coord_atoms_broken[0].iloc[atomIndices].type

        #--- to be implemented
        self.predict_energy = np.ones(self.nmode)

        rates = self.rate_constant_prefactor * np.exp(-self.predict_energy/self.kbt)
#        pdb.set_trace() 
        self.catalog = pd.DataFrame( np.c_[atomIDs, atomIndices, self.predict_energy, rates, disp_max_per_mode[:,1:]],
                                     columns = 'AtomId AtomIndex barrier true_rate ux uy uz'.split()
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
        disps                    = self.predict_disp[self.event_indx,:,:] #self.catalog.iloc[ self.event_indx ]['ux uy uz'.split()]
        self.disp                = disps
        self.tdisp               += disps
        
    def UpdateCords( self ):
        coords  = np.c_[ self.lmpData.coord_atoms_broken[ 0 ]['x y z'.split()] ]
        coords += self.disp        
        
        self.lmpData.coord_atoms_broken[0]['x y z'.split()] = coords
        
        #--- wrap coords
        df      = self.lmpData.coord_atoms_broken[ 0 ]
        atoms   = lp.Atoms(**df['id type x y z'.split() ].to_dict( orient = 'series' ) )
        #  
        wr      = lp.Wrap(atoms, self.box)
        wr.WrapCoord()
        #
        self.lmpData.coord_atoms_broken[0] = pd.DataFrame(atoms.__dict__)

    def Print( self, fout, itime, **kwargs ):
        '''
        save configurations in lammps/kart formats
        '''
        #-----------------------
        #--- lammps format
        #-----------------------
        df    = self.lmpData.coord_atoms_broken[ 0 ]
        atomm = lp.Atoms(**df.to_dict(orient='series'),ux=self.disp[:,0],uy=self.disp[:,1],uz=self.disp[:,2])
        #
        wd    = lp.WriteDumpFile(atomm, self.box )
        with open('%s/%s.xyz'%(self.save_output,fout),'a') as fp:
            wd.Write(fp,itime = itime,
                     attrs=['id', 'type', 'x', 'y', 'z','ux','uy','uz'],
                     fmt='%i %i %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')
            
        #-----------------------
        #--- k-art format
        #-----------------------
        AtomIndices = kwargs[ 'AtomIndices' ] if 'AtomIndices' in kwargs else df.index
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            #--- half step
            if itime > 0:
                fp.write('%s\n'%df.iloc[AtomIndices].shape[0])
                fp.write("Lattice=\" %s \" Time=%e  Step=%s  Energy=0.0  Barrier=%e\n"\
                         %(' '.join(map(str,self.box.CellVector.flatten())),self.mc_time,itime-0.5,self.catalog.iloc[ self.event_indx ].barrier)
                        )
                for item in np.c_[ df.iloc[AtomIndices] ]:
                    fp.write('Ni %e %e %e %d\n'%(item[2],item[3],item[4],item[0]))
            #
            #--- full step
            fp.write('%s\n'%df.iloc[AtomIndices].shape[0])
            fp.write("Lattice=\" %s \" Time=%e  Step=%s  Energy=0.0  Barrier=%e\n"\
                     %(' '.join(map(str,self.box.CellVector.flatten())),self.mc_time,itime,0.0)
                    )
            for item in np.c_[ df.iloc[AtomIndices] ]:
                fp.write('Ni %e %e %e %d\n'%(item[2],item[3],item[4],item[0]))

    def PrintMSD( self, fout, itime ):
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            if itime == 0:
                fp.write('#  Elapsed Time    Sqr Displ.      Sqr Displ.     Sqr Displ.  KMC step\n')
                fp.write('#  ************    ***Total***       Atom Ni        Atom NiV  ********\n')
            fp.write('0.00000000E+00      0.0000000      0.0000000      0.0000000         %d\n'%itime)

                
    def PrintCatalog( self, fout, itime ):
        rwj = utl.ReadWriteJson()
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            rwj.Write([ self.catalog.to_dict( orient = 'list' ) ], fp,
                      mc_time = [ self.mc_time ],
                      mc_step = [ itime ],
                     )
            
        #--- save ovito
#        indices = self.catalog.AtomIndex.astype( int )
        df      = self.lmpData.coord_atoms_broken[ 0 ]
        with open('%s/%s'%(self.save_output,'catalog_ovito.xyz'),'a') as fp:
              for imode, disps in enumerate(self.predict_disp): #   = np.c_[self.catalog[ 'ux uy uz'.split() ]]
                     atomm   = lp.Atoms(**df.to_dict(orient='series'),DisplacementX=disps[:,0],DisplacementY=disps[:,1],DisplacementZ=disps[:,2])
        #
                     wd      = lp.WriteDumpFile(atomm, self.box )
                     wd.Write(fp,itime,
                              attrs=['id', 'type', 'x', 'y', 'z','DisplacementX','DisplacementY','DisplacementZ'],
                              fmt='%i %i %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')
            
            
#         for AtomIndex in self.catalog.AtomIndex.astype( int ):
#             fout = '%s/catalog_descriptors_atomIndx%s.xyz'%(self.save_output,AtomIndex)
#             self.PrintDensityMap(AtomIndex, fout)
        
    @staticmethod    
    def AddGaussianNoise(X,scale = 0.1):

        epsilon_x = np.random.normal(scale=scale,size=X.size).reshape(X.shape)
        X += epsilon_x

    @staticmethod
    def Zscore( X ):
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler.transform( X )
    
    def PrintDensityMap(self, atomIndx, fout):
        with open(fout,'w') as fp:
#                     disp           = np.c_[self.perAtomData.iloc[atomIndx]['ux uy uz'.split()]].flatten()
                    df             = pd.DataFrame(np.c_[self.positions.T,self.descriptors[atomIndx]],
                                                  columns='x y z mass'.split())
                    utl.PrintOvito(df, fp, ' ', attr_list='x y z mass'.split())
    
    def Lammps( self ):
        '''
        run minimization in lammps
        
        version built at: /mnt/home/kkarimi/Project/git/lammps-2Aug2023/src
        
        follow instructions on 'https://docs.lammps.org/Python_head.html'
        '''
        
        #--- lammps data file
        df               = self.lmpData.coord_atoms_broken[ 0 ]
        atom             = lp.Atoms(**df['id type x y z'.split() ].to_dict( orient = 'series' ) )
        mass             = dict(zip(set(df.type),np.ones(len(set(df.type)))))
        wd               = lp.WriteDataFile(atom, self.box, mass) #--- modify!!
        fout             = 'lammps.dat'
        wd.Write( fout )

            
        #--- run lammps
        MEAM_library_DIR = '/mnt/home/kkarimi/Project/git/lammps-27May2021/src/../potentials'
        INC              = '/mnt/home/kkarimi/Project/git/crystalDefect/simulations/lmpScripts'
        args             = "-var OUT_PATH . -var PathEam %s -var INC %s -var buff 0.0 \
                            -var nevery 1000 -var ParseData 1 -var DataFile %s -var ntype 3 -var cutoff 3.54\
                            -var DumpFile dumpMin.xyz -var WriteData data_minimized.dat"%(MEAM_library_DIR,INC,fout)
        lmp              = lammps.lammps( cmdargs = args.split() )
        lmp.file( "%s/in.minimization_constant_volume"%INC )
        
        #--- update coords
        rd               = lp.ReadDumpFile('data_minimized.dat')
        rd.ReadData()
        cords            = np.c_[rd.coord_atoms_broken[0]['x y z'.split()]]
        self.lmpData.coord_atoms_broken[0]['x y z'.split()] = cords


# In[16]:


# fout = 'junk.json'
# data={'eng':[1966,1974],'bra':[1970,1994]}
# df=pd.DataFrame(data)
# rwj = utl.ReadWriteJson()
# rwj.Write([df.to_dict(orient='list')],fout,
#          itime=[10],
#          )
# #help(utl.ReadWriteJson)


# ## main()

# In[4]:


def main():

    mc_steps = eval(confParser['ml mc']['mc_steps'])



    mlmc     = MachineLeranedMC(confParser,
    #                           verbose = True
                             )

    #--- parse atom positions
    mlmc.Parse('%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['dump_file']))

    #--- initialization
    mlmc.Initialize()
#    mlmc.Print(    'allconf',        itime = 0 )
#    mlmc.Print(    'allconf_defect', itime = 0 )        
    mlmc.PrintMSD( 'Diffusion.dat',  itime = 0 )
#    mlmc.DiscretizeTransitionPath()
    
    #--- mc loop
    for mc_istep in range( mc_steps ):
        print('mc_istep=',mc_istep)
        
        #--- build descriptors
#        mlmc.GetDescriptors()

        #--- identify defects
#        mlmc.GetDefects(fp     = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['classifier_load']),
#                        scaler = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['classifier_scaler'])
#                       )

        #--- predict diffusion paths 
        mlmc.GetDisp(fp        = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_load']),
#                     scaler    = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_scaler'])

                    )

        #--- predict energy 
#        mlmc.GetBarrier(fp     = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_barrier']),
#                        scaler = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_en_scaler'])

#                    )

        #--- build catalog
        mlmc.BuildCatalog() # filtr = mlmc.atomTypes == 1 ) #--- only include atomType = 1
        mlmc.PrintCatalog( 'catalog.json', itime = mc_istep )
        
        #--- mc sampling
        mlmc.MCsampling()

        #--- update disp
        mlmc.UpdateDisp()

        #--- save output
#        mlmc.Print( 'allconf', itime = mc_istep + 1 )
        #
        mlmc.Print( 'allconf_defect', mc_istep + 1, 
                   AtomIndices = list(set(mlmc.catalog.AtomIndex.astype(int))),
					junk='xxx',
                  )        
#        mlmc.PrintMSD( 'Diffusion.dat', itime = mc_istep + 1 )

        #--- update coord
        mlmc.UpdateCords()

        #--- minimize via lammps
        mlmc.Lammps()
    
main()

