#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#MachineLeranedMC" data-toc-modified-id="MachineLeranedMC-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>MachineLeranedMC</a></span><ul class="toc-item"><li><span><a href="#main()" data-toc-modified-id="main()-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>main()</a></span></li></ul></li></ul></div>

# In[1]:


#--- import sys. libs
import pickle
import import_ipynb
import configparser
import numpy as np
import sys
import time
import pandas as pd
import os
import pdb
import imp

#--- tensor flow
import tensorflow as tf
from tensorflow import keras

#--- sklearn
from sklearn.preprocessing import StandardScaler

#--- pytorch
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
from neuralNetwork import GraphNet, GraphLevelGNN, GNNModel, GNNModel3rd #GraphLevelGNN_energy
imp.reload(utl)
imp.reload(lp)
imp.reload(bd)

#--- add graphnet class as attribute of main  
import __main__
setattr(__main__, "GraphNet", GraphNet)
setattr(__main__, "GraphLevelGNN", GraphLevelGNN)
setattr(__main__, "GNNModel", GNNModel)
setattr(__main__, "GNNModel3rd", GNNModel3rd)

#--- lammps
if eval(confParser['flags']['RemoteMachine']):
    import lammps

#--- increase width
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


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
        
        self.verbose     =  verbose 
        self.confParser  =  confParser
        
        self.save_output = 'saved_output'
        if os.path.isdir( self.save_output ):
            get_ipython().system('rm -r $self.save_output; mkdir $self.save_output')
        else: 
            get_ipython().system('mkdir $self.save_output')
        
        #--- assign units
        temperature                  = eval(self.confParser[ 'ml mc' ][ 'temperature' ] ) #--- kelvin
        self.rate_constant_prefactor = 1.0e+13 #s^-1
        self.kbt                     = 8.617e-05 #eV K-1
        self.kbt                    *= temperature
        
        #--- 
        self.model_energy_loaded_true  = False
        self.model_defects_loaded_true = False
        self.model_disps_loaded_true   = False
        
#     def Parse(self,fp):
#         '''
#         Parse lammps dump file
#         '''
#         t0           = time.time()
#         self.lmpData = lp.ReadDumpFile( '%s'%(fp) ) 
#         self.lmpData.GetCords( ncount = sys.maxsize)
#         if self.verbose:
#             print('Parse %s ...'%fp)

    
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
        Compute microstructural descriptors  
        '''
        
        if self.verbose:
            print('Compute nodal features ...')
        bd.EnergyBarrier.__init__( self,
                                  None,
                                  None,
                                  self.lmpData,
                                  None,
                                   verbose    = False,#self.verbose,
                                   nconf      = 2, #--- only two events
                                   confParser = self.confParser,
                                   species    = confParser['input files']['species'].split(),
                                   r_cut      = eval(self.confParser['descriptors']['r_cut']),
                                   dr         = eval(self.confParser['descriptors']['dr']),
                                   dr_acsf    = eval(confParser['descriptors']['dr_acsf']),
                                   scale      = eval(self.confParser['descriptors']['scale']),
                                   n_max      = 8,
                                   l_max      = 6,
                      )
        
        self.perAtomData = self.lmpDataa
        
        #--- parinello: to be used for energy predictions
        self.SetDescriptors(
                      soap = True,   
                     )
        self.descriptors_acsf =  self.descriptors.copy()
        #--- smooth density
        self.SetDescriptors(
                      gr = True,
                     )

    def BuildDataForClassifier( self, df, descriptors ):
        gn                   = GraphNet()
        gn.noise_std         = float(self.confParser['gnn classifier']['noise_std'])
        gn.cutoff            = self.r_cut
        natom                = df.shape[0]
        tmp                  = {'id':np.c_[df.id].flatten(),'atom_indx':np.c_[df.index].flatten(),
                                 'x':np.c_[df.x].flatten(),'y':np.c_[df.y].flatten(),'z':np.c_[df.z].flatten(),
                                 'isNonCrystalline':np.c_[np.zeros(natom)].tolist(),'descriptors_acsf':descriptors.tolist()}
        gn.descriptors       = [tmp, tmp]
        
        #--- build neighbor list:
#         fp                   = 'dump_file_nl'
#         fout                 = 'neighbor_list_xxx.xyz'
#         if os.path.isfile( fout ):
#             os.system('rm %s'%fout)
#         if os.path.isfile( '%s/%s.xyz'%(self.save_output,fp) ):
#             os.system('rm %s/%s.xyz'%(self.save_output,fp))
            
#         #
#         self.Print( fp, 0, lammps_xyz = True ) #--- save as xyz file        
#         atom_indices         = ' '.join(map(str,df.index))
#         lib_path             = confParser['input files']['lib_path'].split()[0]
#         #
#         os.system('mv %s/%s.xyz .'%(self.save_output,fp))
#         os.system('ovitos %s/OvitosCna.py %s.xyz %s 1 6 %s %s'%(lib_path,fp,fout,self.r_cut,atom_indices))
        fout                 = self.dir_neighList
        gn.neighlists        = [ fout, fout ] #--- to be used for computing the adj. matrix

        #--- func. call
        gn.DataBuilderForClassifier()
        
        #--- rm xyz files!!!
#        os.system( 'rm %s.xyz %s'%(fp,fout))
        
        return gn.train_dataloaders.dataset
        
    def GetDefects( self, fp):
        '''
        Classify Defects
        '''
        if self.verbose:
            print('identify defects ...')
        
        #--- load ml model
        if not self.model_defects_loaded_true:
            version_xxx        = os.listdir(fp)[0]
            model              = os.listdir('%s/%s/checkpoints'%(fp,version_xxx))[0]
            self.model_defects = GraphLevelGNN.load_from_checkpoint( '%s/%s/checkpoints/%s'%(fp,version_xxx,model ) ) 
            self.model_defects_loaded_true = True
            
        #--- build data for classifier: 
        self.predict_classes     = self.BuildDataForClassifier(self.lmpData.coord_atoms_broken[ 0 ], #--- nodal xyz
                                                               self.descriptors_acsf, #--- nodal features
                                                              )
        
        self.predict_classes.y   = self.model_defects.predict( self.predict_classes ).int()
        assert torch.any(self.predict_classes.y), 'Detect no defect!'
    
#     def DiscretizeTransitionPath( self ):
#          #--- hard-coded values
#         umax = float( self.confParser[ 'neural net regression' ][ 'umax' ] ) 
#         du   = float( self.confParser[ 'neural net regression' ][ 'du' ] ) 
#         xlin = np.arange(-umax,umax+du,du)
#         ylin = np.arange(-umax,umax+du,du)
#         zlin = np.arange(-umax,umax+du,du)
#         self.nbinx = len(xlin)-1
#         self.nbiny = len(ylin)-1
#         self.nbinz = len(zlin)-1
#         self.bins = (xlin, ylin, zlin)
#         self.ux, self.uy, self.uz = np.meshgrid( self.bins[1][:-1], self.bins[0][:-1], self.bins[2][:-1] )

        
#     def GetDispsFromBinaryMaps( self, atomIndex, binaryMap ):
#         binaryMapReshaped = binaryMap.reshape((self.nbinx, self.nbiny, self.nbinz ))
#         filtr = binaryMapReshaped == 1
#         disps = np.c_[self.uy[filtr],self.ux[filtr],self.uz[filtr]]
#         nrows = disps.shape[ 0 ]
#         assert nrows > 0, 'no diffusion path!'
#         return np.c_[np.ones(nrows)*atomIndex,disps]
    
#     def GetDisp( self, fp, scaler ):
#         '''
#         Predict Displacements
#         '''
        
#         #--- load ml model
#         model               = keras.models.load_model(fp)
        
#         #---------------
#         #--- zscore X
#         #---------------        
#         loaded_scaler       = pickle.load( open( scaler, 'rb' ) )
#         filtr               = self.predict_classes > 0
#         X                   = loaded_scaler.transform( np.c_[self.descriptors[ filtr ] ] )


#         #--- reshape X
#         shape               =  (self.shape[0],self.shape[1],self.shape[2],1) #--- rows, cols, thickness, channels
#         n                   =  X.shape[ 0 ]
#         X_reshaped          =  X.reshape((n,shape[0],shape[1],shape[2],1))
        
#         prediction          =  model.predict( X_reshaped )
#         threshold           = 0.5 #--- hard-coded threshold
#         binary_predictions  = (prediction > threshold).astype(int)

#         #---
#         atomIndices         = self.lmpDataa[ filtr ].index
#         self.predict_disp     = np.concatenate([list(map(lambda x: self.GetDispsFromBinaryMaps( x[0],x[1] ) , zip(atomIndices,binary_predictions) ))])
#         self.predict_disp     = self.predict_disp.reshape((self.predict_disp.shape[0],self.predict_disp.shape[2]))
        
    def BuildDataForRegressor( self, df, binary_nonCrystalAtoms ):
        gn                   = GraphNet()
        gn.noise_std         = float(self.confParser['gnn']['noise_std'])
        gn.c_out             = eval(self.confParser['gnn']['c_out'])
        gn.cutoff            = self.r_cut
        r_cut                = float( self.confParser['descriptors']['cutoff_kmc'] )
        gn.verbose           = False
        gn.transition_paths  = []
        gn.energy            = []
        natom                = df.shape[0]
        nonCrystAtomIndices  = np.arange(natom)[ binary_nonCrystalAtoms ]
        assert len( nonCrystAtomIndices ) > 0, 'no defect detected!'

        #--- build neigh. list for non-crystalline atoms
        #--- cutoff = cutoff_kmc used to define clusters
        #--- save cords as a dump file
        fp                   = 'dump_file_nl'
        fout                 = 'neighbor_list.xyz'
        if os.path.isfile( '%s/%s.xyz'%(self.save_output,fp) ):
            os.system('rm %s/%s.xyz'%(self.save_output,fp))
        if os.path.isfile( fout ):
            get_ipython().system('rm $fout')
        self.Print( fp, 0, lammps_xyz = True ) #--- save as xyz file        
        os.system('mv %s/%s.xyz .'%(self.save_output,fp))
        #--- update df
        df[ 'atom_index' ]   = df.index
        #
        atom_indices         = ' '.join(map(str,nonCrystAtomIndices))
        lib_path             = confParser['input files']['lib_path'].split()[0]
        #
        os.system('ovitos %s/OvitosCna.py %s.xyz %s 1 6 %s %s'%(lib_path,fp,fout,r_cut,atom_indices))
        nl                   = lp.ReadDumpFile(fout)
        nl.GetCords()
        nl                   = nl.coord_atoms_broken[0]
        #
        os.system( 'rm %s.xyz %s'%(fp,fout))

        #--- build clusters for non-crystalline atoms
        groups               = nl.groupby(by='id').groups
        for atomIndx in nonCrystAtomIndices:
            atom_id          = df['id'].iloc[ atomIndx ]
            neighbor_indices_nl = groups[ atom_id ]
            xyz              = np.c_[ nl.iloc[ neighbor_indices_nl ]['DX DY DZ'.split()] ]
            xyz              = np.concatenate([xyz,[[0.0,0.0,0.0]]],axis=0) #--- include center
            #--- descriptors
            neighbor_ids     = list( nl.iloc[ neighbor_indices_nl ].J.astype(int) )
            neighbor_ids    += [ atom_id ]
            neighbor_indices_df = utl.FilterDataFrame(df, key='id', val=neighbor_ids).atom_index
            descriptors      = self.descriptors[ neighbor_indices_df ]
            descriptors_acsf = self.descriptors_acsf[ neighbor_indices_df ]
            center_atom_index= np.zeros( len( neighbor_ids ) )
            center_atom_index[ -1 ] = 1
            n                = len( neighbor_ids )
            diffusion_paths  = np.random.random(size=(3*n)).reshape((n,3))
            energy_barrier   = np.random.random(size=(1,))
            #
            sdict = {'x':xyz[:,0],'y':xyz[:,1],'z':xyz[:,2],
                     'center_atom_index':center_atom_index,
                     'descriptors':descriptors,
                     'descriptors_acsf':descriptors_acsf,
                     'diffusion_paths':diffusion_paths,
                     'energy_barrier':energy_barrier,
                     'atom_indx':neighbor_indices_df
                    }
            gn.transition_paths.append( sdict )
        #
        gn.DataBuilder()        
        loader             = DataLoader(gn.large_graph, batch_size=len(gn.large_graph), shuffle=False)
        #
        gn.DataBuilderForEnergy()
        large_graph        = torch_geometric.data.Batch.from_data_list(gn.graphs)
        loader_energy      = DataLoader(large_graph, batch_size=len(gn.graphs), shuffle=False)

        return loader.dataset, loader_energy.dataset

    
    def GetDisp2nd( self, fp ):
        '''
        Predict Displacements
        '''
        if self.verbose:
            print('predict transition paths ...')
        
        #--- load ml model
        if not self.model_disps_loaded_true:
            self.model_disps   = torch.load( fp, map_location=torch.device('cpu') )
            self.model_disps_loaded_true = True
        
        self.spec_events, self.predict_energy   = self.BuildDataForRegressor(self.lmpData.coord_atoms_broken[ 0 ], #--- nodal coords
                                                        self.predict_classes.y.numpy().astype(bool) #--- binary list: non-crystalline atoms
                                                       )

        self.spec_events.y = self.model_disps( self.spec_events.x, self.spec_events.edge_index )

#     def BuildDataForEnergy( self, df, binary_nonCrystalAtoms ):
#         gn                   = GraphNet()
#         gn.noise_std         = float(self.confParser['gnn energy']['noise_std'])
#         gn.c_out             = eval(self.confParser['gnn energy']['c_out'])
#         gn.cutoff            = self.r_cut
#         r_cut                = float( self.confParser['descriptors']['cutoff_kmc'] )
#         gn.verbose           = False
#         gn.transition_paths  = []
#         natom                = df.shape[0]
#         nonCrystAtomIndices  = np.arange(natom)[ binary_nonCrystalAtoms ]
        
#         #--- build neigh. list
#         #--- save cords as a dump file
#         fp                   = 'dump_file_nl'
#         fout                 = 'neighbor_list.xyz'
#         if os.path.isfile( '%s/%s.xyz'%(self.save_output,fp) ):
#             os.system('rm %s/%s.xyz'%(self.save_output,fp))
#         if os.path.isfile( fout ):
#             !rm $fout
#         self.Print( fp, 0, lammps_xyz = True ) #--- save as xyz file        
#         os.system('mv %s/%s.xyz .'%(self.save_output,fp))
#         #--- update df
#         df[ 'atom_index' ]   = df.index
#         #
#         atom_indices         = ' '.join(map(str,nonCrystAtomIndices))
#         lib_path             = confParser['input files']['lib_path'].split()[0]
#         #
#         os.system('ovitos %s/OvitosCna.py %s.xyz %s 1 6 %s %s'%(lib_path,fp,fout,r_cut,atom_indices))
#         nl                   = lp.ReadDumpFile(fout)
#         nl.GetCords()
#         nl                   = nl.coord_atoms_broken[0]
#         #
#         os.system( 'rm %s.xyz %s'%(fp,fout))

#         #--- build clusters for non-crystalline atoms
#         groups               = nl.groupby(by='id').groups
#         for atomIndx in nonCrystAtomIndices:
#             atom_id          = df['id'].iloc[ atomIndx ]
#             neighbor_indices_nl = groups[ atom_id ]
#             xyz              = np.c_[ nl.iloc[ neighbor_indices_nl ]['DX DY DZ'.split()] ]
#             xyz              = np.concatenate([xyz,[[0.0,0.0,0.0]]],axis=0) #--- include center
#             #--- descriptors
#             neighbor_ids     = list( nl.iloc[ neighbor_indices_nl ].J.astype(int) )
#             neighbor_ids    += [ atom_id ]
#             neighbor_indices_df = utl.FilterDataFrame(df, key='id', val=neighbor_ids).atom_index
#             descriptors      = self.descriptors_acsf[ neighbor_indices_df ]
#             center_atom_index= np.zeros( len( neighbor_ids ) )
#             center_atom_index[ -1 ] = 1
#             n                = len( neighbor_ids )
#             energy_barrier   = np.random.random(size=(1,))
#             #
#             sdict = {'x':xyz[:,0],'y':xyz[:,1],'z':xyz[:,2],
#                      'center_atom_index':center_atom_index,
#                      'descriptors_acsf':descriptors,
#                      'energy_barrier':energy_barrier,
#                      'atom_indx':neighbor_indices_df
#                     }
#             gn.transition_paths.append( sdict )
#         #
#         gn.DataBuilderForEnergy()
#         #        
#         large_graph = torch_geometric.data.Batch.from_data_list(gn.graphs)
#         loader      = DataLoader(large_graph, batch_size=len(gn.graphs), shuffle=False)

#         return loader.dataset
        
    def GetBarrier( self, fp):
        '''
        Predict energy barriers
        '''
        if self.verbose:
            print('predict energetics ...')
        #--- load ml model: load every step !!!
        if not self.model_energy_loaded_true:
            self.model_energy     = torch.load( fp, map_location=torch.device('cpu') ) 
            self.model_energy_loaded_true = True

#         self.predict_energy   = self.BuildDataForEnergy(self.lmpData.coord_atoms_broken[ 0 ], #--- nodal coords
#                                                         self.predict_classes.y.numpy().astype(bool) #--- binary list: non-crystalline atoms
#                                                        )
        self.predict_energy.y = self.model_energy( self.predict_energy.x, 
                                                  self.predict_energy.edge_index, 
                                                  self.predict_energy.batch )

        assert torch.all( self.predict_energy.y > 0.0 ), 'predicted barrier <= 0.0!'
        
#         #--- setup input
#         atomIndices    = self.predict_disp[ :, 0 ].astype( int )

#         pixel_maps_input = np.c_[self.descriptors[ atomIndices ] ]
#         vectors_input    = self.predict_disp[ :, 1: ]
#         X                = np.c_[pixel_maps_input,vectors_input]

        
#         #---------------
#         #--- zscore X
#         #---------------        
#         loaded_scaler       = pickle.load( open( scaler, 'rb' ) )
#         X                   = loaded_scaler.transform( X )


#         #--- reshape X
#         shape               =  (self.shape[0],self.shape[1],self.shape[2],1) #--- rows, cols, thickness, channels: pixel map
#         shape_vector_input  = vectors_input.shape[ 1 ]

#         mdime               = X.shape[ 1 ]
#         X_pixels            = X[:,0:mdime-shape_vector_input]
#         X_vector            = X[:,mdime-shape_vector_input:mdime]
#         n                   =  X.shape[ 0 ]
#         X_pixels            =  X_pixels.reshape((n,shape[0],shape[1],shape[2],1))

        
#         self.predict_energy =  model.predict( [X_pixels,X_vector] )

#     def ReturnCenterAtom( self, disp_array ):
#         disp_sq   = disp_array**2
#         atom_indx = (disp_sq[:,0] + disp_sq[:,1] + disp_sq[:,2]).argmax()
#         return np.concatenate([[atom_indx], disp_array[ atom_indx ]])

    def BuildCatalog( self): #, filtr ):
        center_atoms_rows   = self.spec_events.ptr[ 1 : ] - 1
        disp_max_per_mode   = self.spec_events.y[ center_atoms_rows ] 
        atomIndices         = self.spec_events.atom_indx[ center_atoms_rows ].numpy().flatten()
        atomIDs             = self.lmpData.coord_atoms_broken[0].iloc[atomIndices].id
        atomTypes           = self.lmpData.coord_atoms_broken[0].iloc[atomIndices].type

        self.nmode          = len( self.spec_events.ptr ) - 1
        rates               = self.rate_constant_prefactor * np.exp(-self.predict_energy.y.detach().numpy()/self.kbt)
        
        self.catalog        = pd.DataFrame( np.c_[atomIDs, atomIndices,\
                                                  self.predict_energy.y.detach().numpy(),\
                                                  rates,\
                                                  disp_max_per_mode.detach().numpy() ],
                                     columns = 'AtomId AtomIndex barrier true_rate ux uy uz'.split()
                                   )
    def MCsampling( self ):
        ktot_inv         = 1.0 / self.catalog.true_rate.sum()
        normalized_rates = np.cumsum( self.catalog.true_rate ) * ktot_inv
        n                = len( normalized_rates )
        x                = np.random.random()
        self.event_indx  = np.arange( n )[ x < normalized_rates ][ 0 ]
        
        #--- advance time
        inv_rate         = ktot_inv #1.0 / self.catalog.iloc[ self.event_indx ].true_rate
        self.mc_time    += np.random.exponential( scale = inv_rate )

    def UpdateDisp( self ):
        row_ini                  = self.spec_events.ptr[ self.event_indx ]
        row_fin                  = self.spec_events.ptr[ self.event_indx + 1 ]
        udisp                    = self.spec_events.y[ row_ini : row_fin ].detach().numpy()
        atom_indices             = self.spec_events.atom_indx[ row_ini : row_fin ].numpy().flatten()
        self.disp[ :, : ]        = 0.0
        self.disp[atom_indices]  = udisp 
        
        
        
    def UpdateCords( self ):
        
        coords  = np.c_[ self.lmpData.coord_atoms_broken[ 0 ]['x y z'.split()] ]

        self.lmpData0['x y z'.split()] = coords
        
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
        if 'lammps_xyz' in kwargs and kwargs['lammps_xyz']:
            df    = self.lmpData0
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
        elements = self.confParser['ml mc']['species'].split()
        if 'kart' in kwargs and kwargs['kart']:
            AtomIndices = kwargs[ 'AtomIndices' ] if 'AtomIndices' in kwargs else df.index
            with open('%s/%s'%(self.save_output,fout),'a') as fp:
                #--- half step
                if itime > 0:
                    fp.write('%s\n'%df.iloc[AtomIndices].shape[0])
                    fp.write("Lattice=\" %s \" Time=%e  Step=%s  Energy=0.0  Barrier=%e\n"\
                             %(' '.join(map(str,self.box.CellVector.flatten())),self.mc_time,itime-0.5,self.catalog.iloc[ self.event_indx ].barrier)
                            )
                    for item in np.c_[ df.iloc[AtomIndices] ]:
                        element = elements[int(item[1])-1]
                        fp.write('%s %e %e %e %d\n'%(element,item[2],item[3],item[4],item[0]))
                #
                #--- full step
                fp.write('%s\n'%df.iloc[AtomIndices].shape[0])
                fp.write("Lattice=\" %s \" Time=%e  Step=%s  Energy=0.0  Barrier=%e\n"\
                         %(' '.join(map(str,self.box.CellVector.flatten())),self.mc_time,itime,0.0)
                        )
                for item in np.c_[ df.iloc[AtomIndices] ]:
                    fp.write('Ni %e %e %e %d\n'%(item[2],item[3],item[4],item[0]))

    def PrintMSD( self, fout, itime ):
        msd = self.tdisp.var(axis=0)
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            if itime == 0:
                 fp.write('#  Elapsed Time    Sqr DisplX.      Sqr DisplY.     Sqr DisplZ.  Sqr Displ\n')
#                fp.write('#  ************    ***Total***       Atom Ni        Atom NiV  ********\n')
            fp.write('%e %e %e %e %e\n'%(self.mc_time,msd[0],msd[1],msd[2],msd[0]+msd[1]+msd[2]))

                
    def PrintCatalog( self, fout, itime ):
        df    = self.lmpData.coord_atoms_broken[ 0 ]
        natom = df.shape[ 0 ]
        
        rwj = utl.ReadWriteJson()
        with open('%s/%s'%(self.save_output,fout),'a') as fp:
            rwj.Write([ self.catalog.to_dict( orient = 'list' ) ], fp,
                      mc_time = [ self.mc_time ],
                      mc_step = [ itime ],
                     )
            
        #--- save ovito
        with open('%s/%s'%(self.save_output,'catalog_ovito.xyz'),'a') as fp:
            for imode in range( self.nmode ):
                row_ini = self.spec_events.ptr[imode]
                row_fin = self.spec_events.ptr[imode+1]
                udisp   = self.spec_events.y[row_ini:row_fin].detach().numpy()
                atom_indices = self.spec_events.atom_indx[row_ini:row_fin].numpy().flatten()
                disps = np.zeros(3*natom).reshape((natom,3))
                disps[ atom_indices ] = udisp 
                atomm   = lp.Atoms(**df.to_dict(orient='series'),DisplacementX=disps[:,0],DisplacementY=disps[:,1],DisplacementZ=disps[:,2])
                wd      = lp.WriteDumpFile(atomm, self.box )
                wd.Write(fp,itime,
                          attrs=['id', 'type', 'x', 'y', 'z','DisplacementX','DisplacementY','DisplacementZ'],
                          fmt='%i %i %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')
            
            
        
#     @staticmethod    
#     def AddGaussianNoise(X,scale = 0.1):

#         epsilon_x = np.random.normal(scale=scale,size=X.size).reshape(X.shape)
#         X += epsilon_x

#     @staticmethod
#     def Zscore( X ):
#         scaler = StandardScaler()
#         scaler.fit(X)
#         return scaler.transform( X )
    
#     @staticmethod
#     def compute_adjacency_matrices(input_data, rcut):
#         adj_matrices = []

#         for positions in input_data:
#             num_atoms = positions.shape[0]
#             adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

#             for i in range(num_atoms):
#                 adj_matrix[i, i] = 1
#                 for j in range(i + 1, num_atoms):
#                     distance = torch.norm(positions[i] - positions[j])
#                     if distance <= rcut:
#                         adj_matrix[i, j] = 1
#                         adj_matrix[j, i] = 1
#                 assert adj_matrix[i,:].sum() > 0, 'dangling node : increase the cutoff!'
#             adj_matrices.append(adj_matrix)

#         #--- assert no 
#         return adj_matrices

#     def BuildNeighborList( self, indx, atom_indices,cutoff ):
#         atom_indices = ' '.join(map(str,atom_indices))

#         fp = self.dumpFiles[ indx ] #'%s/lammps_data.dat'%confParser['input files']['input_path']
#         fout = 'neighbor_list.xyz'
#         os.system('rm %s'%fout)
#         lib_path = confParser['input files']['lib_path'].split()[0]
#         #--- neighbor list
#         os.system('ovitos %s/OvitosCna.py %s %s 1 6 %s %s'%(lib_path,fp,fout,cutoff,atom_indices))
#         nl = lp.ReadDumpFile(fout)
#         nl.GetCords()
#         return nl.coord_atoms_broken[0]

#     def GetIndxById( self, atom_ids, indx ):
#         df              = pd.DataFrame(self.transition_paths[ indx ])
#         df['indices']   = range(df.shape[0])
#         atom_indices    = utl.FilterDataFrame(df,key='id',val=atom_ids)['indices']
#         return np.c_[atom_indices].flatten()
            
#     def compute_adjacency_matrices2nd(self,input_data, rcut):
#         adj_matrices       = []
#         edge_attrs         = []
#         for indx, positions in enumerate( input_data ):
#             num_atoms      = positions.shape[0]
#             adj_matrix     = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
#             nl             = self.BuildNeighborList(indx,range(len(positions)),rcut) #--- neighbor list
#             #--- add "index" columns
#             nl['index_i']=self.GetIndxById( np.c_[nl.id].flatten(), indx )
#             nl['index_j']=self.GetIndxById( np.c_[nl.J].flatten(), indx )
#             groups         = nl.groupby(by='id').groups
#             atom_i_ids     = list(groups.keys())
#             atom_i_indices = self.GetIndxById( atom_i_ids, indx )
#             for i, atom_id in zip(atom_i_indices,atom_i_ids):
# #                adj_matrix[i, i] = 1
#                 atom_j_ids       = nl.iloc[groups[ atom_id ]].J
#                 atom_j_indices   = self.GetIndxById( atom_j_ids, indx )
#                 for j, jatom_id in zip(atom_j_indices, atom_j_ids ): #[ atom_j_indices > i ]:
#                     if j < i :
#                         continue
#                     filtr = np.all([nl.id==atom_id,nl.J==jatom_id],axis=0)
#                     edge_features = nl.iloc[ filtr ][ ''.split() ]
#                     adj_matrix[i, j] = 1
#                     adj_matrix[j, i] = 1
#                 assert adj_matrix[i,:].sum() > 0, 'dangling node : increase the cutoff!'
# #            pdb.set_trace()
#             #--- edge attributes
#             keys = 'DX  DY  DZ  PBC_SHIFT_X PBC_SHIFT_Y PBC_SHIFT_Z'.split()
#             indices = adj_matrix.nonzero().numpy()
#             nl_reindexed = nl.set_index(['index_i','index_j'],drop=False)
#             edge_attr = list(map(lambda x: list(nl_reindexed[keys].loc[tuple(x)]),indices))

# #            pdb.set_trace()
#             edge_attrs.append( torch.Tensor( edge_attr ) )
#             adj_matrices.append( adj_matrix )

#         #--- assert no 
#         return adj_matrices, edge_attrs
    
    def PrintDensityMap(self, atomIndx, fout):
        with open(fout,'w') as fp:
#                     disp           = np.c_[self.perAtomData.iloc[atomIndx]['ux uy uz'.split()]].flatten()
                    df             = pd.DataFrame(np.c_[self.positions.T,self.descriptors[atomIndx]],
                                                  columns='x y z mass'.split())
                    utl.PrintOvito(df, fp, ' ', attr_list='x y z mass'.split())
        
    def LammpsInit( self, lmp_script ):
        '''
        run minimization in lammps
        '''
        if self.verbose:
            print('create initial data in lammps ...')
            
        #--- run  lammps
        argss            = ' '.join(lmp_script.split()[1:])
        MEAM_library_DIR = '/mnt/home/kkarimi/Project/git/lammps-27May2021/src/../potentials'
        INC              = '/mnt/home/kkarimi/Project/git/crystalDefect/simulations/lmpScripts'
        args             = "-screen none -var OUT_PATH . -var PathEam %s -var INC %s -var buff 0.0\
                            -var nevery 1000  -var T 2000.0 -var time 1.0\
                            -var DumpFile dumpMin.xyz -var WriteData lammps_data.dat %s "%(MEAM_library_DIR,INC,argss)+\
                           "-var rnd %s -var rnd1 %s -var rnd2 %s -var rnd3 %s"%tuple(np.random.randint(1001,9999,size=4))

        lmp              = lammps.lammps( cmdargs = args.split() )
        lmp.file( "%s/%s"%(INC,lmp_script.split()[0]) )
        
        #--- update coords
        self.lmpData     = lp.ReadDumpFile('lammps_data.dat')
        self.lmpData.ReadData()
    
    def Lammps( self ):
        '''
        run minimization in lammps
        
        version built at: /mnt/home/kkarimi/Project/git/lammps-2Aug2023/src
        
        follow instructions on 'https://docs.lammps.org/Python_head.html'
        '''
        if self.verbose:
            print('minimization in lammps ...')

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
        args             = "-screen none -var OUT_PATH . -var PathEam %s -var INC %s -var buff 0.0 \
                            -var nevery 1000 -var ParseData 1 -var DataFile %s -var ntype 3 -var cutoff 3.54\
                            -var DumpFile dumpMin.xyz -var WriteData data_minimized.dat"%(MEAM_library_DIR,INC,fout)
        lmp              = lammps.lammps( cmdargs = args.split() )
        lmp.file( "%s/in.minimization_constant_volume"%INC )
        
        #--- update coords
#        rd               = lp.ReadDumpFile('data_minimized.dat')
        rd               = lp.ReadDumpFile('dumpMin.xyz')
#        rd.ReadData()
        rd.GetCords()
    
        itime            = list(rd.coord_atoms_broken.keys())[0]
        cords            = np.c_[rd.coord_atoms_broken[itime]['x y z'.split()]]
        disp_minimized   = np.c_[rd.coord_atoms_broken[itime]['c_dsp[1]  c_dsp[2]  c_dsp[3]'.split()]]
        
        self.lmpData.coord_atoms_broken[0]['x y z'.split()] = cords
        self.disp       += disp_minimized
        self.tdisp      += self.disp


# # MachineLeranedMC

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
                                verbose = True
                             )

    #--- parse atom positions
#      mlmc.Parse('%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['dump_file']))
    mlmc.LammpsInit(confParser['ml mc']['lammps_script'])

    #--- initialization
    mlmc.Initialize()
#     mlmc.Print(    'allconf',        itime = 0, lammps_xyz = True, kart = True )
#     mlmc.Print(    'allconf_defect', itime = 0, lammps_xyz = True, kart = True )        
    mlmc.PrintMSD( 'Diffusion.dat',  itime = 0 )
#    mlmc.DiscretizeTransitionPath()
    
    #--- mc loop
    for mc_istep in range( mc_steps ):
        print('mc_istep=',mc_istep)
        
        #--- build descriptors
        mlmc.GetDescriptors()

#         #--- identify defects
        mlmc.GetDefects(fp     = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['classifier_load']))

        #--- predict diffusion paths 
        mlmc.GetDisp2nd(fp        = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_load']))

        #--- predict energy 
        mlmc.GetBarrier(fp     = '%s/%s'%(confParser['ml mc']['input_path'],confParser['ml mc']['regressor_barrier']))

        #--- build catalog
        # fix atoms with type 2
        mlmc.BuildCatalog() # filtr = mlmc.atomTypes == 1 ) #--- only include atomType = 1
#        mlmc.PrintCatalog( 'catalog.json', itime = mc_istep )
        
        #--- mc sampling
        mlmc.MCsampling()

        #--- update disp
        mlmc.UpdateDisp()

        #--- update coord
        mlmc.UpdateCords()

        #--- minimize via lammps: further update disp
        mlmc.Lammps()
        
        #--- save output
#         mlmc.Print( 'allconf', itime = mc_istep + 1, lammps_xyz = True, kart = True )
#         #
#         mlmc.Print( 'allconf_defect', itime = mc_istep + 1,
#                    AtomIndices = list(set(mlmc.catalog.AtomIndex.astype(int))),
#                    lammps_xyz = True, kart = True
#                   )        
        mlmc.PrintMSD( 'Diffusion.dat',  itime = mc_istep + 1 )

            
main()

