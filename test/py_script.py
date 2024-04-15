#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#import-libs" data-toc-modified-id="import-libs-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>import libs</a></span></li><li><span><a href="#Train-NN" data-toc-modified-id="Train-NN-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train NN</a></span><ul class="toc-item"><li><span><a href="#main():-classifier" data-toc-modified-id="main():-classifier-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>main(): classifier</a></span></li><li><span><a href="#main():-regressor" data-toc-modified-id="main():-regressor-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>main(): regressor</a></span><ul class="toc-item"><li><span><a href="#Plot" data-toc-modified-id="Plot-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Plot</a></span></li></ul></li><li><span><a href="#test-example:-2d" data-toc-modified-id="test-example:-2d-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>test example: 2d</a></span><ul class="toc-item"><li><span><a href="#fully-connected-in-sklearn" data-toc-modified-id="fully-connected-in-sklearn-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>fully connected in sklearn</a></span></li><li><span><a href="#fully-connected-in-keras" data-toc-modified-id="fully-connected-in-keras-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>fully connected in keras</a></span></li><li><span><a href="#cnn" data-toc-modified-id="cnn-2.3.3"><span class="toc-item-num">2.3.3&nbsp;&nbsp;</span>cnn</a></span></li></ul></li></ul></li></ul></div>

# # import libs

# In[1]:


import configparser
confParser = configparser.ConfigParser()

#--- parse conf. file
confParser.read('configuration.ini')
print('conf. file sections:',confParser.sections())

#--- system libs
import os
import sys
list(map(lambda x:sys.path.append(x), confParser['input files']['lib_path'].split()))
import pdb
import time
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
if not eval(confParser['flags']['RemoteMachine']):
    plt.rc('text', usetex=True)
import pickle

#--- ase
# from dscribe.descriptors import SOAP, ACSF
# import ase
# import ase.io
# import ase.build
# from ase.io import lammpsdata


#--- sklearn
import sklearn
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
#
from scipy.stats import gaussian_kde

#--- tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.data import Data, DataLoader

# PL callbacks
#from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

#--- user modules
import LammpsPostProcess as lp
import utility as utl
import imp
imp.reload(utl)
imp.reload(lp)

#--- increase width
#from IPython.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))


# # Train NN

# In[2]:



    
    
class GNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        edge_dim,
        num_layers=1,
        layer_name="GCN",
        dp_rate=0.1,
        verbose=True,
        **kwargs,
    ):
        """GNNModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of "hidden" graph layers
            layer_name: String of the graph layer to use
            dp_rate: Dropout rate to apply throughout the network
            kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer_by_name = {"linear":geom_nn.Linear,"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, 
                             "GraphConv": geom_nn.GraphConv}
        gnn_layer         = gnn_layer_by_name[layer_name]
        layers = []
        in_channels       = c_in
        for l_idx in range(num_layers-1):
            out_channels      = c_hidden[ l_idx ]
            layers       += [
                gnn_layer(in_channels=in_channels, 
                                           out_channels=out_channels, 
                                           **kwargs),
                nn.ReLU(inplace=True),
                 nn.Dropout(dp_rate),
            ]
            in_channels       = out_channels
        layers            += [gnn_layer(in_channels=in_channels, 
                                                         out_channels=c_out, 
                                                         **kwargs)]
        self.layers        = nn.ModuleList(layers)

        if verbose:
            for indx, layer in enumerate( self.layers ):
                 print('layer %s'%indx, layer )


    def forward(self, x, edge_index ):
        """Forward.

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for layer in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(layer, geom_nn.MessagePassing):
                x    = layer(x, edge_index)
            else:
                x          = layer(x)
        return x


# In[29]:


class GraphNet( GNNModel ):
    def __init__(self,**kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

        GNNModel.__init__(  self, 
                            c_in       = self.c_in,
                            c_hidden   = self.c_hidden,
                            c_out      = self.c_out,
                            edge_dim   = self.edge_dim,
                            num_layers = self.num_layers,
                            layer_name = 'linear',#"GCN",
                            dp_rate    = 0.5,
                        )
    
    def Parse(self,path,nruns):

        self.Catalogs         = {}
        self.transition_paths = []
        self.dumpFiles        = []
        self.descriptors      = []
        self.neighlists       = []
#
        if self.verbose:
            print('parsing %s'%path)
        rwjs = utl.ReadWriteJson()
        for irun in range(nruns):
            try:
                self.descriptors.extend( rwjs.Read('%s/Run%s/descriptors/descriptors_gr.json'%(path,irun)) )
                self.transition_paths.extend( rwjs.Read('%s/Run%s/saved_output/transition_paths.json'%(path,irun)) )
                self.neighlists.append( '%s/Run%s/neighList/neigh_list.xyz'%(path,irun))
                self.dumpFiles.append('%s/Run%s/dumpFile/dump.xyz'%(path,irun))
                os.system('ln -s %s/Run%s/dumpFile/dump.xyz ./dump.%s.xyz'%(path,irun,irun))
                self.Catalogs[irun]     = pd.read_csv('%s/Run%s/saved_output/catalog.txt'%(path,irun))
            except:
                if self.verbose:
                    traceback.print_exc()
                continue
                
        
        self.nruns     = list(self.Catalogs.keys())
        self.nruns.sort()
                                        
        assert len(self.Catalogs) == len(self.transition_paths) == len(self.dumpFiles) == len(self.descriptors) == len(self.neighlists     )        
        
 
    def GenerateDate(self):
#        disp=[]
        
        dx=0.2
        X=np.arange(0,1.0,dx) #linspace(0,1.0,n)
        n=len(X)
#        dx = X[1]-X[0]
        assert n%2==1

        vacancy = 0
        while vacancy == 0 or vacancy == n-1:
            vacancy = int(np.random.random()*n)
        assert 0<vacancy<n-1
        print('vacancy=%s'%vacancy)
        
        occupancy=np.ones(n,dtype=bool)
        occupancy[vacancy] =False       
        filtr = occupancy 

        disp=np.zeros(n*2).reshape((n,2))
        iposn = vacancy-1
        disp[iposn,0]=1.0
        iposn = vacancy+1
        disp[iposn,1]=-1
        
        disp = disp[filtr,:]
#        disp = disp[:,~np.all([disp==0.0],axis=1).flatten()]

#        np.random.shuffle(disp)

        adj_mat = torch.zeros((n, n), dtype=torch.float)
        for i in range(n):
            adj_mat[i,i]=1
            adj_mat[i,(i+1)%n]=1
            adj_mat[i,(i-1)%n]=1
        adj_mat = adj_mat[:,filtr][filtr]

#        print('adj_mat=\n',adj_mat)
#        m = int((n-3)/2)
#        x = np.array([np.concatenate([np.ones(m),[adj_mat[i,(i-1)%n],adj_mat[i,i],adj_mat[i,(i+1)%n]],np.ones(m)]) for i in range(n)])        
        values = X[filtr]
        x = np.concatenate([list(map(lambda x:self.Density1d(X,x,filtr),values))])

        return x,disp,adj_mat
        
    def Density1d(self,xv,query_point,filtr):
         #--- density
        position = query_point
        xrand = xv.copy()
        xrand -= position
        xrand += (xrand >= 0.5) * (-1.0)
        xrand += (xrand < -0.5) * (+1.0)
        assert np.all([-0.5 <= xrand,xrand <= 0.5])
        values = xrand[filtr]

        X =np.arange(-.5,0.5,0.01)
#        pdb.set_trace()
        kernel = gaussian_kde(values,bw_method=0.07)
        Z = kernel(X)
#        ax=utl.PltErr(values,np.ones(len(values)),Plot=False)
#        utl.PltErr(X,Z,title='rho.png',Plot=False,ax=ax)
        return Z

    def Densityy(self,xyc,query_point):
         #--- density
        position = query_point
        xy_rand = xyc.copy()
        xrand = xy_rand[:,0]
        yrand = xy_rand[:,1]

        #--- center data
        xrand -= position[0]
        yrand -= position[1]
        xrand += ((xrand > 0.5) * (-1.0))
        xrand += ((xrand <= -0.5) * (+1.0))
        yrand += ((yrand > 0.5) * (-1.0))
        yrand += ((yrand <= -0.5) * (+1.0))
        assert np.all([np.abs(xrand) <= 0.5])
        assert np.all([np.abs(yrand) <= 0.5])
        values = np.c_[xrand.flatten(), yrand.flatten()]

        X, Y = np.meshgrid(np.arange(-.5,0.5,0.1), np.arange(-0.5,0.5,0.1))#, indexing='ij')
#        pdb.set_trace()
        xy = np.c_[X.flatten(),Y.flatten()].T
        kernel = gaussian_kde(values.T,bw_method=0.07)
        Z = kernel(xy).T
        Z[np.abs(Z)<1.0e-06] = 0.0 
#        cordc = pd.DataFrame(np.c_[xy.T,Z],columns ='x y mass'.split() )
#        with open('rho.xyz','a') as fp:
#               utl.PrintOvito(cordc,fp,'x=%e,y=%e'%(query_point[0],query_point[1]),attr_list='x y mass'.split())
#        cordc = pd.DataFrame(np.c_[values],columns ='x y'.split() )
#        with open('atoms.xyz','a') as fp:
#               utl.PrintOvito(cordc,fp,'x=%e,y=%e'%(query_point[0],query_point[1]),attr_list='x y'.split())

        return Z

    def Density3d(self,xyc,query_point,dx,ndime, dr, l ):
         #--- density
        position = query_point
        xy_rand = xyc.copy()
        xrand = xy_rand[:,0]
        yrand = xy_rand[:,1]
        zrand = xy_rand[:,2]

        #--- center data
        xrand -= position[0]
        yrand -= position[1]
        zrand -= position[2]
        xrand += ((xrand > 0.5) * (-1.0))
        xrand += ((xrand <= -0.5) * (+1.0))
        yrand += ((yrand > 0.5) * (-1.0))
        yrand += ((yrand <= -0.5) * (+1.0))
        zrand += ((zrand > 0.5) * (-1.0))
        zrand += ((zrand <= -0.5) * (+1.0))
        assert np.all([np.abs(xrand) <= 0.5])
        assert np.all([np.abs(yrand) <= 0.5])
        assert np.all([np.abs(zrand) <= 0.5])
        values = np.c_[xrand.flatten(), yrand.flatten(),zrand.flatten()]
#        l = 1.01*dx
        X, Y, zz = np.meshgrid(np.arange(-l,l,dr), 
                               np.arange(-l,l,dr), 
                               np.arange(-l,l,dr))#, indexing='ij')
        xy = np.c_[X.flatten(),Y.flatten(), zz.flatten()].T
        if ndime == 2:
            X, Y, zz = np.meshgrid(np.arange(-l,l,dr), 
                               np.arange(-l,l,dr), 
                               [0.5])#, indexing='ij')
            xy = np.c_[X.flatten(),Y.flatten()].T
         
        
        
#         pdb.set_trace()
        kernel = gaussian_kde(values[:,:ndime].T,bw_method=0.07)
        Z = kernel(xy).T
        Z[np.log10(np.abs(Z))<-3.0] = 1.0e-03 
        #
        xy = np.c_[X.flatten(),Y.flatten(), zz.flatten()].T
        cordc = pd.DataFrame(np.c_[xy.T,Z],columns ='x y z mass'.split() )
        with open('rho.xyz','a') as fp:
              utl.PrintOvito(cordc,fp,'x=%e,y=%e'%(query_point[0],query_point[1]),attr_list='x y z mass'.split())
        cordc = pd.DataFrame(np.c_[values],columns ='x y z'.split() )
        with open('atoms.xyz','a') as fp:
              utl.PrintOvito(cordc,fp,'x=%3.2e,y=%3.2e,z=%3.2e'%(query_point[0],query_point[1],query_point[2]),attr_list='x y z'.split())

        return Z

    def GenerateDate_2d(self):
        disp=[]
    
        dx=dy=0.2
        X=Y=np.arange(0,1,dx) #linspace(0,1.0,n)
        n=m=len(X)
        xv, yv = np.meshgrid(X,Y) #,indexing='ij')
        values = np.c_[xv.flatten(),yv.flatten()]

        #--- rm atom at random
        vacancy_xindx = 0
        vacancy_yindx = 0
        while vacancy_xindx == 0 or vacancy_xindx == n-1:
            vacancy_xindx = int(np.random.random()*n)
        assert 0<vacancy_xindx<n-1
        while vacancy_yindx == 0 or vacancy_yindx == m-1:
            vacancy_yindx = int(np.random.random()*m)
        assert 0<vacancy_yindx<m-1
        print('vacancy_xindx=%s,vacancy_yindx=%s'%(vacancy_xindx,vacancy_yindx))
        occupancy = np.ones(n*m,dtype=bool).reshape((m,n))
        occupancy[vacancy_yindx,vacancy_xindx] = False        
        filtr = occupancy.flatten()
        xv_vac = values[~filtr]

        values = values[filtr]
        np.random.shuffle( values )

        #--- adjacency matrix
        sizet = values.shape[0]
        adj_mat = torch.zeros((sizet, sizet), dtype=torch.float)
#        adj_mat[0,-1] = adj_mat[-1,0]=1.0
        cutoff = 1.01*(dx*dx)
        for i in range(sizet):
            adj_mat[i,i]=1
            for j in range(i+1,sizet):
                xij = values[j,0]-values[i,0]
                yij = values[j,1]-values[i,1]
                if xij > 0.5:
                    xij -= 1.0
                elif xij <= -0.5:
                    xij += 1.0
                if yij > 0.5:
                    yij -= 1.0
                elif yij <= -0.5:
                    yij += 1.0
                assert np.abs(xij)<0.5 and  np.abs(yij)<0.5
                if xij*xij+yij*yij < cutoff:
                    adj_mat[i,j]=adj_mat[j,i]=1

        #--- density
        x = np.concatenate([list(map(lambda x:self.Densityy(values,x),values))])

        #--- tranition paths
        neighbors = adj_mat.sum(axis=0) == 4
        assert neighbors.sum() == 4
        disp=xv_vac-values[neighbors]
        dispx = disp[:,0]
        dispy = disp[:,1]
        dispx += ((dispx > 0.5) * (-1.0))
        dispx += ((dispx <= -0.5) * (+1.0))
        dispy += ((dispy > 0.5) * (-1.0))
        dispy += ((dispy <= -0.5) * (+1.0))
        assert np.all([np.abs(dispx) <= 0.5])
        assert np.all([np.abs(dispy) <= 0.5])
        dispt = np.c_[dispx,dispy]       

        #--- assemble in global matrix
        natom   = sizet
        indices = np.arange(natom)[neighbors]
        nmode = sizet
        disp = np.zeros(sizet*nmode*ndime).reshape((sizet,nmode*ndime))
        for imode,atom_indx in enumerate(indices): 
            modei = atom_indx
            for idime in range(ndime):
        	    disp[atom_indx,modei*ndime+idime]=dispt[imode,idime]
        filtr=[]
        for imode,atom_indx in enumerate(range(natom)): 
            modei = atom_indx
            zero = np.all([np.all([disp[:,modei*ndime] == 0.0]),np.all([disp[:,modei*ndime+1] == 0.0])])
            
            filtr+=[zero,zero]
        filtr = np.array(filtr)
        assert (~filtr).sum() == 8
        disp = disp[:,~filtr]
        disp = disp[:,:2]
        self.PrintOvit(values,disp,adj_mat)

        return x,disp,adj_mat,values

    def nnlist(self,adj_mat,cutoff,sizet,values): 
        for i in range(sizet):
            adj_mat[i,i]=1
            for j in range(i+1,sizet):
                xij = values[j,0]-values[i,0]
                yij = values[j,1]-values[i,1]
                zij = values[j,2]-values[i,2]
                if xij > 0.5:
                    xij -= 1.0
                elif xij <= -0.5:
                    xij += 1.0
                if yij > 0.5:
                    yij -= 1.0
                elif yij <= -0.5:
                    yij += 1.0
                if zij > 0.5:
                    zij -= 1.0
                elif zij <= -0.5:
                    zij += 1.0
                assert np.abs(xij)<=0.5 and  np.abs(yij)<=0.5 and  np.abs(zij)<=0.5
                if xij*xij+yij*yij+zij*zij < cutoff:
                    adj_mat[i,j]=adj_mat[j,i]=1
        return adj_mat
    
    def Generate_subGraphs(self,ndime):
        disp=[]
         
        ndime = 2
        dx=dy=0.1
        X=Y=zz=np.arange(0,1,dx) #linspace(0,1.0,n)
        if ndime == 2:
            zz=[0.5]
        
        n=m=l=len(X)
        xv, yv, zv = np.meshgrid(X,Y,zz) #,indexing='ij')
        values = np.c_[xv.flatten(),yv.flatten(),zv.flatten()]
        if ndime == 2:
            values[:,ndime]=0.0
            l=1
        print('n,m,l=',n,m,l)
        #--- rm atom at random
        vacancy_xindx = 0
        vacancy_yindx = 0
        vacancy_zindx = 0
        while vacancy_xindx == 0 or vacancy_xindx == n-1:
            vacancy_xindx = int(np.random.random()*n)
        assert 0<vacancy_xindx<n-1
        while vacancy_yindx == 0 or vacancy_yindx == m-1:
            vacancy_yindx = int(np.random.random()*m)
        assert 0<vacancy_yindx<m-1
        if ndime == 3:
            while vacancy_zindx == 0 or vacancy_zindx == l-1:
                vacancy_zindx = int(np.random.random()*l)
            assert 0<vacancy_zindx<l-1
        print('vacancy_xindx=%s,vacancy_yindx=%s,vacancy_zindx=%s'%(vacancy_xindx,vacancy_yindx,vacancy_zindx))
        occupancy = np.ones(n*m*l,dtype=bool).reshape((m,n,l))
        occupancy[vacancy_yindx,vacancy_xindx, vacancy_zindx] = False        
        filtr = occupancy.flatten()
        xv_vac = values[~filtr]

        values = values[filtr]
        np.random.shuffle( values )

        #--- adjacency matrix
        sizet = values.shape[0]
        adj_mat = torch.zeros((sizet, sizet), dtype=torch.float)
#        adj_mat[0,-1] = adj_mat[-1,0]=1.0
        self.nnlist(adj_mat, 1.01*(dx*dx),sizet,values)



        #--- neighbor list
        atom_indices = np.arange(sizet)
        adj_matt0 = torch.zeros((sizet, sizet), dtype=torch.float)
        self.nnlist(adj_matt0, (1.01)*(dx*dx),sizet,values)
        pairs        = adj_matt0.nonzero()
        nl           = pd.DataFrame(np.c_[pairs],columns='atom_i_index atom_j_index'.split())


        
        #--- density
        dr = 0.2*dx
        l = np.sqrt(2.0)*dx+0.01
        x = np.concatenate([list(map(lambda x:self.Density3d(values,x,dx,ndime, dr, l),values))])

        #--- tranition paths
#         pdb.set_trace()
        n_nearest = 4 if ndime == 2 else 6
        neighbors = adj_mat.sum(axis=0) == n_nearest #6 #4
        assert neighbors.sum() == n_nearest #6 #4
        disp=xv_vac-values[neighbors]
        dispx = disp[:,0]
        dispy = disp[:,1]
        dispz = disp[:,2]
        dispx += ((dispx > 0.5) * (-1.0))
        dispx += ((dispx <= -0.5) * (+1.0))
        dispy += ((dispy > 0.5) * (-1.0))
        dispy += ((dispy <= -0.5) * (+1.0))
        dispz += ((dispz > 0.5) * (-1.0))
        dispz += ((dispz <= -0.5) * (+1.0))
        assert np.all([np.abs(dispx) <= 0.5])
        assert np.all([np.abs(dispy) <= 0.5])
        assert np.all([np.abs(dispz) <= 0.5])
        dispt = np.c_[dispx,dispy,dispz]       

        #--- assemble in global matrix
        natom   = sizet
        indices = np.arange(natom)[neighbors]
        nmode = sizet
        disp = np.zeros(sizet*nmode*3).reshape((sizet,nmode*3))
        for imode,atom_indx in enumerate(indices): 
            modei = atom_indx
            for idime in range(3):
        	    disp[atom_indx,modei*3+idime]=dispt[imode,idime]
        filtr=[]
        for imode,atom_indx in enumerate(range(natom)): 
            modei = atom_indx
            ff=[]
            for idime in range(3):
                ff.append([np.all([disp[:,modei*3+idime] == 0.0])])
            zero = np.all(ff)
            filtr+=[zero,zero,zero]
        filtr = np.array(filtr)
        print((~filtr).sum())
        assert (~filtr).sum() == n_nearest*3
        disp = disp[:,~filtr]
#         disp = disp[:,:ndime]
#         self.PrintOvit(values,disp,adj_mat)
        
        
        #--- adj-matrices for non-crystalline atoms
        #--- sub-blocks of adj_mat: use neighbor list
        adj_mat_sub_blocks = []
        coords_sub_blocks = []
        rho_sub_blocks    = []
        disp_blocks       = []
        defects=[]
        center_atom_inices = atom_indices[ adj_mat.sum(axis=0) == n_nearest ]
        os.system('rm ovito.xyz rho.xyz atoms.xyz')
        X, Y, zz = np.meshgrid(np.arange(-l,l,dr), 
                               np.arange(-l,l,dr), 
                               np.arange(-l,l,dr))#, indexing='ij')
        if ndime == 2:
            X, Y, zz = np.meshgrid(np.arange(-l,l,dr), 
                               np.arange(-l,l,dr), 
                               [0.5])#, indexing='ij')

        grid = np.c_[X.flatten(),Y.flatten(), zz.flatten()]
        print('grid.shape:',grid.shape)
#         pdb.set_trace()
        for center_atom_indx in center_atom_inices:
            
            #--- get neighbors
            atom_js_indices = list( nl.iloc[nl.groupby(by='atom_i_index').groups[ center_atom_indx ]].atom_j_index )
            atom_js_indices.sort()
            print(center_atom_indx,len(atom_js_indices))

            #--- adj matrix
            adj_mat_per_center = adj_mat[atom_js_indices,:][:,atom_js_indices]
            print('adj_mat_per_center.shape:',adj_mat_per_center.shape)            

            
#            neighbors_per_center = adj_mat_per_center.sum(axis=0) == n_nearest #6 #4
            neighbors_per_center = atom_js_indices == center_atom_indx
            #--- xyz
            cord_per_center    = values[atom_js_indices]
            
            #--- density
            rho = x[atom_js_indices,:] #np.concatenate([list(map(lambda x:self.Density3d(cord_per_center,x),cord_per_center))])
            rho_center = x[center_atom_indx,:]
            position = values[center_atom_indx]
            xy_rand = cord_per_center.copy()
            xrand = xy_rand[:,0]
            yrand = xy_rand[:,1]
            zrand = xy_rand[:,2]

            #--- center data
            xrand -= position[0]
            yrand -= position[1]
            zrand -= position[2]
            xrand += ((xrand > 0.5) * (-1.0))
            xrand += ((xrand <= -0.5) * (+1.0))
            yrand += ((yrand > 0.5) * (-1.0))
            yrand += ((yrand <= -0.5) * (+1.0))
            zrand += ((zrand > 0.5) * (-1.0))
            zrand += ((zrand <= -0.5) * (+1.0))
            assert np.all([np.abs(xrand) <= 0.5])
            assert np.all([np.abs(yrand) <= 0.5])
            assert np.all([np.abs(zrand) <= 0.5])
            valuess = np.c_[xrand.flatten(), yrand.flatten(),zrand.flatten()]
            cordc = pd.DataFrame(np.c_[grid,rho_center],columns ='x y z mass'.split() )
            with open('rho.xyz','a') as fp:
                  utl.PrintOvito(cordc,fp,'x=%e,y=%e,z=%e'%(position[0],position[1],position[2]),attr_list='x y z mass'.split())
            cordc = pd.DataFrame(np.c_[valuess],columns ='x y z'.split() )
            with open('atoms.xyz','a') as fp:
                  utl.PrintOvito(cordc,fp,'x=%3.2e,y=%3.2e,z=%3.2e'%(position[0],position[1],position[2]),attr_list='x y z'.split())


            #--- disp
            natom_per_cluster = len(atom_js_indices)
            disp_per_cluster = np.zeros(natom_per_cluster * 3).reshape((natom_per_cluster,3))
            new_indices = np.arange( natom_per_cluster )
            center_atom_new_index = new_indices[atom_js_indices == center_atom_indx]
            disp_per_cluster[center_atom_new_index,:] = dispt[atom_indices[neighbors]==center_atom_indx,:]

            #--- plot
            fout2 = 'lammps%s.data'%center_atom_indx
            self.PrintOvit(cord_per_center,disp_per_cluster,adj_mat_per_center, 'ovito.xyz', fout2 )


            coords_sub_blocks.append(valuess)#cord_per_center)
            adj_mat_sub_blocks.append(adj_mat_per_center)
            rho_sub_blocks.append(rho)
            disp_blocks.append(disp_per_cluster)
            defects.append(neighbors_per_center.astype(int))
        return rho_sub_blocks,disp_blocks,adj_mat_sub_blocks,coords_sub_blocks,defects
		
    def PrintOvit(self,values,disp,adj_mat, fout, fout2 ):
    #         os.system('rm ovito.xyz')
            ndime = 3
            if 1:
                cordc = pd.DataFrame(values,columns='x y z'.split())
                diffusion_paths = disp
                nmode = int( diffusion_paths.shape[ 1 ] / ndime )
                for imode in range(nmode):
                    diffusion_path = diffusion_paths[:,imode*ndime:(imode+1)*ndime]
                    df = pd.DataFrame(np.c_[cordc,diffusion_path],columns = ' x y z ux uy uz'.split())
                    with open(fout,'a') as fp:
                        utl.PrintOvito(df, fp, 'itime=0', 
                                       attr_list='x y z ux uy uz'.split())
            l=1
            if 1:
                natom = cordc.shape[ 0 ]
                df = pd.DataFrame(np.c_[range(1,natom+1),np.ones(natom),np.ones(natom),cordc['x y z'.split()]],\
                                  columns = 'id junk0 junk1 x y z'.split())
                adj_matrix = adj_mat #kwargs[ 'adjacency' ]
                bonds = adj_matrix.nonzero()
                bonds += 1
                nbond = bonds.shape[ 0 ]
                with open(fout2,'w') as fout:
                    fout.write('# LAMMPS data file written by OVITO\n%s atoms\n%s bonds\n1 atom types\n1 bond types\n%s %s xlo xhi\n%s %s ylo yhi\n%s %s zlo zhi\n\n'\
                               %(natom,nbond,-0.5,0.5,-0.5,0.5,-0.5,0.5))
                    fout.write('Atoms # bond\n\n')
                    np.savetxt(fout,np.c_[df],'%d %d %d %e %e %e')

                    fout.write('\nBonds\n\n')
                    np.savetxt(fout,np.c_[np.arange(1,nbond+1), np.ones(nbond),bonds[:,0],bonds[:,1]],fmt='%d')

			
    def DataBuilder2nd( self ):
            num_snapshots = 100
            ntrain=1
            ndime = 2
            data = list(map(lambda x: self.Generate_subGraphs(ndime),range(num_snapshots)))
    #        data = list(map(lambda x: self.GenerateDate_2d(),range(num_snapshots)))
    #        data = list(map(lambda x: self.GenerateDate(),range(num_snapshots)))


            graphs = []

            snapshots     = range(num_snapshots)
            input_data    = [torch.from_numpy(np.log10(np.c_[j])).float() for i in snapshots for j in data[i][0]]  
    #        input_data    = [torch.from_numpy( np.c_[np.c_[j],np.c_[k]] ).float() for i in snapshots for j,k in zip(data[i][3],data[i][4])]  
    #        input_data    = [torch.from_numpy( np.c_[j] ).float() for i in snapshots for j in data[i][3]]  
    #        input_data    = [torch.from_numpy( np.log10(np.c_[j]) ).float() for i in snapshots for j in data[i][0]]  
            xy            = [torch.from_numpy( np.c_[j] ).float() for i in snapshots for j in data[i][3]]  
    #        pdb.set_trace()
            # Example target data (displacement vectors for each snapshot and each path)
            target_displacements = [torch.from_numpy( np.c_[j] ).float() for i in snapshots for j in data[i][1]]
    #        adj_matrices = [j for i in snapshots for j in data[i][2]]
            adj_matrices = [torch.eye(j.shape[0]) for i in snapshots for j in data[i][2]]
    #        input_data = [torch.from_numpy(np.c_[item.sum(axis=1),i]) for item, i in zip(adj_matrices,xy)]
            adj_matrices = torch.stack(adj_matrices) 
            for item in input_data:
                print('item.shape:',item.shape)


            # Augment the dataset to have order 100 single graphs
            augmented_input_data           = []
            augmented_target_displacements = []
            input_data_tensor              = torch.stack(input_data)
            ntrain_initial                 = input_data_tensor.shape[0]*input_data_tensor.shape[1]
            n_repeat                       = np.max([1,int(ntrain/ntrain_initial)])

            for _ in range(n_repeat):  # Repeat the augmentation process 10 times
                augmented_input, augmented_target = GraphNet.augment_data(input_data, target_displacements, self.noise_std)
                #
                augmented_input_data.extend(augmented_input)
                augmented_target_displacements.extend(augmented_target)

            input_data_tensor = torch.stack(augmented_input_data)
            mean              = input_data_tensor.mean(dim=(0, 1))
            std               = input_data_tensor.std(dim=(0, 1))
            assert torch.all( std > 0 ), 'std == 0!'
            standardized_input_data = [GraphNet.standardize_data(data, mean, std) for data in augmented_input_data]


            # Convert input data to tensors
            target_displacements_tensor = torch.stack(augmented_target_displacements)
            input_data_tensor           = torch.stack(standardized_input_data)
            input_xyz_tensor            = torch.stack(xy)



            # Concatenate nodes and edges for each graph
            graphs = []
            for i in range(len(input_data)):
                x = input_data_tensor[i]  # Node features
                edge_index = adj_matrices[i].nonzero().t()  # Edge indices
                y = target_displacements_tensor[i]  # Target displacements
                cords=input_xyz_tensor[i]
                # Create a Data object for each graph
                data = Data(x=x, edge_index=edge_index, y=y,pos=cords)
                graphs.append(data)

            # Create a single large graph by concatenating Data objects
            large_graph = torch_geometric.data.Batch.from_data_list(graphs)

            # Define batch size and create DataLoader
            # Create DataLoader for training dataset
            train_ratio = 0.5

            # Define batch sizes for training and test dataloaders
            batch_size = len(input_data)

            train_batch_size = int( np.max([1,int(batch_size * train_ratio)]) )

            # Create DataLoader for training dataset
            loader = DataLoader(large_graph, batch_size=train_batch_size, shuffle=False)

            # Accessing batches in the DataLoader
            loader_iter=iter(loader)
            self.dataset_train = next(loader_iter)
            if self.verbose:
                print('dataset_train:',self.dataset_train)
            self.dataset_test = next(loader_iter)
            if self.verbose:
                print('dataset_test:',self.dataset_test)

	 

    @staticmethod
    def compute_adjacency_matrices(input_data, rcut):
            adj_matrices = []

            for positions in input_data:
                num_atoms = positions.shape[0]
                adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)

                for i in range(num_atoms):
                    adj_matrix[i, i] = 1
                    for j in range(i + 1, num_atoms):
                        distance = torch.norm(positions[i] - positions[j])
                        if distance <= rcut:
                            adj_matrix[i, j] = 1
                            adj_matrix[j, i] = 1
                    assert adj_matrix[i,:].sum() > 0, 'dangling node : increase the cutoff!'
                adj_matrices.append(adj_matrix)

            #--- assert no 
            return adj_matrices


    def BuildNeighborList( self, indx, atom_indices,cutoff ):
    #         atom_indices = ' '.join(map(str,atom_indices))

    #         fp = self.dumpFiles[ indx ] #'%s/lammps_data.dat'%confParser['input files']['input_path']
            fout = self.neighlists[ indx ] #'neighbor_list.xyz'
    #         os.system('rm %s'%fout)
    #         lib_path = confParser['input files']['lib_path'].split()[0]
    #         #--- neighbor list
    #         os.system('ovitos %s/OvitosCna.py %s %s 1 6 %s %s'%(lib_path,fp,fout,cutoff,atom_indices))


            nl = lp.ReadDumpFile(fout)
            nl.GetCords()
            return nl.coord_atoms_broken[0]

    def GetIndxById( self, atom_ids, indx ):
            df              = pd.DataFrame(self.transition_paths[ indx ])
            df['indices']   = range(df.shape[0])
            atom_indices    = utl.FilterDataFrame(df,key='id',val=atom_ids)['indices']
            return np.c_[atom_indices].flatten()

    def compute_adjacency_matrices2nd(self,input_data, rcut):
            adj_matrices       = []
            edge_attrs         = []
            for indx, positions in enumerate( input_data ):
                num_atoms      = positions.shape[0]
                adj_matrix     = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
                nl             = self.BuildNeighborList(indx,range(len(positions)),rcut) #--- neighbor list
                #--- add "index" columns
                nl['index_i']=self.GetIndxById( np.c_[nl.id].flatten(), indx )
                nl['index_j']=self.GetIndxById( np.c_[nl.J].flatten(), indx )
                groups         = nl.groupby(by='id').groups
                atom_i_ids     = list(groups.keys())
                atom_i_indices = self.GetIndxById( atom_i_ids, indx )
                for i, atom_id in zip(atom_i_indices,atom_i_ids):
    #                adj_matrix[i, i] = 1
                    atom_j_ids       = nl.iloc[groups[ atom_id ]].J
                    atom_j_indices   = self.GetIndxById( atom_j_ids, indx )
                    for j, jatom_id in zip(atom_j_indices, atom_j_ids ): #[ atom_j_indices > i ]:
                        if j <= i :
                            continue
                        filtr = np.all([nl.id==atom_id,nl.J==jatom_id],axis=0)
                        edge_features = nl.iloc[ filtr ][ ''.split() ]
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1
                    assert adj_matrix[i,:].sum() > 0, 'dangling node : increase the cutoff!'
    #            pdb.set_trace()
                #--- edge attributes
                keys = 'DX  DY  DZ  PBC_SHIFT_X PBC_SHIFT_Y PBC_SHIFT_Z'.split()
                indices = adj_matrix.nonzero().numpy()
                nl_reindexed = nl.set_index(['index_i','index_j'],drop=False)
                edge_attr = list(map(lambda x: list(nl_reindexed[keys].loc[tuple(x)]),indices))

    #            pdb.set_trace()
                edge_attrs.append( torch.Tensor( edge_attr ) )
                adj_matrices.append( adj_matrix )

            #--- assert no 
            return adj_matrices, edge_attrs


    @staticmethod
    def augment_data(input_data, target_displacements, noise_std):
            augmented_input_data = []
            augmented_target_displacements = []

            for data, target in zip(input_data, target_displacements):
                # Add Gaussian noise to input data
                noisy_data = data + torch.randn_like(data) * noise_std
                augmented_input_data.append(noisy_data)

                # Add Gaussian noise to target displacements
                noisy_target = target + torch.randn_like(target) * noise_std
                augmented_target_displacements.append(noisy_target)

            return augmented_input_data, augmented_target_displacements

    @staticmethod
    def standardize_data(data, mean, std):
            return (data - mean) / std

    @staticmethod
    def save_best_model(model, optimizer, epoch, loss, best_loss, path):
            """Save the best model."""
            if loss < best_loss:
    #             state = {
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': loss,
    #             }
    #             torch.save(state, path)
                torch.save(model, path)
                print(f'Saved the best model with loss: {loss:4.3e}')
                return loss
            else:
                return best_loss

        # Save checkpoint during training
    @staticmethod
    def save_checkpoint(model, filename):
    #         checkpoint = {
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #         }
    #        torch.save(checkpoint, filename)
            torch.save(model.state_dict(), filename)

        # Save checkpoint during training
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, filename):
            checkpoint = {
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss': loss,
            }
            torch.save(checkpoint, filename)

    def PrintOvito(self, **kwargs ):
            get_ipython().system("rm 'ovito.xyz'")
            for indx, item in enumerate( self.transition_paths ):
    #             pdb.set_trace()
                cordc = pd.DataFrame(item)['id x y z'.split()]
                diffusion_paths = np.array(item['diffusion_paths'])
                nmode = int( diffusion_paths.shape[ 1 ] / ndime )
                for imode in range(nmode):
                    diffusion_path = diffusion_paths[:,imode*ndime:(imode+1)*ndime]
                    df = pd.DataFrame(np.c_[cordc,diffusion_path],columns = 'id x y z ux uy uz'.split())
                    with open('ovito.xyz','a') as fp:
                        utl.PrintOvito(df, fp, 'itime=%s'%indx, 
                                       attr_list='id x y z ux uy uz'.split())

            l=33.64
            if 'adjacency' in kwargs:
                item = self.transition_paths[ 0 ]
                cordc = pd.DataFrame(item)['id x y z'.split()]
                natom = cordc.shape[ 0 ]
                df = pd.DataFrame(np.c_[cordc.id,np.ones(natom),np.ones(natom),cordc['x y z'.split()]],\
                                  columns = 'id junk0 junk1 x y z'.split())
                adj_matrix = kwargs[ 'adjacency' ]
                bonds = adj_matrix.nonzero()
                bonds += 1
                nbond = bonds.shape[ 0 ]
                with open('lammps.data','w') as fout:
                    fout.write('# LAMMPS data file written by OVITO\n%s atoms\n%s bonds\n1 atom types\n1 bond types\n%s %s xlo xhi\n%s %s ylo yhi\n%s %s zlo zhi\n\n'\
                               %(natom,nbond,0,l,0,l,0,l))
                    fout.write('Atoms # bond\n\n')
                    np.savetxt(fout,np.c_[df],'%d %d %d %e %e %e')

                    fout.write('\nBonds\n\n')
                    np.savetxt(fout,np.c_[np.arange(1,nbond+1), np.ones(nbond),bonds[:,0],bonds[:,1]],fmt='%d')


    # ### example 1d

    # In[30]:


def main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn = GraphNet(
                         c_in       = 256,
                         c_hidden   = [128,32,8],
                         c_out      = 3,
                         num_layers = 4,
                         num_epochs = 30000,
                         noise_std  = 0.001,
                         lr         = 0.0001,
            edge_dim=1,
            layer_name = "GCN",
                         verbose    = True 
                    ).to(device)  # Move model to GPU

    #     gnn.Parse( path  = confParser['gnn']['input_path'],
    #                  nruns = eval(confParser['gnn']['nruns']))

        #--- build dataset based on the input catalogs
        gnn.DataBuilder2nd()

        # Define optimizer and loss function
        optimizer = optim.Adam(gnn.parameters(), lr=gnn.lr)
        criterion = nn.MSELoss()

        epoch0 = 0
        best_loss = np.inf

        # training loop
        training_loss_hist   = []
        validation_loss_hist = []
        get_ipython().system('mkdir best_model')
        for epoch in range( epoch0, epoch0+gnn.num_epochs ):
            optimizer.zero_grad()
            predicted_displacements = gnn(gnn.dataset_train.x.to(device), 
                                          gnn.dataset_train.edge_index.to(device))
            training_loss              = criterion(predicted_displacements, gnn.dataset_train.y.to(device))
            training_loss.backward()
            optimizer.step()
            training_loss_hist += [training_loss.detach().cpu().numpy()]  # Move loss back to CPU

            #--- validation loss
            gnn.eval()
            with torch.no_grad():  # Disable gradient calculation
                    predicted_displacements = gnn(gnn.dataset_test.x.to(device), gnn.dataset_test.edge_index.to(device))
                    validation_loss         = criterion(predicted_displacements, gnn.dataset_test.y.to(device))

                    validation_loss_hist += [validation_loss.cpu().numpy()]  # Move loss back to CPU

            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Training Loss: {training_loss.item():4.3e}, Validation Loss: {validation_loss.item():4.3e}')

                # Update best_loss if validation loss improves and save the model
    #             best_loss = GraphNet.save_best_model(gnn, optimizer, 
    #                                         epoch, training_loss.detach().cpu().numpy(), best_loss, 
    #                                         'best_model/best_model.pth')

        #--- plot loss vs epoch
        get_ipython().system('mkdir png')
        ax = utl.PltErr(None,None,Plot=False)
        utl.PltErr(range(gnn.num_epochs),training_loss_hist,
                    attrs={'fmt':'-','color':'C0'},
                    ax=ax,Plot=False
               )
        utl.PltErr(range(gnn.num_epochs),validation_loss_hist,
                    attrs={'fmt':'-','color':'red'},
                   xscale='log',yscale='log',
                    title='png/loss.png',
                    Plot=False,
                    ax=ax
               )
        return gnn.dataset_train, gnn.dataset_test, gnn


data_train, data_test, model = main()
    # 


    # In[ ]:
def make_prediction(model, data, title):
        u_pred = model(data.x, data.edge_index) #, data.edge_attr)        
        u_pred = u_pred.cpu().detach().numpy()
        u_act  = data.y.cpu()
        ndime  = u_act.shape[ 1 ]

        #--- plot prediction vs. actual

        PrintOvito(data,u_pred,'%s/u_pred.xyz'%title)
        PrintOvito(data,u_act, '%s/u_act.xyz'%title)

def PrintOvito( data, disps, fout ):
        ndime = 3
        os.system('rm %s'%fout)
        box        = lp.Box(BoxBounds=np.array([[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]),\
                            AddMissing=np.array([0,0,0]))

        atom_indx_init = data.ptr[ 0 ]
        for indx, _ in enumerate( data.ptr ):
            if indx == 0:
                continue
            atom_indx_fin = data.ptr[ indx ]
            atom_ids      = np.arange(atom_indx_init.cpu(),atom_indx_fin.cpu())+1
            types         = np.ones(atom_ids.shape[0])
            xyz   = data.pos[ atom_indx_init : atom_indx_fin] 
            tmp   = xyz.cpu()# * std + mean
            cordc = pd.DataFrame( tmp[:,:ndime], columns='x y z'.split())
            disp  = disps[ atom_indx_init : atom_indx_fin, : ]

            nmode = int( disp.shape[ 1 ] / ndime )
            for imode in range(nmode):
                diffusion_path = disp[:,imode*ndime:(imode+1)*ndime]
                df = pd.DataFrame(np.c_[atom_ids,types,cordc,diffusion_path],\
                                  columns = 'id type x y z  DisplacementX DisplacementY DisplacementZ'.split())
                atom  = lp.Atoms(**df.to_dict(orient='series'))
    #             with open(fout,'a') as fp:
    #                 utl.PrintOvito(df, fp, 'irun=%s,imode=%s'%(indx-1,imode), 
    #                                attr_list='x y z ux uy uz'.split())
                wd   = lp.WriteDumpFile(atom, box)
                with open(fout,'a') as fp:
                    wd.Write(fp, itime=0, 
                             attrs='id type x y z DisplacementX DisplacementY DisplacementZ'.split(), 
                             fmt='%d %d %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')

            atom_indx_init = atom_indx_fin


def main(data_train, data_test,model):
    # Example usage
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
    #    model = torch.load('best_model/best_model.pth').to(device)
        os.system('rm -r png;mkdir -p png/train;mkdir -p png/test')
        make_prediction(model, data_train.to(device), title='png/train')
        make_prediction(model, data_test.to(device), title='png/test')

main(data_train, data_test,model)









