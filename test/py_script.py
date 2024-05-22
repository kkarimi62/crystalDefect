


class DataSet:

    def __init__(self,confParser,verbose=True):
        self.noise_std            = eval(confParser['NeuralNets']['noise_std'])
        self.verbose              = verbose
        self.cutoff               = 3.0

    def Parse(self,path,nruns, **kwargs):
        '''
        Parse dataset
        '''
        #
        self.Catalogs         = {}
        self.transition_paths = []
        self.descriptors      = []
        self.dumpFiles        = []
        self.neighlists       = []
        #
        if self.verbose:
            print('parsing %s'%path)
        rwjs = utl.ReadWriteJson()

        #--- dirs and files to be parsed
        file_dirs = ['saved_output/descriptorsEveryAtom.json',
             'saved_output/transition_paths.json',
             'neighList/neigh_list.xyz',
             'dumpFile/dump.xyz',
             'saved_output/catalog.txt'
            ]
        for irun in range(nruns):
            if not self.fileExists(path,irun,file_dirs):
                continue            
            self.descriptors.extend( rwjs.Read('%s/Run%s/saved_output/descriptorsEveryAtom.json'%(path,irun)) )
            self.transition_paths.extend( rwjs.Read('%s/Run%s/saved_output/transition_paths.json'%(path,irun)) )
            self.neighlists.append( '%s/Run%s/neighList/neigh_list.xyz'%(path,irun))
            self.dumpFiles.append(  '%s/Run%s/dumpFile/dump.xyz'%(path,irun))
            os.system('ln -s %s/Run%s/dumpFile/dump.xyz ./dump.%s.xyz'%(path,irun,irun))
            self.Catalogs[irun]     = pd.read_csv('%s/Run%s/saved_output/catalog.txt'%(path,irun))
        #        
        self.nruns     = list(self.Catalogs.keys())
        self.nruns.sort()

    def fileExists(self,path,irun,file_dirs):
        '''
        return True if passed dirs exist
        '''
        file_exist = []
        for myfile in file_dirs:
            file_exist.append( os.path.isfile('%s/Run%s/%s'%(path,irun,myfile)) )
        return np.all(np.array(file_exist))

    def Process( self, Ovito_Output = False, train_ratio = 0.8 ):
        '''
        Build dataloader in pytorch
        '''
        #
        ntrain        = 1
        num_snapshots = len( self.transition_paths ) #--- total no. of transition paths
        snapshots     = range(num_snapshots)
        #--- include the center atom within each cluster
#        filtrs        = list(map(lambda x: np.array(np.ones(len(self.transition_paths[x]['center_atom_index']))).flatten().astype(bool), snapshots))
        filtrs = list(map(lambda x: np.array(self.transition_paths[x]['center_atom_index']).flatten().astype(bool), snapshots))
        #--- nodal cords
        input_xyz     = [torch.from_numpy( np.c_[np.c_[self.transition_paths[ i ]['x'],\
                                                       self.transition_paths[ i ]['y'],\
                                                       self.transition_paths[ i ]['z']][filtrs[i]]] ).float() for i in snapshots]
        #--- nodal descriptors
        input_data    = [torch.from_numpy( np.c_[np.log10(np.array(self.transition_paths[ i ]['descriptors']))][filtrs[i]] ).float() for i in      snapshots]
        #--- atom indices
        input_atom_indx  = [torch.from_numpy( np.c_[self.transition_paths[ i ]['atom_indx']][filtrs[i]] ).int() for i in      snapshots]
        
        #--- target data (displacement vectors for each path)
        displacement_vecs    = [torch.from_numpy(np.array(self.transition_paths[ i ]['diffusion_paths'])[:,:3]).float() for i in snapshots]
        target_displacements = [torch.from_numpy(np.array(self.transition_paths[ i ]['multi_hot_encoded_diffusion_paths'])[filtrs[i]]).float() for i in snapshots]

        #--- add gaussian noise
        augmented_input_data           = []
        augmented_input_xyz            = []
        augmented_target_displacements = []
        augmented_displacement_vecs    = []
        n_repeat                       = 1 #np.max([1,int(ntrain/ntrain_initial)])
        #
        for _ in range(n_repeat):  # Repeat the augmentation process 10 times
            augmented_input  = DataSet.augment_data( input_data,           self.noise_std )
            augmented_target = DataSet.augment_data( target_displacements, 0.0) #self.noise_std )
            augmented_vecs   = DataSet.augment_data( displacement_vecs,  0.0) #self.noise_std )
            augmented_xyz    = DataSet.augment_data( input_xyz,            self.noise_std )
            #
            augmented_input_data.extend(augmented_input)
            augmented_input_xyz.extend(augmented_xyz)
            augmented_target_displacements.extend(augmented_target)
            augmented_displacement_vecs.extend(augmented_vecs)

        #--- adjacency matrix
        adj_matrices      = self.compute_adjacency_matrices(augmented_input_xyz, rcut=self.cutoff)
        
        #--- verify adj matrix?
        if Ovito_Output:
            self.PrintOvito(adjacency = adj_matrices, input_data=input_data)

        #--- Concatenate input data along a new dimension to form a single tensor
        input_data = np.vstack(augmented_input_data) 

        #--- Standardize the augmented input data
        mean              = input_data.mean(axis=0)
        std               = input_data.std(axis=0)
        assert np.all( std > 0 ), 'std == 0!'
        standardized_input_data = [DataSet.standardize_data(data, mean, std) for data in augmented_input_data]
        
        #--- Standardize edge attributes
#         mean              = edge_attrs.mean(dim=(0, 1))
#         std               = edge_attrs.std(dim=(0, 1))
#         standardized_edge_attrs = [GraphNet.standardize_data(data, mean, std) for data in edge_attrs]


        #--- Convert input data to tensors
        target_displacements_tensor = augmented_target_displacements
        target_disps_tensor         = augmented_displacement_vecs 
        input_data_tensor           = standardized_input_data
        input_xyz_tensor            = augmented_input_xyz
#         edge_attrs_tensor           = torch.stack(standardized_edge_attrs)

        #--- Concatenate nodes and edges for each graph
        graphs = []
        for i in range(len(input_data_tensor)):
            x             = input_data_tensor[i]  # Node features
            cords         = input_xyz_tensor[i]  # Node features
            edge_index    = adj_matrices[i].nonzero().t()  # Edge indices
#             edge_features = edge_attrs_tensor[ i ][ :, : self.edge_dim ]
            atom_indx     = input_atom_indx[ i ]
            y             = target_displacements_tensor[i]  # Target displacements
            y_atom_wise   = target_disps_tensor[i]  # Target displacements

            # Create a Data object for each graph
            data = Data(x=x, edge_index=edge_index, y=y, pos=cords, atom_indx=atom_indx, y_atom_wise=y_atom_wise) #edge_attr = edge_features)
            graphs.append(data)
        
        #--- Create a single large graph by concatenating Data objects
        self.large_graph = torch_geometric.data.Batch.from_data_list(graphs)

        #--- Define batch size and create DataLoader
        batch_size = len(input_data_tensor)
        
        #--- Define batch sizes for  training and test dataloaders
        train_batch_size = int( np.max([1,int(batch_size * train_ratio)]) )
        test_batch_size  =  batch_size - train_batch_size
        
        #--- Create DataLoader for training dataset
        loader           = DataLoader(self.large_graph, batch_size=train_batch_size, shuffle=False)

        #--- Accessing batches in the DataLoader
        loader_iter      = iter(loader)
        self.dataset_train = next(loader_iter)
        if self.verbose:
            print('dataset_train:',self.dataset_train)
        self.dataset_test = self.dataset_train
        if test_batch_size > 0:
            self.dataset_test = next(loader_iter)
            if self.verbose:
                print('dataset_test:',self.dataset_test)

    def DataBuilderForClassifier( self, Ovito_Output = False ):
        
        ntrain        = 1 #self.ntrain
        num_snapshots = len( self.descriptors )
        snapshots     = range(num_snapshots)
        #--- nodal cords
        input_xyz     = [torch.from_numpy( np.c_[np.c_[self.descriptors[ i ]['x'],\
                                                       self.descriptors[ i ]['y'],\
                                                       self.descriptors[ i ]['z']]] ).float() for i in snapshots]
        #--- nodal descriptors
#         input_data   = [torch.from_numpy( np.c_[np.log10(np.array(self.descriptors[ i ]['descriptors'])),\
#                                                  ] ).float() for i in      snapshots]
        input_data    = [torch.from_numpy( np.c_[self.descriptors[ i ]['x'],\
                                         self.descriptors[ i ]['y'],\
                                         self.descriptors[ i ]['z'],\
                                         np.array(self.descriptors[ i ]['descriptors_acsf'])]).float() for i in snapshots]

        #--- target data
        labels        = [torch.from_numpy(np.array(self.descriptors[ i ]['isNonCrystalline']).flatten()).long() for i in snapshots]
        
        #--- Augment the dataset 
        augmented_input_data           = []
        augmented_input_xyz            = []
        augmented_labels               = []
        n_repeat                       = 1 #np.max([1,int(ntrain/ntrain_initial)])

        for _ in range(n_repeat):  # Repeat the augmentation process 10 times
            augmented_input  = DataSet.augment_data(input_data, self.noise_std)
            augmented_target = labels #GraphNet.augment_data(labels, 0)
            augmented_xyz    = DataSet.augment_data(input_xyz, self.noise_std)
            #
            augmented_input_data.extend(augmented_input)
            augmented_input_xyz.extend(augmented_xyz)
            augmented_labels.extend(augmented_target)
           
        #--- adj. matrix
        adj_matrices_attrs      = self.compute_adjacency_matrices2nd(self.descriptors, rcut=self.cutoff)
        adj_matrices            = adj_matrices_attrs[ 0 ]
        edge_attrs              = adj_matrices_attrs[ 1 ]
        
        #--- verify adj matrix??
        if Ovito_Output:
            self.PrintOvito(adjacency = adj_matrices, input_data=input_data)

        #--- Concatenate input data along a new dimension to form a single tensor
        input_data_tensor = np.vstack(augmented_input_data)

        #--- Standardize the augmented input data
        mean              = input_data_tensor.mean(axis=0)#dim=(0, 1))
        std               = input_data_tensor.std(axis=0)#dim=(0, 1))
        assert np.all( std > 0 ), 'std == 0!'
        standardized_input_data = [DataSet.standardize_data(data, mean, std) for data in augmented_input_data]
        
        #--- Standardize edge attributes
#         mean              = edge_attrs.mean(dim=(0, 1))
#         std               = edge_attrs.std(dim=(0, 1))
#         standardized_edge_attrs = [GraphNet.standardize_data(data, mean, std) for data in edge_attrs]


        #--- Convert input data to tensors
        labels_tensor               = augmented_labels
        input_data_tensor           = standardized_input_data
        input_xyz_tensor            = augmented_input_xyz
#         edge_attrs_tensor           = torch.stack(standardized_edge_attrs)



        #--- Concatenate nodes and edges for each graph
        graphs = []
        for i in range(len(input_data_tensor)):
            x             = input_data_tensor[i]  # Node features
            cords         = input_xyz_tensor[i]  # Node features
            edge_index    = adj_matrices[i].nonzero().t()  # Edge indices
#             edge_features = edge_attrs_tensor[ i ][ :, : self.edge_dim ]
            y             = labels_tensor[i]  # Target displacements

            #--- Create a Data object for each graph
            data = Data(x=x, edge_index=edge_index, y=y, pos=cords) #edge_attr = edge_features)
            graphs.append(data)

        np.random.shuffle(graphs)

        #--- Define batch size and create DataLoader
        batch_size  = len(input_data)
        
        # Define the split ratio (e.g., 80% for training, 20% for testing)
        train_ratio = 0.8

        #--- Define batch sizes for training and test dataloaders
        train_batch_size = int( np.max([1,int(batch_size * train_ratio)]) )
        test_batch_size  = batch_size - train_batch_size
        assert test_batch_size > 0, 'test_batch_size = %s'%test_batch_size

        large_graph_train = torch_geometric.data.Batch.from_data_list(graphs[:train_batch_size])
        large_graph_test  = torch_geometric.data.Batch.from_data_list(graphs[train_batch_size:])

        #--- Create DataLoader for training dataset
        self.train_dataloaders = DataLoader(large_graph_train, batch_size=train_batch_size, shuffle=False)
        self.test_dataloaders  = DataLoader(large_graph_test, batch_size=test_batch_size, shuffle=False)

    def compute_adjacency_matrices(self,input_data, rcut):
        adj_matrices = []
        for indx, positions in enumerate(input_data):
            #
            num_atoms = positions.shape[0]
            adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
            #
            for i in range(num_atoms):
                adj_matrix[i, i] = 1
                for j in range(i + 1, num_atoms):
                    drij = abs(positions[i] - positions[j])
#                    assert drij[0] <= 0.5 * lx and drij[1] <= 0.5 * ly and drij[2] <= 0.5 * lz, 'cutoff > 0.5 L!'
                    distance = torch.norm(drij)
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

    def GetIndxById( self, df, atom_ids ):
        df['indices']   = range(df.shape[0])
        atom_indices    = utl.FilterDataFrame(df,key='id',val=atom_ids)['indices']
        return np.c_[atom_indices].flatten()
            
    def compute_adjacency_matrices2nd(self,input_data, rcut):
        adj_matrices       = []
        edge_attrs         = []
        for indx, positions in enumerate( input_data ):
            df             = pd.DataFrame(positions)
            num_atoms      = df.shape[0]
            adj_matrix     = torch.zeros((num_atoms, num_atoms), dtype=torch.float)
            nl             = self.BuildNeighborList(indx,range(num_atoms),rcut) #--- neighbor list
            #--- add "index" columns
            nl['index_i']  = self.GetIndxById( df, np.c_[nl.id].flatten() )
            nl['index_j']  = self.GetIndxById( df, np.c_[nl.J].flatten() )
            adj_matrix[nl['index_i'],nl['index_j']] = 1
            #--- edge attributes
#             keys = 'DX  DY  DZ  PBC_SHIFT_X PBC_SHIFT_Y PBC_SHIFT_Z'.split()
#             indices = adj_matrix.nonzero().numpy()
#             nl_reindexed = nl.set_index(['index_i','index_j'],drop=False)
#             edge_attr = list(map(lambda x: list(nl_reindexed[keys].loc[tuple(x)]),indices))
#             edge_attrs.append( torch.Tensor( edge_attr ) )
            adj_matrices.append( adj_matrix )

        #--- assert no 
        return adj_matrices, edge_attrs


    @staticmethod
    def augment_data(input_data, noise_std):
        augmented_input_data = []
        
        for data in input_data:
            # Add Gaussian noise to input data
            noisy_data = data + np.random.randn(*data.shape) * noise_std
            augmented_input_data.append(noisy_data)
            
            
        return augmented_input_data


    @staticmethod
    def standardize_data(data, mean, std):
        return (data - mean) / std
    

class MyConvNetModel( tf.keras.Model ):
    def __init__(self,reshape = (3,3,3), filters=16, num_hidden_layers=1, **kwargs):
        super().__init__()
        self.myLayers = []
        self.myLayers.append(tf.keras.layers.Reshape(reshape))
        self.myLayers.append( tf.keras.layers.Conv3D( filters=filters,**kwargs))
        filters       *=  2
        for i in range( num_hidden_layers ):
            self.myLayers.append(tf.keras.layers.AveragePooling3D( pool_size = 2 ))
            self.myLayers.append(tf.keras.layers.Conv3D( filters=filters,**kwargs ))
            filters *= 2
        self.myLayers.append(tf.keras.layers.Flatten())
        
    def call( self, inputs ):
        x = inputs
        for layer in self.myLayers:
            x = layer( x )
        return x

class MyDenseNetModel( tf.keras.Model ):
    def __init__(self,num_hidden_layers=1,units = 10,  **kwargs):
        super().__init__()
        self.myLayers = []
        for i in range( num_hidden_layers ):
            self.myLayers.append( tf.keras.layers.Dense( units,**kwargs))
        
    def call( self, inputs ):
        x = inputs
        for layer in self.myLayers:
            x = layer( x )
        return x
    
class MyConvNetClassifier( tf.keras.Model ):
    def __init__(self, cout = 10, **kwargs):
        super().__init__()
        self.convnet =  MyConvNetModel( **kwargs )
        self.classifier = tf.keras.layers.Dense( cout, activation='sigmoid' )
        
    def call( self, inputs ):
        x = self.convnet( inputs )
        return self.classifier( x ) 

class MyDenseNetClassifier( tf.keras.Model ):
    def __init__(self, cout = 1, **kwargs):
        super().__init__()
        self.convnet =  MyDenseNetModel( **kwargs )
        self.classifier = tf.keras.layers.Dense( cout, activation='sigmoid' )
        
    def call( self, inputs ):
        x = self.convnet( inputs )
        return self.classifier( x ) 

def TrainingLoop(model, data_train, data_test, **kwargs):
        optimizer = tf.keras.optimizers.Adam( learning_rate = kwargs['learning_rate'] )
        model.compile( optimizer =  optimizer,
                       loss      =  kwargs['loss'],
                       metrics   =  ["mse"]
                     )

        history = model.fit( data_train.x.numpy(), data_train.y.numpy(), 
                   validation_data      = ( data_test.x.numpy(), data_test.y.numpy() ),
                    epochs              = kwargs['epochs'], 
                    verbose             = True,
                     batch_size     = 128,
                 )
        #--- plot vall loss

        model.save(kwargs['checkpoint_file'])
#         return model


class ModelValidation:
    def __init__(self):
         pass   
    def SetModel(self, model):
         self.model=model

    def ConfusionMatrix(self, data, fout):
        y_pred = ( self.model.predict(data.x.numpy()) > 0.5 ).astype(int)
        cm = confusion_matrix(data.y, y_pred,
                         labels=[0,1]
                        )
        np.savetxt(fout,np.c_[cm])

    def GetDispsFromBinaryMaps( self, binaryMap, xv,yv,zv,ev,nbinx, nbiny,nbinz,nbine ):
        binaryMapReshaped = binaryMap.reshape((nbinx, nbiny, nbinz, nbine ))
        filtr = binaryMapReshaped == 1
        return np.c_[yv[filtr],xv[filtr],zv[filtr],ev[filtr]]

    def TransitionPaths(self, data, title,umax=2.0,du=0.2,emax=2.0,de=0.2):
        binary_maps_pred = ( self.model(data.x).numpy() > 0.5 ).astype(int)    
        binary_maps_true  = data.y

        xlin = np.arange(-umax,umax+du,du)
        ylin = np.arange(-umax,umax+du,du)
        zlin = np.arange(-umax,umax+du,du)
        elin = np.arange(0,emax+de,de)
        nbinx = len(xlin)-1
        nbiny = len(ylin)-1
        nbinz = len(zlin)-1
        nbine = len(elin)-1
        bins = (xlin, ylin, zlin, elin )
        xv, yv, zv, ev = np.meshgrid( bins[1][:-1], bins[0][:-1], bins[2][:-1], bins[3][:-1] )

        u_pred = np.concatenate([list(map(lambda x: self.GetDispsFromBinaryMaps( x,xv,yv,zv,ev,nbinx, nbiny,nbinz,nbine ) , binary_maps_pred ))])
        u_true = np.concatenate([list(map(lambda x: self.GetDispsFromBinaryMaps( x,xv,yv,zv,ev,nbinx, nbiny,nbinz,nbine ) , binary_maps_true ))])

        #--- plot e
        self.PrintOvito(data,u_pred,'%s/u_pred.xyz'%title)
        self.PrintOvito(data,u_true, '%s/u_act.xyz'%title)
   
 
    def PrintOvito( self,data, disps, fout ):
        os.system('rm %s'%fout)
        ndime = 3
        box        = lp.Box(BoxBounds=np.array([[0,10.62],[0,10.62],[0,10.62]]),\
                            AddMissing=np.array([0,0,0]))

        atom_indx_init = data.ptr[ 0 ]
        for indx, _ in enumerate( data.ptr ):
            if indx == 0:
                continue
            atom_indx_fin = data.ptr[ indx ]
            atom_ids      = np.arange(atom_indx_init.cpu(),atom_indx_fin.cpu())+1
            natoms        = atom_ids.shape[ 0 ]
            types         = np.ones( natoms )
            xyz           = data.pos[ atom_indx_init : atom_indx_fin] 
            tmp           = xyz.cpu()# * std + mean
            cordc         = pd.DataFrame( tmp[:,:ndime], columns='x y z'.split())
            disp          = disps[ atom_indx_init ][:,:ndime] # : atom_indx_fin, : ]
            for path in disp:
                df             = pd.DataFrame(np.c_[atom_ids,types,cordc,path.reshape((1,3))],\
                                  columns = 'id type x y z DisplacementX DisplacementY DisplacementZ'.split())
                atom           = lp.Atoms(**df.to_dict(orient='series'))
                wd             = lp.WriteDumpFile(atom, box)
                with open(fout,'a') as fp:
                    wd.Write(fp, itime=0, 
                             attrs='id type x y z DisplacementX DisplacementY DisplacementZ'.split(), 
                             fmt='%d %d %4.3e %4.3e %4.3e %4.3e %4.3e %4.3e')

            atom_indx_init = atom_indx_fin


def main():
    ds       = DataSet( confParser )
    ds.Parse( path  = confParser['neural net']['input_path'],
              nruns = eval(confParser['neural net regression']['nruns']))

    #--- write call backs
    
    #--- identification of defects
    #--- load data
    ds.DataBuilderForClassifier()
    
    #--- build model
    myModel  = MyDenseNetClassifier( cout = 1,
                                units=64, num_hidden_layers = 3,
                                activation='relu')
    
    #--- training
    TrainingLoop( myModel, ds.train_dataloaders.dataset, ds.test_dataloaders.dataset,
                             learning_rate=1e-4,epochs=10, loss='binary_crossentropy',
                              checkpoint_file = 'best_model_defect_classification/model.tf'
                      )
    
    #--- validation
    model    = tf.keras.models.load_model('best_model_defect_classification/model.tf')
    mv       = ModelValidation()
    mv.SetModel(model)
    mv.ConfusionMatrix(ds.train_dataloaders.dataset,'defect_classify/cm_train.txt') 
    mv.ConfusionMatrix(ds.test_dataloaders.dataset,'defect_classify/cm_test.txt') 

    #--- predict reaction paths/energies
    #--- load data
    ds.Process()
    
    #--- create model
    myModel  = MyConvNetClassifier( cout = ds.dataset_train.y.shape[1],
                                    reshape=(10,10,10,1),
                                filters=16, num_hidden_layers = 3,
                                kernel_size=(3,3,3), activation='relu', padding='same',
                              )
    
    #--- training
    TrainingLoop(myModel, ds.dataset_train, ds.dataset_test,
                 learning_rate=1e-4,loss='binary_crossentropy',epochs=4
                  checkpoint_file = 'best_model_transition_paths/model.tf'
                 )
    
    #--- inference
    model    = tf.keras.models.load_model('best_model_transition_paths/model.tf')
    os.system('mkdir -p predictions/train;mkdir -p predictions/test')
    os.system('rm predictions/train/u_atoms.xyz')
#    PrintOvito(ds.dataset_train, ds.dataset_train.y_atom_wise, 'predictions/train/u_atoms.xyz')
    mv       = ModelValidation()
    mv.SetModel( model )
    mv.TransitionPaths( ds.dataset_train, title = 'predictions/train' )
    mv.TransitionPaths( ds.dataset_test,  title = 'predictions/test'  )



main()



