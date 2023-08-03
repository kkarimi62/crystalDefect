import pandas as pd
import pdb
import sys
import numpy as np

def isInside(atoms, box, smBox):
    #
    wrap = lp.Wrap( atoms, box )
    wrap.WrapCoord()
    wrap.Set( atoms ) #--- atoms: set wrapped xyz   
    #
    mapp = lp.Map( atoms, box ) 
    mapp.ChangeBasis()
    mapp.Set( atoms ) #--- atoms: add mapped xyz
    #--- expand
    copy = lp.Copy( atoms, box )    
    copy.Expand( epsilon = 1.0)
    xatoms0 = copy.Get()
    #--- filter atoms
    cptmp = lp.Copy(xatoms0, smBox) #--- important: must be reference frame!!
#    pdb.set_trace()
    #
    a=smBox.CellVector[0,0]
    l=np.diag(box.CellVector).min()
    indices = cptmp.isInsideExpanded( np.c_[xatoms0.xm,xatoms0.ym,xatoms0.zm], 
                                      epsilon = 4.0*a/l )
    
    xatoms0.id = list(map(int,xatoms0.id))
    xatoms0.type = list(map(int,xatoms0.type))
    ids = list(pd.DataFrame(xatoms0.__dict__)['id'][indices]) #--- inside
    assert len(set(ids))==len(ids),'box includes atoms with their images: decrease the size!'
    

    ido = list(set(atoms.id)-set(ids)) #--- outside
    
    return DataFrameGet(pd.DataFrame(atoms.__dict__),
                        key='id',val=ids)
    
#     return DataFrameSet(pd.DataFrame(atoms.__dict__),
#             key='id', val=ido,
#             setkey='type', setval=np.max(atoms.type)+1,      
#             )
    

    
def FreezeAtom(atoms, box, 
               center,
               **kwargs ):
    
    if 'size' in kwargs:
        size = kwargs['size']
        smBox = lp.Box(CellOrigin = center-size*np.array([1.0,1.0,1.0])/3.0**0.5,
                    CellVector=2*size*np.array([[1,0,0],
                                                [0,1,0],
                                                [0,0,1]]) )
    elif 'volume' in kwargs:
        smBox = kwargs['volume']
    else:
        print('provide either size or volume!')
        return None

#    atomc = lp.Atoms(**atoms.__dict__.copy())
    #--- get atoms inside the subvolume
    wrap = lp.Wrap( atoms, smBox )
    filtr = wrap.isInside()
    
    return lp.Atoms( **pd.DataFrame(atoms.__dict__)[filtr].to_dict(orient='list') )

#    df = isInside(atomc, box, smBox )
#    return lp.Atoms( **df.to_dict(orient='list') ) 


def CenterAtoms( atoms, box,
                 atomd, boxd,
                idcent = 0,
                CenterAtZero = True
               ):
    
    wrap = lp.Wrap(atoms,box)
    assert np.all(wrap.isInside()), 'aotms outside original box!'

    
#    idout = np.sum(atoms.id)-np.sum(df['id']) #--- atom id taken out
    
    #--- center
    rcent = np.array(pd.DataFrame(atoms.__dict__)[atoms.id==idout][['x','y','z']].iloc[0].tolist())
    
    #--- dimensionless
    wrap = lp.Wrap(atoms,box)
    wrap.GetDimensionlessCords()
    rdim = wrap.beta[atoms.id==idout].flatten()
        
    #--- wrap
    wrap = lp.Wrap(atomd,boxd)
    wrap.GetDimensionlessCords()

    #--- center dimensionless xyz
    for idime in range(3):
#        pdb.set_trace()
        filtr = wrap.beta[:,idime]-rdim[idime] > 0.5
        wrap.beta[:,idime][filtr] -= 1.0
        filtr = wrap.beta[:,idime]-rdim[idime] <= -0.5
        wrap.beta[:,idime][filtr] += 1.0
    wrap.GetXYZ()
    wrap.Set(atomd)

    #--- add box bounds
    boxd.CellOrigin=rcent-np.matmul(box.CellVector,np.array([.5,.5,.5]))
    loo=boxd.CellOrigin
    hii=boxd.CellOrigin+np.matmul(box.CellVector,np.array([1,1,1]))
    boxd.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]


    if CenterAtZero:     
        atomd.x -= rcent[0]
        atomd.y -= rcent[1]
        atomd.z -= rcent[2]
        boxd.CellOrigin -= rcent
        loo=boxd.CellOrigin
        hii=boxd.CellOrigin+np.matmul(box.CellVector,np.array([1,1,1]))
        boxd.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]

    wrap = lp.Wrap(atomd,boxd)
#    display(pd.DataFrame(atomd.__dict__)[~wrap.isInside()])
#    print(np.all(wrap.isInside()))
    assert np.all(wrap.isInside()), 'aotms outside box!'


def ReplaceAtom( atoms,  Replace ):
    told = list(Replace.keys())[0]
    tnew = list(Replace.values())[0]    
    df=pd.DataFrame(atoms.__dict__)
    indr = int(np.random.choice(df[df.type==told].index))
#    display(df.iloc[indr])
    df['type'].iloc[indr] = tnew
#    df.iloc[indr]
    return df, int(df.id.iloc[indr])
    
def isEmpty( val ):
    if type(val) == type({}):
        return True if len( val ) == 0 else False

path = sys.argv[1]
output = sys.argv[2]
lib_path = sys.argv[3]
nout = int(sys.argv[4])

sys.path.append(lib_path)
import LammpsPostProcess2nd as lp

#--- read data
rd = lp.ReadDumpFile(path)
rd.ReadData()
atoms = lp.Atoms( **rd.coord_atoms_broken[0].to_dict(orient='series') )
box = lp.Box( BoxBounds = rd.BoxBounds[0], AddMissing = np.array([0.0,0.0,0.0] ))

#--- pick at random & remove
df=pd.DataFrame(atoms.__dict__)
df=df.sample(n=df.shape[0]-nout)
#idout = np.sum(atoms.id)-np.sum(df['id']) #--- atom id taken out
atomd = lp.Atoms(**df.to_dict(orient='series'))
atomd.id=np.arange(1,len(atomd.id)+1) #--- reset id
boxd = lp.Box( BoxBounds = rd.BoxBounds[0], AddMissing = np.array([0.0,0.0,0.0] ))

#--- center atom positions
#CenterAtoms( atoms, box,
#             atomd, boxd, #--- will be modified
#            idcent = idout,
#            CenterAtZero=True, #--- don't change!
#            )
#--- write
lp.WriteDataFile(atomd,boxd,rd.mass).Write(output)

