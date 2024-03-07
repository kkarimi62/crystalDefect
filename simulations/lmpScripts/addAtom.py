import sys
import os
import pdb
import numpy as np
import pandas as pd

def WriteDataFile(AtomskOutpt,outpt):
    #--- read data file
    lmpData = lp.ReadDumpFile( AtomskOutpt )
    lmpData.ReadData()
    #--- atom obj
    df = lmpData.coord_atoms_broken[0]
    atoms = lp.Atoms( **df.to_dict(orient='series') )
    #--- box
    box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )
#    pdb.set_trace()

    #--- wrap
    wrap = lp.Wrap( atoms, box )
    wrap.WrapCoord()
    wrap.Set( atoms )


    #--- center
    #--- add box bounds
    l = 2.0*3.0 #--- put H within a cube with size l
    distance = 0.0 #3.0 #--- distance from extended dislocation
    
    rcent = np.matmul(box.CellVector,np.array([.5,.5,.5])) + box.CellOrigin
    rcent += np.array([-0.5*l,-0.5*l,-0.5*l]) #--- new origin
#    rcent += np.array([-0.5*l,distance,-0.5*l]) #--- new origin
    
    loo=rcent
    hii=rcent+np.matmul(np.array([[l,0,0],[0,l,0],[0,0,l]]),np.array([1,1,1]))
    box.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]
    box = lp.Box( BoxBounds = box.BoxBounds,AddMissing = np.array([0.0,0.0,0.0] ) )
    
    #--- wrap
    wrap = lp.Wrap( atoms, box )
    filtr = wrap.isInside()

#     sfile=open('ovito.xyz','w')
#     utl.PrintOvito(pd.DataFrame(df[filtr]), sfile, 'junk', attr_list=['type','x', 'y', 'z','id'])
#     sfile.close()

    

#    pdb.set_trace()
    AL_ids = df[filtr].id #index
    print('n=',len(AL_ids))
    for ID in AL_ids:
        os.system('atomsk %s -add-atom H near %s final.cfg'%(inputt,ID))
        os.system('atomsk final.cfg lmp')
        os.system('mv final.lmp %s'%inputt)
        os.system('rm *.cfg')

    #--- perturb and write
    lmpData = lp.ReadDumpFile( inputt )
    lmpData.ReadData()
    #--- atom obj
    df = lmpData.coord_atoms_broken[0]
    xyz_perturbed = np.c_[df['x y z'.split()]] + np.random.normal(loc=0.0, scale=0.1, size=3*len(df)).reshape((len(df),3))
    df_new = pd.DataFrame(np.c_[df['id type'.split()],xyz_perturbed],columns='id type x y z'.split())

    atoms = lp.Atoms( **df_new.to_dict(orient='series') )
    #--- box
    box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )
    mass=lmpData.mass #{1:26.98153860,2:1.00800000}
#    pdb.set_trace()
    wd = lp.WriteDataFile(atoms, box,mass)
    wd.Write(outpt) #,itime=itime,
                 #attrs=['id', 'type', 'x', 'y', 'z','val'], 
                 #fmt='%i %i %4.3e %4.3e %4.3e %4.3e')

#    if inputt != outpt:    
#        os.system('mv %s %s'%(inputt,outpt))

    
    #--- hydrogen atoms outside
#    isHyd = df['type'] == 2
#    hydOutside = np.all([~filtr,isHyd],axis=0)
#    print(df[filtr]['id'])


    #
#    df = pd.DataFrame(atoms.__dict__)
#    atoms = lp.Atoms( **df[~hydOutside].to_dict(orient='series') )
#    box = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )

#    write data file
#    mass = {1:26.982,2:1.008}
#    lp.WriteDataFile(atoms,box,mass).Write(outpt)

lib_path = sys.argv[1]
inputt = sys.argv[2]
outpt = sys.argv[3]
natom = int(sys.argv[4])

sys.path.append(lib_path)
import LammpsPostProcess2nd as lp
#import utility as utl 

os.system('atomsk %s -add-atom H random %s final.cfg'%(inputt,natom))
os.system('atomsk final.cfg lmp')
#WriteDataFile(inputt,outpt)
#os.system('atomsk final.cfg lmp')
#os.system('rm final.lmp')
os.system('mv final.lmp %s'%outpt)
os.system('rm *.cfg')

