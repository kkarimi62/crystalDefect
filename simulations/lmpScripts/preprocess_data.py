import pdb

def SetTypes( atoms, itype ):
    atoms.type=np.ones(len(atoms.type),dtype=int)


def Center(atoms,box):
    #--- center
    #--- add box bounds
    rcent = np.matmul(box.CellVector,np.array([.5,.5,.5]))
    box.CellOrigin -= rcent
    loo=box.CellOrigin
    hii=box.CellOrigin+np.matmul(box.CellVector,np.array([1,1,1]))
    box.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]

    atoms.x -= rcent[0]
    atoms.y -= rcent[1]
    atoms.z -= rcent[2]

def zeroShift(atoms,box):
    atoms.x -= box.CellOrigin[0]
    atoms.y -= box.CellOrigin[1]
    atoms.z -= box.CellOrigin[2]
    
    box.CellOrigin = np.array([0.0,0.0,0.0])
    loo=np.array([0.0,0.0,0.0])
    hii=np.matmul(box.CellVector,np.array([1,1,1]))
    box.BoxBounds=np.c_[loo,hii,np.array([0,0,0])]


if __name__ == '__main__':
	import sys
	import numpy as np

	fin      = sys.argv[ 1 ]
	fout     = sys.argv[ 2 ]
	lib_path = sys.argv[ 3 ]

	sys.path.append( lib_path )

	import LammpsPostProcess2nd as lp

	#--- parse
	lmpData  = lp.ReadDumpFile( fin )
	lmpData.GetCords() #ReadData()
	
	#--- mass is hard-coded!
	mass     = { 1:58.71, 2:58.71 }
	
	#--- atoms & box
	itime    = list(lmpData.coord_atoms_broken.keys())[ 0 ]
	atoms    = lp.Atoms( **lmpData.coord_atoms_broken[0].to_dict(orient='series') )
	box      = lp.Box( BoxBounds = lmpData.BoxBounds[0],AddMissing = np.array([0.0,0.0,0.0] ) )

	#--- wrap
	wrap     = lp.Wrap( atoms, box )
	wrap.WrapCoord()
	wrap.Set( atoms )

	#--- shift coords
	#    Center( atoms, box )
	zeroShift( atoms, box )

	#--- set atom types 
	SetTypes( atoms, 1 )

	#--- save
	lp.WriteDataFile(atoms,box,mass).Write( fout )


