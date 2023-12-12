import os
import sys
import pdb 
import numpy as np

if __name__ == '__main__':
	lib_path = sys.argv[1] #'/mnt/home/kkarimi/Project/git/HeaDef/postprocess'
    
	fin      = sys.argv[2]

	
	sys.path.append(lib_path)

	import LammpsPostProcess as lp

	fp = 'data_void_twoGroups.dat'
	os.system('ovitos %s/OvitosCna.py %s %s 1 0'%(lib_path,fin,fp))

	#--- load cna data
	rd = lp.ReadDumpFile( fp )
	rd.GetCords()
	df = rd.coord_atoms_broken[ 0 ]

	#--- assign types
	filtr = df.StructureType.astype(int) == 0
	indices_secondType = df[ filtr ].index
	df.loc[ indices_secondType,'type' ] = 2

	#--- print
    
	atom = lp.Atoms(**df.to_dict(orient='series'))
    
	box      = lp.Box(BoxBounds=rd.BoxBounds[0],AddMissing=np.array([0,0,0]))
    
	wd   = lp.WriteDataFile(atom, box, {1:1,2:1}) #--- modify!!
    
	wd.Write( fin )

	os.system('rm %s'%fp)
	#pdb.set_trace()
