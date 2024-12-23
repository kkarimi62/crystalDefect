import os
import sys
import pdb 
import numpy as np

if __name__ == '__main__':
    lib_path = sys.argv[1] 
    fin      = sys.argv[2]
    structureType = sys.argv[3] #--- either 'Other', 'FCC', 'HCP', 'BCC', 'ICO'
    typeAssigned  = int(sys.argv[4])
#     rmin     = float(sys.argv[3])
#     rmax     = float(sys.argv[4])

    structureTypes = {'Other':0, 'FCC':1, 'HCP':2, 'BCC':3, 'ICO':4}

    sys.path.append(lib_path)

    import LammpsPostProcess2nd as lp

    fp = 'data_void_twoGroups.dat'

	#--- cna analysis
    os.system('ovitos %s/OvitosCna.py %s %s 1 0'%(lib_path,fin,fp)) #--- args: fin, fout, nevery, AnalysisType


    rd    = lp.ReadDumpFile( fin )

    rd.ReadData()


    mass  = rd.mass
    box      = lp.Box(BoxBounds=rd.BoxBounds[0],AddMissing=np.array([0,0,0]))

    #--- load cna data
    rd = lp.ReadDumpFile( fp )
    rd.GetCords()
    df = rd.coord_atoms_broken[ 0 ]
    atoms = lp.Atoms(**df.to_dict(orient='series'))

    #--- assign types

    #--- within (rmin,rmax)
#     center = box.CellOrigin + np.matmul( box.CellVector, 0.5 * np.array( [ 1, 1, 1 ] ) )
#     dr     = np.c_[ atoms.x, atoms.y, atoms.z ] - center
#     dr_sq  = np.sum( dr * dr, axis = 1 )
#     filtr = np.all([dr_sq < rmax * rmax,dr_sq >= rmin * rmin],axis=0)
    
    #--- non-crystalline atoms
    filtr = df.StructureType.astype(int) == structureTypes[ structureType ]
    indices_secondType = df[ filtr ].index
    df.loc[ indices_secondType,'type' ] = typeAssigned

    #--- print
    mass[ str(typeAssigned) ] = mass[ '1' ]
    atom = lp.Atoms(**df.to_dict(orient='series'))
    lp.WriteDataFile( atom, box, mass ).Write( fin )

    os.system('rm %s'%fp)
    #pdb.set_trace()
