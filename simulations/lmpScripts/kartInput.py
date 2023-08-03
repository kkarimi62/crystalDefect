import pdb
import sys
import numpy as np


path = sys.argv[1]
output = sys.argv[2]
lib_path = sys.argv[3]
temp = sys.argv[4]

sys.path.append(lib_path)
import LammpsPostProcess2nd as lp

#--- read lammps data file
rd = lp.ReadDumpFile(path)
rd.ReadData()

#--- output readable format for kart 
[lx,ly,lz] = rd.BoxBounds[0].astype(float)[:,1]-rd.BoxBounds[0].astype(float)[:,0]
df = rd.coord_atoms_broken[ 0 ]
#pdb.set_trace()

#--- output
sfile = open(output,'w')
sfile.write('run_id:            0\ntotal energy :    0.0000\n')
sfile.write('%s\t%s\t%s\n'%(lx,ly,lz))
np.savetxt(sfile,np.c_[df.type,df.x,df.y,df.z],fmt='%i %s %s %s')
sfile.close()

sfile = open('.natom.txt','w')
sfile.write( '%s\n'%(len(df)))
sfile.close()

sfile = open('.lx.txt','w')
sfile.write( '%s\n'%(np.ceil(lx)))
sfile.close()

sfile = open('.temp.txt','w')
sfile.write( '%s\n'%temp)
sfile.close()
