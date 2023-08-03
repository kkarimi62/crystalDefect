import sys
import pdb
import numpy as np

def Fetch(fp):
    with open(fp) as sfile:
        slist = sfile.readlines()
        line_start = slist.index(' Listing topologies in system:\n')

        line_no = line_start
        while slist[line_no].split()[0]!='known_tc,':
            line_no += 1
        assert slist[line_no].split()[0]=='known_tc,'
        line_end = line_no
            


        xstr= slist[line_start+1:line_end]

        topos = list(map(lambda x:int(x.split()[0][:-1]),xstr))
        return topos


fp0 = sys.argv[1]
indx = int(sys.argv[2])
fout = sys.argv[3]

#topo = list(set(Fetch(fp1)) - set(Fetch(fp0)))
topo = Fetch(fp0) #--- to be ignored
#topo.pop(indx)
with open(fout,'w') as ff:
    np.savetxt(ff,np.c_[topo],fmt='%d')

