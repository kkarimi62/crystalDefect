import pdb

with open('kmc.sh') as fp:
	strs = fp.readlines()

for xstr,indx in zip(strs,range(len(strs))):
	if xstr[:6] == 'setenv':
		val = xstr.split(maxsplit=2)[2] 
		if xstr.split()[2] == '.true.':
			val = True
		elif xstr.split()[2] == '.false.':
			val = False
		ystr = 'export %s=%s\n'%(xstr.split()[1],val)
		strs[indx] = ystr
	elif xstr[:3] == 'set':
		ystr = xstr[4:] #'=%s\n'%(xstr.split()[1],val)
#		print(xstr.split())
#		print(ystr)

		strs[indx] = ystr
#	if indx > 27:
#		break
strs[0] = '#!/bin/bash'
open('kmc_bash.sh','w').writelines(strs[:-1])
