import pdb
import numpy as np
import traceback
import os

number_hidden_layers  = dict(zip(range(3),[1,2,4]))
num_kmc_steps         = 100
nrun                  = 1
elapsed_times         = []
nvacs                 = []
os.system('rm wall_times.txt')
for (key, val) in number_hidden_layers.items():
	for irun in range( nrun ):
		path = 'mlmc/ni/multipleVacs/results/kmc/vac%s/Run%s'%(key,irun)
		file_title = 'mlmc_ni_multipleVacs_results_kmc_vac%s.%s.err'%(key,irun)
		try:
        
			with open('%s/saved_output/Diffusion.dat'%path,'r') as fpp:
				assert len(fpp.readlines()) == 2 + num_kmc_steps
			with open('%s/%s'%(path,file_title),'r') as fp:
				elapsed_time = float(fp.readlines()[-1])
				elapsed_times.append(elapsed_time)
				nvacs.append(val)
		except:
			traceback.print_exc()
			continue
np.savetxt('wall_times.txt',np.c_[nvacs,elapsed_times],header='num_vacancies wallTime(s)')
print('saved in wall_times.txt')
