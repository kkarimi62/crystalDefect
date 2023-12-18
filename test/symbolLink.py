

if __name__ == '__main__':
	import os
	import sys
	#--- 

	source       = os.getcwd() + '/neuralNet/ni/new_void5th/Run0'
	destination  = os.getcwd() + '/neuralNet/ni/new_void5th'
	N = 32
	#---
	for irun in range(1,N):
		writPath = '%s/Run%s' % ( destination, irun ) # --- curr. dir
		os.system( 'ln -s %s %s' % ( source, writPath ) )
