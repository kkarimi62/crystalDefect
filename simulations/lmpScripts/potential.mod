# NOTE: This script can be modified for different pair styles 
# See in.elastic for more info.

# Choose potential
pair_style	eam/alloy
pair_coeff              * * ${PathEam}/NiCoCr.lammps.eam Ni Co Cr
#
#pair_style                eam
#pair_coeff                * * ${PathEam}/Ni_u3.eam

#pair_style meam/c
#pair_coeff * * ${PathEam}/library_CoNiCrFeMn.meam Co Ni Cr Fe Mn ${PathEam}/parameters_CoNiCrFeMn.meam Co Ni Cr Fe Mn

#pair_style meam/c
#pair_coeff * * ${PathEam}/library_CoNiCrFeMn.meam Ni Co Cr Fe Mn ${PathEam}/parameters_CoNiCrFeMn.meam Ni Co Cr Fe Mn

# Setup neighbor style
#neighbor 1.0 nsq
#neigh_modify once no every 1 delay 0 check yes
#--- kart settings
neighbor        0.0  bin # multi
#neigh_modify    delay 0 every 1 check no one 50 page 550 #binsize 0.1


# Setup minimization style
min_style	     cg
min_modify	     dmax ${dmax} line quadratic

# Setup output
thermo		1000
#thermo_style custom step temp pe press pxx pyy pzz pxy pxz pyz lx ly lz vol
thermo_style custom step temp pe press vol
thermo_modify norm no
