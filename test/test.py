from dscribe.descriptors import SOAP
import ase
import ase.io
import pdb

atoms = ase.io.read("equilibrated.dat",format="lammps-data",style='atomic')

species = 'Co Ni Cr Fe Mn'.split()
r_cut = 6.0
n_max = 8
l_max = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=True,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
)


# set atomic numbers for each species
numbers = list(map(lambda x:soap.index_to_atomic_number[x-1],atoms.get_atomic_numbers()))
atoms.set_atomic_numbers(numbers)
#assert soap.check_atomic_numbers(atoms.get_atomic_numbers())


soap_descriptors = soap.create( atoms,centers=[0])


pdb.set_trace()

