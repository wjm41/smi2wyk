from ccdc.docking import Docker
from ccdc.io import MoleculeReader, EntryReader

docker = Docker()
settings = docker.settings

MLL1_protein_file = '/Applications/CCDC/Python_API_2022/docs/example_files/2w5y_protein_prepared.mol2'
settings.add_protein_file(MLL1_protein_file)

MLL1_native_ligand_file = '/Applications/CCDC/Python_API_2022/docs/example_files/SAH_native.mol2'
MLL1_native_ligand = MoleculeReader(MLL1_native_ligand_file)[0]
MLL1_protein = settings.proteins[0]
settings.binding_site = settings.BindingSiteFromLigand(MLL1_protein, MLL1_native_ligand, 8.0)

settings.fitness_function = 'plp'
settings.autoscale = 10.
settings.early_termination = False
import tempfile
batch_tempd = tempfile.mkdtemp()
settings.output_directory = batch_tempd
settings.output_file = '/Applications/CCDC/Python_API_2022/docs/example_files/docked_ligands.mol2'

MLL1_ligand_file = '/Applications/CCDC/Python_API_2022/docs/example_files/SAH.mol2'
MLL1_ligand = MoleculeReader(MLL1_ligand_file)[0]
settings.add_ligand_file(MLL1_ligand_file, 10)

results = docker.dock() 

ligands = results.ligands
atoms = ligands[0].molecule.atoms

for atom in atoms:
    print(atom.coordinates)
    
print(ligands[0].molecule.to_string('sdf'))
