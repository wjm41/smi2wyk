import pandas as pd
from tqdm import tqdm
import numpy as np
from mpi4py import MPI

from ccdc.search import SMARTSSubstructure, SubstructureSearch

from smi2wyk.wren_code.utils import get_aflow_label_spglib, CifStringParser, count_wyks, count_params, count_distinct_wyckoff_letters, return_spacegroup_number

def slice_indices_according_to_rank_and_size(my_rank, mpi_size, length_of_object_to_slice):
    mpi_borders = np.linspace(0, length_of_object_to_slice, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_rank]
    border_high = mpi_borders[my_rank+1]
    return border_low, border_high

any_atom = SMARTSSubstructure("[X]")
any_atom_search = SubstructureSearch()
any_atom_search.add_substructure(any_atom)

any_atom_search.settings.has_3d_coordinates = True
any_atom_search.settings.no_disorder = True
any_atom_search.settings.no_errors = True
any_atom_search.settings.only_organic = True
any_atom_search.settings.not_polymeric = True

# any_atom_search.settings.max_hit_structures = 10
any_atom_search.settings.max_hits_per_structure = 1

search_results = any_atom_search.search()

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

my_start_index, my_end_index = slice_indices_according_to_rank_and_size(mpi_rank, mpi_size, len(search_results))

my_results = search_results[my_start_index, my_end_index]

entry_identifiers = []
entry_smiles = []
entry_wyckoffs = []
for hit in tqdm(my_results):
    try:
        cif_string = hit.crystal.to_string(format='cif')
        struct = CifStringParser(cif_string, occupancy_tolerance=10).get_structures()[0]
        wyckoff_label = get_aflow_label_spglib(struct)

        entry_identifiers.append(hit.entry.identifier)
        entry_smiles.append(hit.molecule.smiles)
        entry_wyckoffs.append(wyckoff_label)
    except:
        pass
    
mpi_comm.Barrier()
    
    
my_df = pd.DataFrame({'identifier': entry_identifiers,
                            'smiles': entry_smiles,
                            'wyckoff': entry_wyckoffs})

df_wyckoff = mpi_comm.gather(my_df, root=0)

if mpi_rank == 0:
    
    df_wyckoff = pd.concat(df_wyckoff).reset_index()
    print(f'Successful featurisations: {len(df_wyckoff)/len(search_results)*100:.2f}%')
    print(f'Unique featurisations: {len(df_wyckoff.identifier.unique())/len(search_results)*100:.2f}%')

    df_wyckoff.to_csv('csd_organic.csv', index=False)