import pandas as pd
from tqdm import tqdm
from fire import Fire
import numpy as np
from mpi4py import MPI

from ccdc.search import SMARTSSubstructure, SubstructureSearch

from smi2wyk.wren_code.utils import get_aflow_label_spglib, CifStringParser, get_aflow_label_with_aflow_from_ccdc_crystal


def slice_indices_according_to_rank_and_size(my_rank, mpi_size, length_of_object_to_slice):
    mpi_borders = np.linspace(
        0, length_of_object_to_slice, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_rank]
    border_high = mpi_borders[my_rank+1]
    return border_low, border_high


def search_ccdc_for_organic_mols(n:int = 100):
    any_atom = SMARTSSubstructure("[X]")
    any_atom_search = SubstructureSearch()
    any_atom_search.add_substructure(any_atom)

    any_atom_search.settings.has_3d_coordinates = True
    any_atom_search.settings.no_disorder = True
    any_atom_search.settings.no_errors = True
    any_atom_search.settings.only_organic = True
    any_atom_search.settings.not_polymeric = True

    any_atom_search.settings.max_hit_structures = n
    any_atom_search.settings.max_hits_per_structure = 1

    search_results = any_atom_search.search()
    return search_results

def slice_search_results(my_rank, number_of_processes, search_results):
    

    my_start_index, my_end_index = slice_indices_according_to_rank_and_size(my_rank, number_of_processes, len(search_results))

    my_results = search_results[my_start_index: my_end_index]
    return  my_results

def get_aflow_labels_for_search_results(search_results):
    entry_identifiers = []
    entry_smiles = []
    entry_spacegroup = []
    entry_wyckoffs_spg = []
    entry_wyckoffs_aflow = []

    for hit in tqdm(search_results):
        try:
            cif_string = hit.crystal.to_string(format='cif')
            struct = CifStringParser(cif_string, occupancy_tolerance=10).get_structures()[0]
            wyckoff_label_spglib = get_aflow_label_spglib(struct)
            wyckoff_label_aflow = get_aflow_label_with_aflow_from_ccdc_crystal(hit.crystal)
            entry_identifiers.append(hit.entry.identifier)
            entry_smiles.append(hit.molecule.smiles)
            entry_spacegroup.append(hit.crystal.spacegroup_number_and_setting)
            entry_wyckoffs_spg.append(wyckoff_label_spglib)
            entry_wyckoffs_aflow.append(wyckoff_label_aflow)
        except:
            pass
        
    df_wyckoff = pd.DataFrame({'identifier': entry_identifiers,
                            'smiles': entry_smiles,
                            'csd_spacegroup': entry_spacegroup,
                            'wyckoff_spg': entry_wyckoffs_spg,
                            'wyckoff_aflow': entry_wyckoffs_aflow})
    return df_wyckoff
    
def save_results(df_wyckoff, search_results):    
    df_wyckoff = pd.concat(df_wyckoff).reset_index()
    print(f'Successful featurisations: {len(df_wyckoff)/len(search_results)*100:.2f}%')
    print(f'Unique featurisations: {len(df_wyckoff.identifier.unique())/len(search_results)*100:.2f}%')

    df_wyckoff.to_csv('aflow_vs_spg.csv', index=False)
    return

def main(n:int = 100):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    search_results = search_ccdc_for_organic_mols(n)
    my_results = slice_search_results(mpi_rank, mpi_size, search_results)
    my_df = get_aflow_labels_for_search_results(my_results)
    
    mpi_comm.Barrier()
        
    df_wyckoff = mpi_comm.gather(my_df, root=0)

    if mpi_rank == 0:
        save_results(df_wyckoff, search_results)
    return

if __name__ == '__main__':
    Fire(main)