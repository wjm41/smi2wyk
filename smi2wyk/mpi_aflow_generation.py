import pandas as pd
from tqdm import tqdm
from fire import Fire
import numpy as np
from mpi4py import MPI


from smi2wyk.aflow import get_aflow_label_spglib, CifStringParser, get_aflow_label_with_aflow_from_ccdc_crystal
from smi2wyk.ccsd_utils import search_ccdc_for_organic_mols
from smi2wyk.utils import slice_object_according_to_rank_and_size

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
    
def save_results(df_wyckoff, search_results, output_path):    
    df_wyckoff = pd.concat(df_wyckoff).reset_index()
    print(f'Successful featurisations: {len(df_wyckoff)/len(search_results)*100:.2f}%')
    print(f'Unique featurisations: {len(df_wyckoff.identifier.unique())/len(search_results)*100:.2f}%')

    df_wyckoff.to_csv(output_path, index=False)
    return

def main(num_results:int = -1,
         output_path:str = 'aflow_vs_spg.csv'):
    
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    
    search_results = search_ccdc_for_organic_mols(num_results)
    my_results = slice_object_according_to_rank_and_size(mpi_rank, mpi_size, search_results)
    my_df = get_aflow_labels_for_search_results(my_results)
    
    mpi_comm.Barrier()
        
    df_wyckoff = mpi_comm.gather(my_df, root=0)

    if mpi_rank == 0:
        save_results(df_wyckoff, search_results, output_path)
    return

if __name__ == '__main__':
    Fire(main)