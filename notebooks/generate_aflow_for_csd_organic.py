import pandas as pd
from tqdm import tqdm

from ccdc.search import SMARTSSubstructure, SubstructureSearch

from smi2wyk.wren_code.utils import get_aflow_label_spglib, CifStringParser, count_wyks, count_params, count_distinct_wyckoff_letters, return_spacegroup_number

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


entry_identifiers = []
entry_smiles = []
entry_wyckoffs = []
for hit in tqdm(search_results):
    try:
        cif_string = hit.crystal.to_string(format='cif')
        struct = CifStringParser(cif_string, occupancy_tolerance=10).get_structures()[0]
        wyckoff_label = get_aflow_label_spglib(struct)

        entry_identifiers.append(hit.entry.identifier)
        entry_smiles.append(hit.molecule.smiles)
        entry_wyckoffs.append(wyckoff_label)
    except:
        pass
    
df_wyckoff = pd.DataFrame({'identifier': entry_identifiers,
                           'smiles': entry_smiles,
                           'wyckoff': entry_wyckoffs})

print(f'Successful featurisations: {len(df_wyckoff)/len(search_results)*100:.2f}%')
print(f'Unique featurisations: {len(df_wyckoff.identifier.unique())/len(search_results)*100:.2f}%')

df_wyckoff.to_csv('csd_organic.csv', index=False)

# tqdm.pandas()

# df_wyckoff['n_atoms'] = df_wyckoff['wyckoff'].progress_apply(count_wyks)
# df_wyckoff['n_wyk'] = df_wyckoff['wyckoff'].progress_apply(count_distinct_wyckoff_letters)
# df_wyckoff['spg'] = df_wyckoff['wyckoff'].progress_apply(return_spacegroup_number)
# df_wyckoff['n_param'] = df_wyckoff['wyckoff'].progress_apply(count_params)

# df_wyckoff.to_csv('csd_organic_params.csv', index=False)
