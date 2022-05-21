from ccdc import io, utilities
from ccdc.search import SMARTSSubstructure, SubstructureSearch


def insepct_entry(csd_entry: str):
    csd_reader = io.EntryReader()

    entry = csd_reader.entry(csd_entry)
    crystal = entry.crystal

    print(f'SMILES: {crystal.molecule.smiles}')
    print(f'Crystal System: {crystal.crystal_system}')
    print(f'Spacegroup Symbol: {crystal.spacegroup_symbol}')
    print(f'Spacegroup Number: {crystal.spacegroup_number_and_setting}')
    print(f'Has disorder: {crystal.has_disorder}')
    print(f'Disorder details: {entry.disorder_details}')
    
    print('\n'.join('%-17s %s' % (op, utilities.print_set(crystal.atoms_on_special_positions(op))) for op in crystal.symmetry_operators))
    return

def spacegroup_num_and_str_from_crystal(row, reader: io.EntryReader = None):
    
    if reader is None:
        reader = io.EntryReader()

    csd_entry = row['identifier']
    entry = reader.entry(csd_entry)
    crystal = entry.crystal
    try:
        spg_num = crystal.spacegroup_number_and_setting[0]
        spg_str= crystal.spacegroup_symbol
        row['spg_num'] = spg_num
        row['spg_str'] = spg_str
    except:
        row['spg_num'] = None
        row['spg_str'] = None
    return row

def spacegroup_str_from_crystal(csd_entry: str, reader: io.EntryReader = None):
    
    if reader is None:
        reader = io.EntryReader()

    entry = reader.entry(csd_entry)
    crystal = entry.crystal
    spg = crystal.spacegroup_symbol
    return spg
    
def search_ccdc_for_organic_mols(n_hits: int = 100):
    any_atom = SMARTSSubstructure("[X]")
    any_atom_search = SubstructureSearch()
    any_atom_search.add_substructure(any_atom)

    any_atom_search.settings.has_3d_coordinates = True
    any_atom_search.settings.no_disorder = True
    any_atom_search.settings.no_errors = True
    any_atom_search.settings.only_organic = True
    any_atom_search.settings.not_polymeric = True

    if n_hits != -1:
        any_atom_search.settings.max_hit_structures = n_hits
    any_atom_search.settings.max_hits_per_structure = 1

    search_results = any_atom_search.search()
    return search_results