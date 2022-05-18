from typing import List
from ccdc import io, utilities

def write_slurm_script(job_name: str,
                       run_time: str,
                       output_name: str,
                       script: str,
                       args: List,
                       file_name: str,
                       package_dir: str = None,
                       email: bool = False,
                       gpu: bool = False,
                       conda_env: str = 'ampere'):

    if gpu:
        slurm_options = [
            '#!/bin/bash',
            f'#SBATCH -J {job_name}',
            '#SBATCH -A LEE-WJM41-SL2-GPU',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            '#SBATCH --gres=gpu:1',
            f'#SBATCH --time={run_time}',
            '#SBATCH --mail-user=wjm41@cam.ac.uk',
            f'#SBATCH --output={output_name}',
            '#SBATCH -p ampere',
        ]
    else:
        slurm_options = [
            '#!/bin/bash',
            f'#SBATCH -J {job_name}',
            '#SBATCH -A LEE-WJM41-SL2-CPU',
            '#SBATCH --nodes=1',
            '#SBATCH --ntasks=1',
            ' #SBATCH --cpus-per-task=1',
            f'#SBATCH --time={run_time}',
            '#SBATCH --mail-user=wjm41@cam.ac.uk',
            f'#SBATCH --output={output_name}',
            '#SBATCH -p icelake-himem',
        ]
    if email:
        slurm_options.append('#SBATCH --mail-type=ALL')

    if gpu:
        module_options = [
            '. /etc/profile.d/modules.sh',
            'module purge',
            'module load rhel8/default-amp',
            'module load miniconda/3',
            f'source activate {conda_env}',
        ]
    else:
        module_options = [
            '. /etc/profile.d/modules.sh',
            'module purge',
            'module load rhel8/default-amp',
            'module load miniconda/3',
            f'source activate {conda_env}',
        ]
    if package_dir is not None:
        pre_empt = f'cd {package_dir}; pip install . --use-feature=in-tree-build'
    else:
        pre_empt = ''

    slurm_options = '\n'.join(slurm_options)
    module_options = '\n'.join(module_options)
    command_to_run = ' '.join([script]+args)

    string_to_write = f'{slurm_options}\n{module_options}\n{pre_empt}\n{command_to_run}'

    with open(file_name, 'w') as f:
        f.write(string_to_write)

    return

def insepct_entry(csd_entry: str):
    csd_reader = io.EntryReader()

    entry_name = 'ABIMUW'
    entry = csd_reader.entry(entry_name)
    crystal = entry.crystal

    print(f'SMILES: {crystal.molecule.smiles}')
    print(f'Crystal System: {crystal.crystal_system}')
    print(f'Spacegroup Symbol: {crystal.spacegroup_symbol}')
    print(f'Spacegroup Number: {crystal.spacegroup_number_and_setting}')
    print(f'Has disorder: {crystal.has_disorder}')
    print(f'Disorder details: {entry.disorder_details}')
    
    print('\n'.join('%-17s %s' % (op, utilities.print_set(crystal.atoms_on_special_positions(op))) for op in crystal.symmetry_operators))
    return

def spacegroup_from_crystal(csd_entry: str, reader: io.EntryReader = None):
    
    if reader is None:
        reader = io.EntryReader()

    entry = reader.entry(csd_entry)
    crystal = entry.crystal
    spg = crystal.spacegroup_number_and_setting[0]
    return spg
    