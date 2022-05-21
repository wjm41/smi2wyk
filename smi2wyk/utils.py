from typing import List

import numpy as np

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


def slice_indices_according_to_rank_and_size(my_rank, mpi_size, length_of_object_to_slice):
    mpi_borders = np.linspace(
        0, length_of_object_to_slice, mpi_size + 1).astype('int')

    border_low = mpi_borders[my_rank]
    border_high = mpi_borders[my_rank+1]
    return border_low, border_high

def slice_object_according_to_rank_and_size(my_rank, number_of_processes, object_to_slice):
    
    my_start_index, my_end_index = slice_indices_according_to_rank_and_size(my_rank, number_of_processes, len(object_to_slice))

    my_results = object_to_slice[my_start_index: my_end_index]
    return  my_results
