
import os 
from typing import List
from pathlib import Path
import re

import yaml
import pandas as pd

from smi2wyk.utils import write_slurm_script

def tokenize_smiles(smi):
    """
    Tokenize a SMILES molecule or reaction 
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    smi_tokenized = ' '.join(tokens)
    return smi_tokenized

def tokenize_prototype_label(aflow_str: str) -> int:
    """Tokenize prototype label of Wyckoff representation."""
    aflow_str, _ = aflow_str.split(":")
    prototype, pearson, spg_no, *wyk = aflow_str.split("_")
    prototype_tokenized = ' '.join(re.split('(\d+)',prototype))
    prototype_tokenized = re.sub('([A-Z])([A-Z])', r'\1 \2', prototype_tokenized)
    return prototype_tokenized

def tokenize_pearson_symbol(aflow_str: str) -> int:
    """Tokenize Pearson Symbol of Wyckoff representation."""
    aflow_str, _ = aflow_str.split(":")
    prototype, pearson, spg_no, *wyk = aflow_str.split("_")
    pearson_tokenized = re.split('(\d+)',pearson)
    pearson_tokenized = ' '.join(list(pearson_tokenized[0]) + pearson_tokenized[1:])
    # pearson_tokenized = [' '.join(re.split('([a-zA-Z])', token)) for token in pearson_tokenized]
    return pearson_tokenized


def return_default_training_params(n_gpu=1):
        
    training_params = dict(
        save_checkpoint_steps = 2500,
        keep_checkpoint = 2,
        seed = 42,
        train_steps = 500000,
        valid_steps = 5000,
        warmup_steps = 8000,
        report_every = 1000,
        
        decoder_type = 'transformer',
        encoder_type = 'transformer',
        word_vec_size = 256,
        rnn_size = 256, 
        layers = 4,
        transformer_ff = 2048,
        heads = 8,
        global_attention = 'general',
        global_attention_function = 'softmax',
        self_attn_type = 'scaled-dot',
        
        accum_count = 4,
        optim = 'adam',
        adam_beta1 = 0.9,
        adam_beta2 = 0.998,
        decay_method = 'noam',
        learning_rate = 2.0,
        max_grad_norm = 0.0,

        batch_size = 1024,
        batch_type = 'tokens',
        normalization = 'tokens',
        dropout = 0.1,
        label_smoothing = 0.0,

        max_generator_batches = 32,

        param_init = 0.0,
        param_init_glorot = 'true',
        position_encoding = 'true',

        world_size = n_gpu,
        gpu_ranks = [i for i in range(n_gpu)]
        )
    
    return training_params

def return_data_params(data_path, 
                       share_vocab:bool, 
                       weighted_sampling:bool =False, 
                       weight_folder_names: List = None, 
                       weights: List = None):
    
    data_params = dict(
        save_data = f'{data_path}',
        src_vocab = f'{data_path}/vocab.src',
        tgt_vocab = f'{data_path}/vocab.tgt',
        share_vocab = share_vocab,
    )
    
    if weighted_sampling:
        assert (len(weight_folder_names) == len(weights), "Number of weight folders and weights must be the same")
        data_path_dict = dict(valid = dict(
                path_src = f'{data_path}/src-valid.csv',
                path_tgt = f'{data_path}/tgt-valid.csv',
            ))
        
        for index, (weight, folder_name) in enumerate(zip(weights, weight_folder_names)):
            data_path_dict[f'train_{index}'] = dict(
                path_src = f'{data_path}/{folder_name}/src-train.csv',
                path_tgt = f'{data_path}/{folder_name}/tgt-train.csv',
                weight = weight,
            )
        
    else:
        data_path_dict = dict(
            train = dict(
                path_src = f'{data_path}/src-train.csv',
                path_tgt = f'{data_path}/tgt-train.csv',
            ),
            valid = dict(
                path_src = f'{data_path}/src-valid.csv',
                path_tgt = f'{data_path}/tgt-valid.csv',
            )
        )
        
    data_params['data'] = data_path_dict
    return data_params

def write_preprocess_yaml(data_path, 
                        share_vocab:bool = False, 
                        weighted_sampling:bool =False, 
                        weight_folder_names: List = None, 
                        weights: List = None):
    preprocess_file_name = f'{data_path}/preprocess.yaml'
    
    preprocess_dict = dict(
        overwrite = True,
        n_sample = -1,
        )
    
    data_params = return_data_params(data_path, 
                                     share_vocab=share_vocab, 
                                     weighted_sampling=weighted_sampling, 
                                     weight_folder_names=weight_folder_names, 
                                     weights=weights)
    preprocess_dict.update(data_params)
    
    with open(preprocess_file_name, 'w') as f:
        yaml.dump(preprocess_dict, f)
    return

def write_training_yaml(data_path, 
                        dataset_name:str,  
                        share_vocab:bool = False, 
                        weighted_sampling:bool =False, 
                        weight_folder_names: List = None, 
                        weights: List = None,
                        n_gpu:int = 1):
    
    if weighted_sampling:
        training_file_name = f'{data_path}/train_weighted.yaml'

    else:
        training_file_name = f'{data_path}/train_single.yaml'
    
    training_dict = dict(    
        save_model = f'/rds-d2/user/wjm41/hpc-work/models/smi2wyk/{dataset_name}/model',
    )
    
    training_data_params = return_data_params(data_path, 
                                            share_vocab=share_vocab, 
                                            weighted_sampling=weighted_sampling, 
                                            weight_folder_names=weight_folder_names, 
                                            weights=weights)
    
    training_model_params = return_default_training_params(n_gpu=n_gpu)
     
    training_dict.update(training_data_params)
    training_dict.update(training_model_params)
    
    with open(training_file_name, 'w') as f:
        yaml.dump(training_dict, f)
    return

def write_tokenized_dataframe(df: pd.DataFrame,
                              data_path:str,
                              index: str,
                              tgt_col:str):
    Path(data_path).mkdir(parents=True, exist_ok=True)
    df.smi_tokenized.to_csv(f'{data_path}/src-{index}.csv', index=False, header=False)
    df.identifier.to_csv(f'{data_path}/id-{index}.csv', index=False, header=False)
    df[tgt_col].to_csv(f'{data_path}/tgt-{index}.csv', index=False, header=False)
    return 

def write_train_val_test(df: pd.DataFrame,
                         dataset_name:str, 
                         tgt_col: str = 'tgt', 
                         share_vocab:bool = False, 
                         weighted_sampling:bool =False, 
                         weight_folder_names: List = None, 
                         weights: List = None,
                         test_split:bool = True,
                         n_gpu:int = 1):

    df = df.drop_duplicates(subset=['smiles'])
    
    if test_split:
        df_train_and_val = df.sample(frac=0.9, random_state=42)
        df_test = df.drop(df_train_and_val.index)
    else:
        df_train_and_val = df.copy()

    df_train = df_train_and_val.sample(frac=0.9, random_state=42)
    df_valid = df_train_and_val.drop(df_train.index)

    data_dir = str(Path(os.getcwd()).parents[0])+'/data'
    
    data_path = f'{data_dir}/{dataset_name}'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    write_tokenized_dataframe(df_train, data_path, 'train', tgt_col)
    write_tokenized_dataframe(df_valid, data_path, 'valid', tgt_col)
    
    if test_split:
        write_tokenized_dataframe(df_test, data_path, 'test', tgt_col)
    
    write_preprocess_yaml(data_path,
                          share_vocab = share_vocab, 
                        weighted_sampling = weighted_sampling,
                        weight_folder_names = weight_folder_names, 
                        weights = weights)
    write_training_yaml(data_path, 
                        dataset_name = dataset_name, 
                        share_vocab=share_vocab, 
                        weighted_sampling=weighted_sampling, 
                        weight_folder_names = weight_folder_names, 
                        weights = weights,
                        n_gpu = n_gpu)
    return

def submit_training_job(dataset:str):
    data_dir = f'/home/wjm41/ml_physics/smi2wyk/data/{dataset}'
    log_dir = f'/home/wjm41/ml_physics/smi2wyk/runs/{dataset}'

    preprocess_script = f'onmt_build_vocab -config {data_dir}/preprocess.yaml'
    train_script = f'onmt_train -config {data_dir}/train_single.yaml -tensorboard True -tensorboard_log_dir {log_dir}'
    script = f'{preprocess_script}\n{train_script}'

    file_name = f'subm_train_{dataset}'
    run_time = '8:00:00'
    current_dir = os.getcwd()
    output_name = f'{current_dir}/{file_name}.out'

    write_slurm_script(job_name=f'{file_name}',
                    run_time=f'{run_time}',
                    output_name=output_name,
                    script=script,
                    file_name=file_name,
                    email=True,
                    conda_env='DebiasedMT',
                    gpu=True
                    )

    print(f"Submitted transformer training jobs on {dataset}")

    return file_name

def submit_translation_job(dataset:str, step:int, beam_size:int = 10):
    data_dir = f'/home/wjm41/ml_physics/smi2wyk/data/{dataset}'

    script_dir = '/home/wjm41/ml_physics/smi2wyk/smi2wyk'
    model_dir = f'/rds-d2/user/wjm41/hpc-work/models/smi2wyk'
    model_path = f'{model_dir}/{dataset}/model_step_{step}.pt'
    pred_name = f'{data_dir}/pred_step_{step}.txt'  
    
    translate_script = f'onmt_translate -model {model_path} -src {data_dir}/src-test.csv -output {pred_name} -n_best {beam_size} -beam_size {beam_size} -gpu 0'
    score_script = f'python {script_dir}/score_predictions.py -targets {data_dir}/tgt-test.csv -beam_size {beam_size} -predictions {pred_name}'
    script = f'{translate_script}\n{score_script}'

    file_name = f'subm_test_{dataset}_{step}'
    run_time = '1:00:00'
    current_dir = os.getcwd()
    output_name = f'{current_dir}/{file_name}.out'

    write_slurm_script(job_name=f'{file_name}',
                    run_time=f'{run_time}',
                    output_name=output_name,
                    script=script,
                    file_name=file_name,
                    email=True,
                    conda_env='DebiasedMT',
                    gpu=True
                    )

    print(f"Submitted translation & scoring jobs on {dataset}")

    return file_name

#TODO check if sbatch works as importable python function

# TODO - add weighed sampling by space group