
import os 
import yaml

from pathlib import Path
import re


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


def write_default_preprocess_yaml(data_path):
    preprocess_file_name = f'{data_path}/preprocess.yaml'
    
    preprocess_dict = dict(
        save_data = f'{data_path}',
        src_vocab = f'{data_path}/vocab.src',
        tgt_vocab = f'{data_path}/vocab.tgt',
        overwrite = True,
        n_sample = -1,
        share_vocab = False,
        data = dict(
            train = dict(
                path_src = f'{data_path}/src-train.csv',
                path_tgt = f'{data_path}/tgt-train.csv',
            ),
            valid = dict(
                path_src = f'{data_path}/src-valid.csv',
                path_tgt = f'{data_path}/tgt-valid.csv',
            )
        )
    )
    
    with open(preprocess_file_name, 'w') as f:
        yaml.dump(preprocess_dict, f)
    return

def write_default_training_yaml(data_path, dataset_name):
    training_file_name = f'{data_path}/train_single.yaml'
    
    training_dict = dict(
        save_data = f'{data_path}',
        src_vocab = f'{data_path}/vocab.src',
        tgt_vocab = f'{data_path}/vocab.tgt',

        share_vocab = False,
        data = dict(
            train = dict(
                path_src = f'{data_path}/src-train.csv',
                path_tgt = f'{data_path}/tgt-train.csv',
            ),
            valid = dict(
                path_src = f'{data_path}/src-valid.csv',
                path_tgt = f'{data_path}/tgt-valid.csv',
            )
        ),
        
        save_model = f'/rds-d2/user/wjm41/hpc-work/models/smi2wyk/{dataset_name}/model',
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

        world_size = 1,
        gpu_ranks = [0],
    )
    
    with open(training_file_name, 'w') as f:
        yaml.dump(training_dict, f)
    return

def write_train_val_test(df, dataset_name, tgt_col: str = 'tgt'):

    df = df.drop_duplicates(subset=['smiles'])
    df_train_and_val = df.sample(frac=0.9, random_state=42)
    df_test = df.drop(df_train_and_val.index)

    df_train = df_train_and_val.sample(frac=0.9, random_state=42)
    df_valid = df_train_and_val.drop(df_train.index)

    data_dir = str(Path(os.getcwd()).parents[0])+'/data'
    
    data_path = f'{data_dir}/{dataset_name}'
    Path(data_path).mkdir(parents=True, exist_ok=True)
    df_train.smi_tokenized.to_csv(f'{data_path}/src-train.csv', index=False, header=False)
    df_train.identifier.to_csv(f'{data_path}/id-train.csv', index=False, header=False)
    df_train[tgt_col].to_csv(f'{data_path}/tgt-train.csv', index=False, header=False)

    df_valid.smi_tokenized.to_csv(f'{data_path}/src-valid.csv', index=False, header=False)
    df_valid.identifier.to_csv(f'{data_path}/id-valid.csv', index=False, header=False)
    df_valid[tgt_col].to_csv(f'{data_path}/tgt-valid.csv', index=False, header=False)

    df_test.smi_tokenized.to_csv(f'{data_path}/src-test.csv', index=False, header=False)
    df_test.identifier.to_csv(f'{data_path}/id-test.csv', index=False, header=False)
    df_test[tgt_col].to_csv(f'{data_path}/tgt-test.csv', index=False, header=False)
    
    write_default_preprocess_yaml(data_path)
    write_default_training_yaml(data_path, dataset_name = dataset_name)
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

    !sbatch {file_name}
    return

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

    !sbatch {file_name}
    return

#TODO check if sbatch works as importable python function

# TODO - add weighed sampling by space group