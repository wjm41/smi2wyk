from typing import List

import pandas as pd
import fire

VALID_BRAVAIS_LATTICES = ['aP', 'mP', 'mS', 'mA', 'mB', 'mC','oP', 'oS', 'oF', 'oI', 'tP', 'tI', 'hP','hR','cP', 'cF','cI']


def get_rank(row, base, max_rank, tgt_column='target'):
    for i in range(1, max_rank+1):
        if row[tgt_column] == row['{}{}'.format(base, i)]:
            return i
    return 0

def read_targets(targets:str = 'targets.txt',
                 ignore_last_number: bool = False):
    if ignore_last_number:
        with open(targets, 'r') as f:
            targets = [''.join(line.strip().split(' ')[:-1])
                       for line in f.readlines()]
    else:
        with open(targets, 'r') as f:
            targets = [''.join(line.strip().split(' '))
                       for line in f.readlines()]
            
    return targets

def read_preds(preds:str = 'predictions.txt',
               beam_size:int = 5,
              ignore_last_number: bool = False):
    
    predictions = [[] for i in range(beam_size)]

    with open(preds, 'r') as f:

        if ignore_last_number:
            for i, line in enumerate(f.readlines()):

                predictions[i % beam_size].append(
                    ''.join(line.strip().split(' ')[:-1]))
        else:
            for i, line in enumerate(f.readlines()):

                predictions[i % beam_size].append(
                    ''.join(line.strip().split(' ')))
                
    return predictions

def score_accuracy(targets:List, predictions:List, beam_size:int = 5):
    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    
    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds

    test_df['rank'] = test_df.apply(lambda row: get_rank(
        row, 'prediction_', beam_size), axis=1)

    correct = 0
    for i in range(1, beam_size+1):
        correct += (test_df['rank'] == i).sum()

        print('Top-{}: {:.1f}%'.format(i, correct / len(test_df) * 100))
    return test_df

def read_bravais(aflow_string):
    return aflow_string.split(':')[1][:2]

def score_bravais(targets:List, predictions:List, beam_size:int = 5):
    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    test_df['bravais'] = test_df.target.apply(read_bravais)
    print(test_df.bravais.value_counts())
    
    for i, preds in enumerate(predictions):
        test_df[f'prediction_{i+1}'] = preds
        test_df[f'bravais_{i+1}'] = test_df[f'prediction_{i+1}'].apply(read_bravais)
        
    test_df['rank'] = test_df.apply(lambda row: get_rank(
        row, 'bravais_', beam_size, tgt_column='bravais'), axis=1)

    correct = 0
    for i in range(1, beam_size+1):
        correct += (test_df['rank'] == i).sum()
        valid = len(test_df[test_df[f'bravais_{i}'].isin(VALID_BRAVAIS_LATTICES)])
        print(f'Top-{i}: {100*correct/len(test_df):.1f}%, valid bravais lattices: {100*valid/len(test_df):.1f}%')
    return test_df

def score_predictions(beam_size:int = 5,
                      tgt_path:str = 'targets.txt',
                      pred_path:str = 'predictions.txt',
                      inds:List = None,
                      ignore_last_number: bool = False,
                      bravais:bool = False):
    
    targets = read_targets(tgt_path, ignore_last_number)
    with open(pred_path, 'r') as f:
        targets = targets[:int(len(f.readlines())/beam_size)]

    predictions = read_preds(pred_path, beam_size, ignore_last_number)
    
    if inds is not None:
        old_len = len(targets)
        targets = [targets[i] for i in inds]
        predictions = [[predictions[j][i] for i in inds] for j in range(beam_size)]
        print(f'Number of datapoints in test subset: {len(targets)} ({100*len(targets)/old_len:.2f}%)')

    else:   
        print(f'Number of datapoints in test set: {len(targets)}')
    if bravais:
        test_df = score_bravais(targets, predictions, beam_size)
    else:
        test_df = score_accuracy(targets, predictions, beam_size)
    return test_df

def score_predictions_by_frequency(beam_size:int = 5,
                                    tgt_path:str = 'targets.txt',
                                    src_path:str = 'tgt-train.txt',
                                    inds:List = None,
                                    ignore_last_number: bool = False):
    
    targets = read_targets(tgt_path, ignore_last_number)

    targets_in_training_set = read_targets(src_path, ignore_last_number)
    occurence_count = Counter(targets_in_training_set) 
    top_n_most_frequent = occurence_count.most_common(n=beam_size)
    predictions = [[str(top_n_most_frequent[i][0])]*len(targets) for i in range(beam_size)]
    # print(predictions)
    if inds is not None:
        old_len = len(targets)
        targets = [targets[i] for i in inds]
        predictions = [[predictions[j][i] for i in inds] for j in range(beam_size)]
        print(f'Number of datapoints in test subset: {len(targets)} ({100*len(targets)/old_len:.2f}%)')

    else:   
        print(f'Number of datapoints in test set: {len(targets)}')
    test_df = score_accuracy(targets, predictions, beam_size)
    return test_df



if __name__ == "__main__":
    fire.Fire(score_predictions)
