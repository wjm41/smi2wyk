from __future__ import division, unicode_literals
import argparse
import pandas as pd


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


def main(opt):
    if opt.ignore_last_number:
        with open(opt.targets, 'r') as f:
            targets = [''.join(line.strip().split(' ')[:-1])
                       for line in f.readlines()]
    else:
        with open(opt.targets, 'r') as f:
            targets = [''.join(line.strip().split(' '))
                       for line in f.readlines()]
    predictions = [[] for i in range(opt.beam_size)]

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']

    with open(opt.predictions, 'r') as f:
        # print(len(f.readlines()))
        test_df = test_df.iloc[:int(len(f.readlines())/opt.beam_size)]
    print(len(test_df))
    total = len(test_df)
    with open(opt.predictions, 'r') as f:
        # print(len(f.readlines()))
        if opt.ignore_last_number:
            for i, line in enumerate(f.readlines()):

                predictions[i % opt.beam_size].append(
                    ''.join(line.strip().split(' ')[:-1]))
        else:
            for i, line in enumerate(f.readlines()):

                predictions[i % opt.beam_size].append(
                    ''.join(line.strip().split(' ')))

    for i, preds in enumerate(predictions):
        test_df['prediction_{}'.format(i + 1)] = preds

    test_df['rank'] = test_df.apply(lambda row: get_rank(
        row, 'prediction_', opt.beam_size), axis=1)

    correct = 0

    for i in range(1, opt.beam_size+1):
        correct += (test_df['rank'] == i).sum()

        print('Top-{}: {:.1f}%'.format(i, correct / total * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                        help='Show % of invalid SMILES')
    parser.add_argument('-predictions', type=str, default="",
                        help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="",
                        help="Path to file containing targets")
    parser.add_argument('-ignore_last_number', action='store_true',
                        help="Whether or not to ignore the last number.")
    opt = parser.parse_args()
    main(opt)
