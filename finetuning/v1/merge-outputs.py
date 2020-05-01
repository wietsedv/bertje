""" Merge true labels and predicted labels to a single file for conlleval evaluation """

import argparse
import os


def read_file(path, filename):
    path = os.path.join(path, filename)
    items = []
    with open(path) as f:
        for line in f:
            items.append(line.strip().split('\t'))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True, help="Base path")
    args = parser.parse_args()

    for name in ['train', 'dev', 'test']:
        true_name = name + '.true.tsv'
        pred_name = name + '.pred.tsv'
        out_name = name + '.tsv'

        true_items = read_file(args.path, true_name)
        print('▶ Read {} lines from {}'.format(len(true_items), true_name))
        pred_items = read_file(args.path, pred_name)
        print('▶ Read {} lines from {}'.format(len(pred_items), pred_name))

        with open(os.path.join(args.path, out_name), 'w') as f:
            for true, pred in zip(true_items, pred_items):
                if len(true) > 1:
                    f.write(true[0] + ' ' + true[1] + ' ' + pred[1] + '\n')
                else:
                    f.write('\n')
        print('▶ Saved to {}'.format(out_name))


if __name__ == '__main__':
    main()
