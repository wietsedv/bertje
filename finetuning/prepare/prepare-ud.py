import os
from argparse import ArgumentParser
import random


def read_conllu(path):
    data = [[]]
    with open(path) as f:
        for line in f:
            line = line.strip()

            # New sentence
            if len(line) == 0:
                if len(data[-1]) > 0:
                    data.append([])
                continue

            if line[0] == '#':
                continue

            parts = line.split()
            tok = parts[1]
            label = parts[3].lower()
            data[-1].append((tok, label))

    if len(data[-1]) == 0:
        del data[-1]
    return data


def prepare_ud(conll_path):
    train_path = os.path.join(conll_path, 'nl_lassysmall-ud-train.conllu')
    dev_path = os.path.join(conll_path, 'nl_lassysmall-ud-dev.conllu')
    test_path = os.path.join(conll_path, 'nl_lassysmall-ud-test.conllu')

    train = read_conllu(train_path)
    dev = read_conllu(dev_path)
    test = read_conllu(test_path)

    return train, dev, test


def write_tsv(path, data):
    label_counts = {}

    with open(path, 'w') as f:
        for sent in data:
            for tok, label in sent:
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
                f.write('{}\t{}\n'.format(tok, label))
            f.write('\n')

    print('Labels in {} ({} labels):'.format(path, len(label_counts)))
    total = sum(label_counts.values())
    for label in sorted(label_counts, key=label_counts.get, reverse=True):
        count = label_counts[label]
        print('{:10} {:>8} ({:.2f}%)'.format(label, count, count / total * 100))
    print('')


def save_data(data, out_path):
    if len(data) == 0:
        print('No data found')
        return

    os.makedirs(os.path.join(out_path, 'pos'))

    train, dev, test = data

    # Write to files
    write_tsv(os.path.join(out_path, 'pos', 'train.tsv'), train)
    write_tsv(os.path.join(out_path, 'pos', 'dev.tsv'), dev)
    write_tsv(os.path.join(out_path, 'pos', 'test.tsv'), test)

    total = len(train) + len(dev) + len(test)
    print('NER: Train={:.2f}, Dev={:.2f}, Test={:.2f}'.format(len(train) / total, len(dev) / total, len(test) / total))


def main():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", dest="in_path", required=True,
                        help="Path to Universal Dependencies UD_Dutch-LassySmall directory", metavar="FILE")
    parser.add_argument("-o", dest="out_path", default='ud-lassysmall', help="Target location", metavar="FILE")
    parser.add_argument("--seed", dest="seed", default=7892, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print('provide a valid input path')
        return
    if os.path.exists(args.out_path):
        print('output path already exists')
        return

    random.seed(args.seed)

    print(' > Preparing NER data')
    save_data(prepare_ud(args.in_path), args.out_path)


if __name__ == '__main__':
    main()
