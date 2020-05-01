import os
from argparse import ArgumentParser
import numpy as np


def clean_doc(doc_path):
    with open(doc_path) as f:
        lines = [line.strip() for line in f.readlines()]
        text = ' '.join(lines)

    return text


def prepare_examples(in_path, label):
    examples = []
    for filename in os.listdir(in_path):
        if not filename.endswith('.txt'):
            continue

        doc_path = os.path.join(in_path, filename)
        examples.append((label, clean_doc(doc_path)))

    return examples


def save_data(examples, out_path):
    with open(out_path, 'w') as f:
        for label, sent in examples:
            f.write('{}\t{}\n'.format(label, sent))


def read_truth(path, prob_names):
    truth = {n: '?' for n in prob_names}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                prob_name, label = line.split()
                truth[prob_name] = label
    return truth


def main():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", dest="in_path", required=True,
                        help="Path to 110kDRBD dataset directory", metavar="FILE")
    parser.add_argument("-o", dest="out_path", default='110kdrbd', help="Target location", metavar="FILE")
    parser.add_argument("-n", dest="name", default='train', help="Target name")
    parser.add_argument("--seed", dest="seed", default=7892, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print('provide a valid input path')
        return

    np.random.seed(args.seed)

    os.makedirs(args.out_path, exist_ok=True)

    print('\n > Preparing 110kDRBD data')
    examples = []
    for label in ['pos', 'neg']:
        in_dir = os.path.join(args.in_path, args.name, label)
        examples.extend(prepare_examples(in_dir, label))

    np.random.shuffle(examples)

    out_path = os.path.join(args.out_path, args.name + '.tsv')
    save_data(examples, out_path)
    print('Wrote {} examples to {}'.format(len(examples), out_path))


if __name__ == '__main__':
    main()
