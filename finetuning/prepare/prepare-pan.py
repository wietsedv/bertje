import os
from argparse import ArgumentParser
import random
import json
import nltk


def split_doc(doc_path, author, n_sents=3):
    with open(doc_path, encoding='utf-8-sig') as f:
        text = list(f.readlines())[0].strip()

    sents = nltk.sent_tokenize(text, 'dutch')
    examples = []
    for i in range(len(sents) - (n_sents - 1)):
        examples.append((author, ' '.join(sents[i:i+n_sents])))
    return examples


def prepare_problem(in_path, prob_name, truth):
    prob_path = os.path.join(in_path, prob_name)

    examples = []
    for i in range(1, 6):
        doc_path = os.path.join(prob_path, 'known0{}.txt'.format(i))
        if not os.path.exists(doc_path):
            break
        examples.extend(split_doc(doc_path, prob_name))

    unk_path = os.path.join(prob_path, 'unknown.txt')
    unk_examples = split_doc(unk_path, prob_name + '|' + truth)
    return examples, unk_examples


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
                        help="Path to PAN dataset directory", metavar="FILE")
    parser.add_argument("-o", dest="out_path", default='pan', help="Target location", metavar="FILE")
    parser.add_argument("--seed", dest="seed", default=7892, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print('provide a valid input path')
        return
    # if os.path.exists(args.out_path):
    #     print('output path already exists')
    #     return

    with open(os.path.join(args.in_path, 'contents.json')) as f:
        conf = json.load(f)

    truth = read_truth(os.path.join(args.in_path, 'truth.txt'), conf['problems'])

    print('Language:', conf['language'])
    print('Number of problems:', len(conf['problems']))

    random.seed(args.seed)

    os.makedirs(args.out_path, exist_ok=True)

    print('\n > Preparing PAN data')
    examples, unks = [], []
    for prob_name in conf['problems']:
        prob_examples, prob_unk = prepare_problem(args.in_path, prob_name, truth[prob_name])
        examples.extend(prob_examples)
        unks.extend(prob_unk)

    train_path = os.path.join(args.out_path, 'train.tsv')
    save_data(examples, train_path)
    print('Wrote {} examples to {}'.format(len(examples), train_path))

    test_path = os.path.join(args.out_path, 'test.tsv')
    save_data(unks, test_path)
    print('Wrote {} examples to {}'.format(len(unks), test_path))


if __name__ == '__main__':
    main()
