import argparse
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


def read_labels(filename):
    labels = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            _, label = line.split('\t')
            labels.append(label)
    return labels


def compare_labels(true_labels, pred_labels):
    true_set = set(true_labels)
    pred_set = set(pred_labels)

    print('\n▶ Label usage:')
    print('  ~ Used in both: {}'.format(true_set | pred_set))
    print('  ~ Extra in true: {}'.format(true_set - pred_set))
    print('  ~ Extra in pred: {}'.format(pred_set - true_set))

    print('\n▶ Raw counts:')
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)

    sorted_labels = sorted(true_counts, key=true_counts.get, reverse=True) + sorted(pred_set - true_set)
    print('\tTrue\tPred\tDiff')
    for label in sorted_labels:
        diff = pred_counts[label] - true_counts[label]
        direction = '+' if diff > 0 else '-' if diff < 0 else ' '
        if diff < 0:
            diff = -diff
        print('{}\t{}\t{}\t{}{:4}'.format(label, true_counts[label], pred_counts[label], direction, diff))

    print('\n▶ Confusion matrix:')
    sorted_labels = sorted(true_set | pred_set)
    padded_labels = [lab + ' ' * (4 - len(lab)) if len(lab) < 8 else lab for lab in sorted_labels]
    cm = confusion_matrix(true_labels, pred_labels, labels=sorted_labels)
    print('         \tpredicted:')
    print('         \t' + '\t'.join(padded_labels))
    for i in range(len(cm)):
        prefix = 'true: ' if i == 0 else ' ' * 6
        prefix += padded_labels[i]
        print(prefix + '\t' + '\t'.join([str(n) for n in cm[i]]))

    print('\n▶ Classification report:')
    print(classification_report(true_labels, pred_labels, digits=3))

    print('\n▶ Classification report w/o O label:')
    print(classification_report(true_labels, pred_labels, labels=list(true_set - {'O'}), digits=3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, type=str, required=True, help="Base path")
    parser.add_argument("--name", default=None, type=str, required=True, help="File name [train,dev,test]")
    args = parser.parse_args()

    true_path = os.path.join(args.path, args.name + '.true.tsv')
    pred_path = os.path.join(args.path, args.name + '.pred.tsv')

    true_labels = read_labels(true_path)
    print('▶ Read true labels from {}'.format(true_path))

    pred_labels = read_labels(pred_path)
    print('▶ Read pred labels from {}'.format(pred_path))

    if len(true_labels) != len(pred_labels):
        print('True and pred file do not have the same amount of labels ({} and {})'.format(
            len(true_labels), len(pred_labels)))
        exit(-1)

    print('\nFull label comparison:')
    compare_labels(true_labels, pred_labels)

    if set([lab[0] for lab in true_labels]) == {'B', 'I', 'O'}:
        true_label_cats = [lab if lab == 'O' else lab[2:] for lab in true_labels]
        pred_label_cats = [lab if lab == 'O' else lab[2:] for lab in pred_labels]
        print('\nBIO category comparison:')
        compare_labels(true_label_cats, pred_label_cats)

    if 'O' in true_labels:
        true_label_binary = ['O' if lab == 'O' else 'X' for lab in true_labels]
        pred_label_binary = ['O' if lab == 'O' else 'X' for lab in pred_labels]
        print('\nBinary comparison:')
        compare_labels(true_label_binary, pred_label_binary)


if __name__ == '__main__':
    main()
