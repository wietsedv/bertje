import os
from glob import glob
from argparse import ArgumentParser
import random
from xml.etree import ElementTree


def read_iob(path):
    data = [[]]
    with open(path) as f:
        for line in f:
            line = line.strip()

            # New sentence
            if len(line) == 0:
                if len(data[-1]) > 0:
                    data.append([])
                continue

            # Add token to sentence
            tok, label = line.split('\t')
            data[-1].append((tok, label))

    if len(data[-1]) == 0:
        del data[-1]
    return data


def read_mwe(path):
    data = [[]]
    with open(path, encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if line[:4].lower() == '<doc' or line[:5].lower() == '<sent':
                continue

            # New sentence
            if line == '</sent>' or line == '</doc>':
                if len(data[-1]) > 0:
                    data.append([])
                continue

            # Add token to sentence
            tok, label = line.split('\t')[:2]
            data[-1].append((tok, label))

    if len(data[-1]) == 0:
        del data[-1]
    return data


def read_srl_xml(path):
    def parse_node(node):
        begin, end = int(node.attrib['begin']), int(node.attrib['end'])
        preds = ['O' for _ in range(end - begin)]
        mods = ['O' for _ in range(end - begin)]

        if 'pb' in node.attrib:
            pb = node.attrib['pb']

            if pb == 'rel':
                preds[0] = 'B-pred'
                for i in range(1, len(preds)):
                    preds[i] = 'I-pred'
            elif pb[3].isdigit():  # Predicate / Argument
                preds[0] = 'B-arg'
                for i in range(1, len(preds)):
                    preds[i] = 'I-arg'
            elif pb[3] == 'M':  # Modifier
                mod = pb[5:].lower()
                mods[0] = 'B-' + mod
                for i in range(1, len(mods)):
                    mods[i] = 'I-' + mod

        for childnode in node:
            subpreds, submods, childbegin = parse_node(childnode)

            childbegin -= begin
            for i in range(len(subpreds)):
                if preds[childbegin + i] == 'O' and subpreds[i] != 'O':
                    preds[childbegin + i] = subpreds[i]
                if mods[childbegin + i] == 'O' and submods[i] != 'O':
                    mods[childbegin + i] = submods[i]
        return preds, mods, begin

    tree = ElementTree.parse(path)
    root = tree.getroot()

    tokens = root.find('sentence').text.split()
    preds, mods, _ = parse_node(root.find('node'))
    assert len(tokens) == len(preds)
    assert len(tokens) == len(mods)

    sent_preds = [(t, l) for t, l in zip(tokens, preds)]
    sent_mods = [(t, l) for t, l in zip(tokens, mods)]
    return sent_preds, sent_mods


def read_ste_xml(path):
    def parse_spt_nodes(parentnode, labels):
        for node in parentnode.findall('node'):
            label = None

            geo = node.find('geo')
            temp = node.find('temp')
            spat = node.find('spat')
            if geo is not None:
                if 'rel' in geo.attrib:
                    label = 'geo-rel'  # + geo.attrib['rel']
                elif 'type' in geo.attrib:
                    label = 'geo-location'   # + geo.attrib['type']
            elif spat is not None:
                if 'rel' in spat.attrib:
                    label = 'spat-rel'   # + spat.attrib['rel']
            elif temp is not None:
                if 'ta' in temp.attrib and temp.attrib['ta']:
                    label = 'temp-form-' + temp.attrib['ta']
                elif 'rel' in temp.attrib:
                    label = 'temp-rel'  # + temp.attrib['rel']
                elif 'type' in temp.attrib and temp.attrib['type'] in {'cal', 'clock'}:
                    label = 'temp-type-' + temp.attrib['type']

            if label and label.replace('-', '').isalpha():
                begin, end = int(node.attrib['begin']), int(node.attrib['end'])
                for i in range(begin, end):
                    labels[i] = label

            for child in node.findall('node'):
                parse_spt_nodes(child, labels)

    def parse_temp_nodes(parentnode, labels):
        for node in parentnode.findall('node'):
            temp = node.find('temp')
            if temp is not None and 'ta' in temp.attrib and temp.attrib['ta']:
                label = temp.attrib['ta']

                begin, end = int(node.attrib['begin']), int(node.attrib['end'])
                for i in range(begin, end):
                    labels[i] = label

            for child in node.findall('node'):
                parse_temp_nodes(child, labels)

    spt_data, temp_data = [], []

    try:
        tree = ElementTree.parse(path)
        root = tree.getroot()
    except ElementTree.ParseError:
        with open(path) as f:
            lines = list(f.readlines())
        if lines[-1].strip() != '</treebank>':
            lines.append('</treebank>')
        root = ElementTree.fromstringlist(lines)

    for sentnode in root:
        tokens = sentnode.find('sentence').text.split()

        spt_labels = ['O' for _ in range(len(tokens))]
        parse_spt_nodes(sentnode, spt_labels)
        spt_data.append(list(zip(tokens, spt_labels)))

        temp_labels = ['O' for _ in range(len(tokens))]
        parse_temp_nodes(sentnode, temp_labels)
        temp_data.append(list(zip(tokens, temp_labels)))

    return spt_data, temp_data


def prepare_ner(sonar_path):
    filepaths = os.path.join(sonar_path, 'NE', 'SONAR_1_NE', 'IOB', '*.iob')
    data = []
    for filepath in glob(filepaths):
        data.append(read_iob(filepath))
    return data


def prepare_pos(sonar_path):
    filepaths = os.path.join(sonar_path, 'POS', 'SONAR_1_POS', '*.mwe')
    data, data_fine = [], []
    for filepath in glob(filepaths):
        sents = read_mwe(filepath)
        pos_sents, fine_sents = [], []
        for sent in sents:
            valid = True

            pos_sent, fine_sent = [], []
            for tok, full_label in sent:
                label = full_label.split('(')[0]
                if not label.isalpha() or not label.isupper() or label in {'ASJ', 'ADH'} or full_label[-1] != ')':
                    valid = False
                    break
                pos_sent.append((tok, label.lower()))
                fine_sent.append((tok, full_label.lower()))

            if valid:
                pos_sents.append(pos_sent)
                fine_sents.append(fine_sent)
        data.append(pos_sents)
        data_fine.append(fine_sents)
    return data, data_fine


def prepare_srl(sonar_path):
    dirpaths = os.path.join(sonar_path, 'SRL', 'SONAR_1_SRL', 'MANUAL500', '*')
    preds, mods = [], []
    for dirpath in glob(dirpaths):
        sents_preds, sents_mods = [], []
        for filepath in glob(os.path.join(dirpath, '*.xml')):
            sent_preds, sent_mods = read_srl_xml(filepath)
            sents_preds.append(sent_preds)
            sents_mods.append(sent_mods)
        preds.append(sents_preds)
        mods.append(sents_mods)
    return preds, mods


def prepare_spt(sonar_path):
    filepaths = os.path.join(sonar_path, 'SPT', 'SONAR_1_STEx', '*.tempcor.utf8')
    spt_data, temp_data = [], []
    for filepath in glob(filepaths):
        spt, temp = read_ste_xml(filepath)
        spt_data.append(spt)
        temp_data.append(temp)
    return spt_data, temp_data


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
        print('{:20} {:>8} ({:.2f}%)'.format(label, count, count / total * 100))
    print('')
    return label_counts.keys()


def save_data(data, out_path, name, dev_size=0.1, test_size=0.15):
    if len(data) == 0:
        print('No data for {}'.format(name))
        return

    os.makedirs(os.path.join(out_path, name))

    random.shuffle(data)

    # Split train/dev/test
    num_dev, num_test = int(len(data) * dev_size), int(len(data) * test_size)
    num_train = len(data) - num_dev - num_test
    train, dev, test = data[:num_train], data[num_train:num_train + num_dev], data[-num_test:]

    # Flatten grouped sentences
    train = [sent for sents in train for sent in sents]
    dev = [sent for sents in dev for sent in sents]
    test = [sent for sents in test for sent in sents]

    # Write to files
    label_set = []
    label_set.extend(write_tsv(os.path.join(out_path, name, 'train.tsv'), train))
    label_set.extend(write_tsv(os.path.join(out_path, name, 'dev.tsv'), dev))
    label_set.extend(write_tsv(os.path.join(out_path, name, 'test.tsv'), test))
    print(len(set(label_set)), 'labels total')

    total = len(train) + len(dev) + len(test)
    print('{}: Train={:.2f}, Dev={:.2f}, Test={:.2f}'.format(
        name, len(train) / total, len(dev) / total, len(test) / total))


def main():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", dest="in_path", required=True,
                        help="Path to SONAR1 directory in SoNaR corpus v1.2.1", metavar="FILE")
    parser.add_argument("-o", dest="out_path", default='sonar', help="Target location", metavar="FILE")
    parser.add_argument("--seed", dest="seed", default=3242, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print('provide a valid input path')
        return
    if os.path.exists(args.out_path):
        print('output path already exists')
        return

    random.seed(args.seed)

    print(' > Preparing NER data')
    save_data(prepare_ner(args.in_path), args.out_path, 'ner')

    print(' > Preparing POS data')
    pos, pos_fine = prepare_pos(args.in_path)
    save_data(pos, args.out_path, 'pos')
    save_data(pos_fine, args.out_path, 'pos-fine')

    print(' > Preparing SRL data')
    preds, mods = prepare_srl(args.in_path)
    save_data(preds, args.out_path, 'srl-preds')
    save_data(mods, args.out_path, 'srl-mods')

    print(' > Preparing SPT data')
    spt_data, temp_data = prepare_spt(args.in_path)
    save_data(spt_data, args.out_path, 'spt')
    save_data(temp_data, args.out_path, 'spt-temp')

    # print(' > Preparing COREF data')
    # save_data(prepare_cor(args.in_path), args.out_path, 'cor')


if __name__ == '__main__':
    main()
