import os
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import json

sns.set(style='whitegrid')
task = None


def save_plot(name, title=None, **kwargs):
    # plt.figure(**kwargs)
    # if title:
    #     plt.title(title)
    # plt.show()
    plt.savefig(f'plots/{task}/{name}.png', bbox_inches='tight')
    plt.close()


def load_predictions(path):
    layers_preds = None
    with open(path) as f:
        for line in f:
            preds = line.rstrip().split('\t')

            if layers_preds is None:
                layers_preds = [[] for _ in range(len(preds))]

            for i in range(len(preds)):
                layers_preds[i].append(preds[i])

    label_names = sorted(set(layers_preds[0]))
    return label_names, layers_preds


def only_with_mistakes(model_layers_preds, tokens, sents):
    n_items = len(model_layers_preds[0][0])
    n_layers = len(model_layers_preds[0])

    n_diff = 0
    model_layers_preds_mistakes = [[[] for _ in range(len(layers_preds))] for layers_preds in model_layers_preds]
    tokens_subset, sents_subset = [], []

    for n in range(n_items):
        preds1 = [model_layers_preds[0][i][n] for i in range(n_layers)]
        preds2 = [model_layers_preds[1][i][n] for i in range(n_layers)]

        if len(set(preds1)) > 1 or len(set(preds2)) > 1:
            n_diff += 1
            tokens_subset.append(tokens[n])
            sents_subset.append(sents[n])

            for i in range(n_layers):
                model_layers_preds_mistakes[0][i].append(preds1[i])
                model_layers_preds_mistakes[1][i].append(preds2[i])

    print(f'{n_diff:,}/{n_items:,} ({n_diff/n_items:.3f}) tokens with some mistakes\n')
    return model_layers_preds_mistakes, tokens_subset, sents_subset


def aggregate_data(label_names, layers_preds):
    orig_data, score_data = [], []

    true_labels = layers_preds[0]
    for lab in label_names:
        orig_data.append((lab, sum(lab2 == lab for lab2 in true_labels)))

    for layer_nr in range(1, len(layers_preds)):
        pred_labels = layers_preds[layer_nr]
        for i in range(len(label_names)):
            score = f1_score(true_labels, pred_labels, labels=[label_names[i]], average=None).item()
            score_data.append((str(layer_nr - 1).zfill(2), label_names[i], score))

    orig_data = pd.DataFrame(orig_data, columns=['tag', 'size'])
    score_data = pd.DataFrame(score_data, columns=['layer', 'tag', 'score'])
    return orig_data, score_data


def summarize_data(orig_data, score_data):
    print(orig_data)
    print(score_data.describe())


def plot_tag_distribution(orig_data: pd.DataFrame, full_orig_data: pd.DataFrame):
    orig_data['subset'] = 'filtered'
    full_orig_data['subset'] = 'full'
    data = pd.concat([orig_data, full_orig_data], ignore_index=True)
    plt.figure(figsize=(6, 3))
    sns.barplot(x='tag', y='size', hue='subset', data=data)
    plt.xticks(rotation=90)
    save_plot('preds-tag-distribution', 'POS tag distribution')


def plot_data(orig_datas, score_datas, model_names):
    score_datas[0]['model'] = model_names[0]
    score_datas[1]['model'] = model_names[1]
    data = pd.concat(score_datas, ignore_index=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x='tag', y='score', hue='model', data=data)
    plt.xticks(rotation=90)
    save_plot(f'preds-tag-scores', 'layerwise macro averaged scores per tag')

    for orig_data, score_data, model_name in zip(orig_datas, score_datas, model_names):
        pos_groups = {
            # 'closed classes': ['adp', 'aux', 'cconj', 'det', 'num', 'part', 'pron', 'sconj'],
            'closed classes': ['pron', 'det', 'sconj', 'aux'],
            # 'closed classes (2/2)': ['num', 'adp', 'cconj', 'part'],
            'open classes': ['adj', 'adv', 'intj', 'noun', 'propn', 'verb'],
            'open classes w/o verb': ['adj', 'adv', 'intj', 'noun', 'propn'],
            'other': ['num', 'adp', 'cconj', 'punct', 'sym', 'x']
        }

        plt.figure(figsize=(6, 4))
        sns.lineplot(x='layer', y='score', hue='tag', data=score_data)
        save_plot(f'preds-tags-layer-changes-{model_name}', 'F1 scores per POS tag per layer')

        plt.figure(figsize=(6, 4))
        sns.lineplot(x='layer', y='score', data=score_data)
        save_plot(f'preds-macro-layer-changes-{model_name}', 'Macro averaged F1 scores per layer')

        for name, items in pos_groups.items():
            slug = name.replace(' ', '_').replace('2)', '').replace('/', '').replace('(', '')

            group_data = score_data[score_data['tag'].isin(items)]
            plt.figure(figsize=(6, 4))
            sns.lineplot(x='layer', y='score', hue='tag', data=group_data)
            save_plot(f'preds-tags-layer-changes-{model_name}-{slug}',
                      f'F1 scores per POS tag per layer for {name}')

            plt.figure(figsize=(6, 4))
            sns.lineplot(x='layer', y='score', data=group_data)
            save_plot(f'preds-macro-layer-changes-{model_name}-{slug}',
                      f'Macro averaged F1 scores per layer for {name}')


def mistake_patterens(layers_preds):
    patterns, labels = [], []

    for preds in zip(*layers_preds):
        true = preds[0]
        pattern = [pred == true for pred in preds[1:]]
        pattern = [str(pattern[i])[0] for i in range(len(pattern)) if i == 0 or pattern[i] != pattern[i-1]]

        if len(pattern) > 2:
            pattern = [pattern[0], '~', pattern[-1]]

        patterns.append(' > '.join(pattern))
        labels.append(true)

    # print(f'{len(set(patterns))}/{len(patterns)} are unique')
    return patterns, labels


def show_mistake_patterens(models_patterns, labels, model_names):
    pattern_order = ['T',
                     'F',
                     'T > F',
                     'F > ~ > F',
                     'T > ~ > F',
                     'F > T',
                     'F > ~ > T',
                     'T > ~ > T']

    full_data = []
    for model_name, patterns in zip(model_names, models_patterns):
        pattern_counts = Counter(patterns)
        for pattern in sorted(pattern_counts, key=pattern_counts.get, reverse=True):
            print(f'{pattern_counts[pattern]}\t{pattern}')

        data = []
        for pattern, label in zip(patterns, labels):
            data.append((label, pattern))
            full_data.append((model_name, label, pattern))

        data = pd.DataFrame(data, columns=['tag', 'pattern'])

        plt.figure(figsize=(12, 3))
        sns.countplot(x='tag', hue='pattern', data=data, hue_order=pattern_order)
        plt.legend(ncol=2, loc='upper right')
        sns.despine()
        save_plot(f'layer-error-patterns-tags-{model_name}',
                  f'layerwise prediction error patterns')

    full_data = pd.DataFrame(full_data, columns=['model', 'tag', 'pattern'])
    plt.figure(figsize=(6, 5))
    sns.countplot(y='pattern', hue='model', data=full_data, order=pattern_order)
    save_plot(f'layer-error-patterns', f'layerwise prediction error patterns')


def show_pattern_groups(pattern_data):
    print(pattern_data.describe())

    # for label in pattern_data.label.unique():
    #     data = pattern_data[pattern_data.label == label]

    #     sns.barplot(data)


def layerwise_confusion_matrices(layers_preds, label_names, model_name):
    y_true = layers_preds[0]
    layers_preds = layers_preds[1:]

    subset = {'adj', 'adv', 'propn', 'noun', 'verb'}

    for i in range(len(layers_preds)):
        filename = f'layer-errors-{model_name}-{str(i).zfill(2)}'

        # at layer i
        y_pred = layers_preds[i]
        plt.figure(figsize=(8.5, 3))
        plot_heatmap(y_pred, y_true, 'predicted tag', 'actual tag', label_names,
                     f'cm {model_name} layer {i}', filename, subset)

        # before layer i
        layers_before = layers_preds[:i]
        if len(layers_before) > 0:
            y_pred = [pred for preds in layers_before for pred in preds]
            plt.figure(figsize=(8.5, 3))
            plot_heatmap(y_pred, y_true * len(layers_before), 'predicted tag', 'actual tag', label_names,
                         f'cumulative errors {model_name} before layer {i}', f'{filename}-before', subset)

        # after layer i
        layers_after = layers_preds[i + 1:]
        if len(layers_after) > 0:
            y_pred = [pred for preds in layers_after for pred in preds]
            plt.figure(figsize=(8.5, 3))
            plot_heatmap(y_pred, y_true * len(layers_after), 'predicted tag', 'actual tag', label_names,
                         f'cumulative errors {model_name} after layer {i}', f'{filename}-after', subset)

    y_pred = [pred for preds in layers_preds for pred in preds]
    plt.figure(figsize=(8.5, 3))
    plot_heatmap(y_pred, y_true * len(layers_preds), 'predicted tag', 'actual tag', label_names,
                 f'cumulative errors {model_name}', f'layer-errors-{model_name}-full', subset)


def layerwise_change_confusion_matrices(layers_preds, label_names, model_name):
    # label_names = ['det', 'pron', 'sconj', 'adv', 'adj', 'noun', 'propn', 'verb', 'aux']

    layers_preds = layers_preds[1:]

    for i in range(1, len(layers_preds)):
        cm = confusion_matrix(layers_preds[i-1], layers_preds[i], labels=label_names)
        cm_norm = confusion_matrix(layers_preds[i-1], layers_preds[i], labels=label_names, normalize='true')
        sns.heatmap(cm_norm, annot=cm, xticklabels=label_names, yticklabels=label_names, cmap=sns.cm.rocket_r, fmt='g')
        plt.ylabel(f'{model_name} layer {i-1}')
        plt.xlabel(f'{model_name} layer {i}')
        plt.figure(figsize=(6, 4))
        save_plot(f'layer-changes-{model_name}-{str(i).zfill(2)}', f'changes {model_name} layer {i-1}-{i}')


def model_patterns_confusion_matrices(model_patterns, tokens):
    patterns1, patterns2 = model_patterns

    patterns1 = [str(p) for p in patterns1]
    patterns2 = [str(p) for p in patterns2]

    names = sorted(set(patterns1))

    cm = confusion_matrix(patterns1, patterns2, labels=names)
    cm_norm = confusion_matrix(patterns1, patterns2, labels=names, normalize='true')
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_norm, annot=cm, xticklabels=names, yticklabels=names, cmap=sns.cm.rocket_r, fmt='g')
    plt.ylabel('BERTje')
    plt.xlabel('mBERT')
    save_plot(f'model-pattern-diffs', 'patterns')

# pattern_order = ['T',
#                  'F',
#                  'T > F',
#                  'F > ~ > F',
#                  'T > ~ > F',
#                  'F > T',
#                  'F > ~ > T',
#                  'T > ~ > T']


def print_examples(sents, tokens, layers_preds1, layers_preds2, patterns1, patterns2):
    labels = layers_preds1[0]
    layers_preds1 = layers_preds1[1:]
    layers_preds2 = layers_preds2[1:]

    # layers_preds1 = [[p[i] for p in layers_preds1] for i in range(len(tokens)) if labels[i] == 'noun']
    # layers_preds2 = [[p[i] for p in layers_preds2] for i in range(len(tokens))]

    print('\n > model-pattern-diffs examples')
    tokens = [tuple(pieces) for pieces in tokens]
    all_pairs = list(zip(patterns1, patterns2, tokens, labels))

    all_pairs2 = [(p1, p2, t, lab) for (p1, p2, t, lab) in all_pairs]
    print(f'number of examples: {len(all_pairs2)}')

    pattern_pair_tokens = Counter(all_pairs2)
    for pair, count in pattern_pair_tokens.most_common(None):
        print(f'\n\n######################################')
        token = ' '.join(pair[2]).replace(' ##', '')
        print(f'token: "{token}", tag: "{pair[3]}"', end=', ')
        print(f'BERTje: "{pair[0]}", mBERT: "{pair[1]}"', end=', ')
        print(f'count: {count},\n')

        i = -1
        for _ in range(count):
            i = all_pairs.index(pair, i+1)
            p1 = [p[i] for p in layers_preds1]
            p2 = [p[i] for p in layers_preds2]
            sent = sents[i].replace(' ##', '')
            print(f'sent:\t{sent}\nBERTje:\t{p1}\nmBERT:\t{p2}\n')


def plot_heatmap(x, y, x_name, y_name, label_names, title, name, label_subset=None):
    cm = confusion_matrix(y, x, labels=label_names)
    cm_norm = confusion_matrix(y, x, labels=label_names, normalize='true')

    xticklabels = label_names
    yticklabels = label_names

    if label_subset is not None:
        cm = [cm[i] for i in range(len(cm)) if label_names[i] in label_subset]
        cm_norm = [cm_norm[i] for i in range(len(cm_norm)) if label_names[i] in label_subset]
        yticklabels = sorted(label_subset)

    sns.heatmap(cm_norm, annot=cm, xticklabels=xticklabels, yticklabels=yticklabels,
                linewidths=.2, cmap=sns.cm.rocket_r, fmt='g')
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    save_plot(name, title)


def plot_model_cms(models_layers_preds, label_names):
    layers_preds1, layers_preds2 = models_layers_preds

    n_layers = len(layers_preds1)

    true_labels = layers_preds1[0]
    n_items = len(true_labels)

    full_preds1_tf, full_preds2_tf = [], []
    full_preds1_ft, full_preds2_ft = [], []
    full_preds1_ff, full_preds2_ff = [], []

    for n in range(1, n_layers):
        print(f'layer {n}')
        preds1 = layers_preds1[n]
        preds2 = layers_preds2[n]
        layer_id = str(n-1).zfill(2)
        plot_heatmap(preds2, preds1, 'mBERT', 'BERTje', label_names,
                     f'full prediction differences in layer {n-1}', f'models-error-diffs-full-{layer_id}')

        def tf(i): return preds1[i] == true_labels[i] and preds2[i] != true_labels[i]
        def ft(i): return preds1[i] != true_labels[i] and preds2[i] == true_labels[i]
        def ff(i): return preds1[i] != true_labels[i] and preds2[i] != true_labels[i]

        # True False
        preds1_tf = [preds1[i] for i in range(n_items) if tf(i)]
        preds2_tf = [preds2[i] for i in range(n_items) if tf(i)]
        full_preds1_tf.extend(preds1_tf)
        full_preds2_tf.extend(preds2_tf)
        plot_heatmap(preds2_tf, preds1_tf, 'mBERT', 'BERTje', label_names,
                     f'tf prediction differences in layer {n-1}', f'models-error-diffs-tf-{layer_id}')

        # False True
        preds1_ft = [preds1[i] for i in range(n_items) if ft(i)]
        preds2_ft = [preds2[i] for i in range(n_items) if ft(i)]
        full_preds1_ft.extend(preds1_ft)
        full_preds2_ft.extend(preds2_ft)
        plot_heatmap(preds2_ft, preds1_ft, 'mBERT', 'BERTje', label_names,
                     f'ft prediction differences in layer {n-1}', f'models-error-diffs-ft-{layer_id}')

        # False False
        preds1_ff = [preds1[i] for i in range(n_items) if ff(i)]
        preds2_ff = [preds2[i] for i in range(n_items) if ff(i)]
        full_preds1_ff.extend(preds1_ff)
        full_preds2_ff.extend(preds2_ff)
        plot_heatmap(preds2_ff, preds1_ff, 'mBERT', 'BERTje', label_names,
                     f'ff prediction differences in layer {n-1}', f'models-error-diffs-ff-{layer_id}')

    plot_heatmap(full_preds1_tf, full_preds2_tf, 'mBERT', 'BERTje', label_names,
                 f'full tf prediction differences', f'models-error-diffs-tf-full')
    plot_heatmap(full_preds1_ft, full_preds2_ft, 'mBERT', 'BERTje', label_names,
                 f'full ft prediction differences', f'models-error-diffs-ft-full')
    plot_heatmap(full_preds1_ff, full_preds2_ff, 'mBERT', 'BERTje', label_names,
                 f'full ff prediction differences', f'models-error-diffs-ff-full')


def align_predictions(models_inputs, models_layers_preds):
    inputs1, inputs2 = models_inputs
    layers_preds1, layers_preds2 = models_layers_preds

    tokens, sents = [], []

    n = 0
    for inp1, inp2 in zip(inputs1, inputs2):
        for ((i1, j1), _), ((i2, j2), _) in zip(inp1['spans'], inp2['spans']):
            valid1, valid2 = (j1 - i1 < 6), (j2 - i2 < 6)
            if valid1 and valid2:
                n += 1
                tokens.append(inp1['tokens'][i1:j1])
                sents.append(' '.join(inp1['tokens']))
            elif valid1 and not valid2:
                for preds in layers_preds1:
                    preds.pop(n)
            elif valid2 and not valid1:
                for preds in layers_preds2:
                    preds.pop(n)

    return layers_preds1, layers_preds2, tokens, sents


def main():
    global task

    parser = ArgumentParser(description='Summarize exported results')
    parser.add_argument('task')
    parser.add_argument('--path', default='data')
    parser.add_argument('--version', default='v0')
    parser.add_argument('--type', default='single')
    args = parser.parse_args()

    task = args.task
    os.makedirs(f'plots/{task}', exist_ok=True)

    model_names = ['bertje', 'mbert']
    model_layers_preds = []
    model_inputs = []

    label_names = []

    model_patterns = []
    labels = []

    print('0')

    for model_name in model_names:
        print(model_name)
        preds_path = os.path.join(args.path, args.task, 'predictions', args.version, f'{model_name}.{args.type}.json')
        inputs_path = os.path.join(args.path, args.task, f'{model_name}.test.json')

        with open(inputs_path) as f:
            model_inputs.append(json.load(f))

        label_names, layers_preds = load_predictions(preds_path)
        model_layers_preds.append(layers_preds)

    print('1')

    # label_names = ['det', 'pron', 'sconj', 'adv', 'adj', 'noun', 'propn', 'verb', 'aux']
    # label_names = ['adp', 'sconj', 'pron', 'det', 'adj', 'propn', 'noun', 'verb', 'aux', 'adv',
    #                'cconj', 'punct', 'num', 'sym', 'x']

    layers_preds1, layers_preds2, tokens, sents = align_predictions(model_inputs, model_layers_preds)
    full_model_layers_preds = layers_preds1, layers_preds2
    model_layers_preds, tokens, sents = only_with_mistakes(model_layers_preds, tokens, sents)

    for lab in sorted(set(model_layers_preds[0][0])):
        print(f'{lab}\t{len([lab2 for lab2 in model_layers_preds[0][0] if lab2 == lab])}')

    # plot_model_cms(model_layers_preds, label_names)

    print('2')

    # orig_datas, score_datas = [], []

    for model_name, layers_preds, full_layers_preds in zip(model_names, model_layers_preds, full_model_layers_preds):
        # patterns, labels = mistake_patterens(layers_preds)
        # model_patterns.append(patterns)
        # print('3')

        # full_orig_data, full_score_data = aggregate_data(label_names, full_layers_preds)
        # orig_data, score_data = aggregate_data(label_names, layers_preds)
        # orig_datas.append(orig_data)
        # score_datas.append(score_data)

        # plot_tag_distribution(orig_data, full_orig_data)
        # print('4')

        layerwise_confusion_matrices(layers_preds, label_names, model_name)
        # layerwise_change_confusion_matrices(layers_preds, label_names, model_name)
        print('5')

    # plot_data(orig_datas, score_datas, model_names)
    # print('6')

    show_mistake_patterens(model_patterns, labels, model_names)
    print('7')

    # pattern_data = pd.DataFrame({
    #     'label': labels,
    #     model_names[0]: model_patterns[0],
    #     model_names[1]: model_patterns[1]
    # })
    # show_pattern_groups(pattern_data)
    # print('8')

    model_patterns_confusion_matrices(model_patterns, tokens)
    print('9')

    print_examples(sents, tokens, *model_layers_preds, *model_patterns)


if __name__ == '__main__':
    main()
