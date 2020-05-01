import os
from glob import glob
import json
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set(style='whitegrid')


def load_summaries(base_path='data', task=None, version=None, agg=None, model=None, layer=None):
    summaries = {}

    model = '*' if model is None else model
    layer = '*' if layer is None else layer.zfill(2)

    tasks = os.listdir(base_path) if task is None else [task]

    for t in tasks:
        if not os.path.isdir(os.path.join(base_path, t)):
            continue

        if t not in summaries:
            summaries[t] = {}

        task_path = os.path.join(base_path, t, 'summaries')
        if not os.path.isdir(task_path):
            print(f'There are no summaries for task {t}')
            continue

        versions = os.listdir(task_path) if version is None else [version]

        for v in versions:
            if v not in summaries[t]:
                summaries[t][v] = {}

            version_path = os.path.join(task_path, v)
            aggs = os.listdir(version_path) if agg is None else [agg]

            for a in aggs:
                if a not in summaries[t][v]:
                    summaries[t][v][a] = {}

                path = os.path.join(version_path, a, f'{model}.{layer}.*.json')

                for filepath in glob(path):
                    m, lay = filepath.split('/')[-1].split('.')[:2]
                    lay = int(lay)

                    if m not in summaries[t][v][a]:
                        summaries[t][v][a][m] = {}

                    with open(filepath) as f:
                        summaries[t][v][a][m][lay] = json.load(f)

    return summaries


def flatten_summaries(summaries):
    flat = []
    for task, task_res in summaries.items():
        for version, version_res in task_res.items():
            for agg, agg_res in version_res.items():
                for model, model_res in agg_res.items():
                    for layer, result in model_res.items():
                        summary = {'task': task, 'version': version, 'aggregate': agg,
                                   'model': model, 'layer': layer, 'result': result}
                        flat.append(summary)
    return flat


def get_weights(weights):
    weights = F.softmax(torch.tensor(weights), 0)
    center = (weights * torch.arange(1, len(weights) + 1)).sum().item()
    weights = list(weights.numpy())
    return weights, center


def plot_accuracies(acc_table, task, agg, color='muted', val_min=1, val_max=0):
    layer_names = ['lex'] + [str(i) for i in range(1, 13)]

    f = plt.figure(figsize=(6, 4))  # ()

    # for agg, color in [('mix', 'muted')]:
    sns.set_color_codes(color)

    keys = sorted([key for key in acc_table.keys() if key[1] == agg])

    x = [name for key in keys for name in layer_names]
    y = [acc_table[key][i] - acc_table[key][i - 1] if i >
         0 else 0 for key in keys for i in range(len(acc_table[key]))]
    hue = [' '.join(key) for key in keys for acc in acc_table[key]]

    if val_min == 1:
        val_min = min([val_min] + y)
    if val_max == 0:
        val_max = max([val_max] + y)

    sns.barplot(x=x, y=y, hue=hue, palette=color)

    ax = f.axes[0]
    ax.legend(ncol=2, loc='upper right', frameon=True)
    ax.set(ylim=(val_min, val_max), ylabel='Accuracy deltas', xlabel='Layer')
    # ax.set_title(task)
    sns.despine(left=True, bottom=True)
    f.tight_layout()
    # plt.show()

    return f


def plot_weights(weights_table, task):
    layer_names = ['lex'] + [str(i) for i in range(1, 13)]

    for key, layers in weights_table.items():
        data = np.zeros((13, 12))

        show_plot = False
        for layer, weights in enumerate(layers):
            if weights is None:
                continue

            data[13 - len(weights):, layer - 1] = weights
            show_plot = True

        if not show_plot:
            continue

        f = plt.figure(figsize=(6, 4))
        sns.heatmap(data, xticklabels=layer_names[1:], yticklabels=list(reversed(layer_names)), annot=True)
        # ax.set_title(f'{key[0]} {task}')
        ax = f.axes[0]
        ax.set(xlabel='Cumulative layer', ylabel='Weighted layer')
        sns.despine(left=True, bottom=True)
        f.tight_layout()

        return f


def plot_weight_curves(weights_table, task):
    layer_names = ['lex'] + [str(i) for i in range(1, 13)]

    f = plt.figure(figsize=(6, 3))

    for key, layers in weights_table.items():
        weights = layers[12]
        if weights is None:
            continue

        sns.lineplot(x=layer_names, y=weights, label=key[0], sort=False)

    # ax.set_title(task)
    ax = f.axes[0]
    ax.set(xlabel='Cumulative layer', ylabel='Weighted layer')
    sns.despine(left=True, bottom=True)
    f.tight_layout()

    return f


def plot_overall(overall_summary):
    f = plt.figure(figsize=(6, 4))

    for metric, color in [('center', 'pastel'), ('expected', 'muted')]:
        sns.set_color_codes(color)

        x = [res[metric] for model, model_res in overall_summary.items() for task, res in model_res.items()]
        y = [task for model, model_res in overall_summary.items() for task, res in model_res.items()]
        hue = [model for model, model_res in overall_summary.items() for task, res in model_res.items()]

        sns.barplot(x=x, y=y, hue=hue, palette=color, order=[
                    'udalpino-pos', 'udlassy-pos', 'conll2002-ner', 'udlassy-dep', 'sonar-coref', 'udalpino-dep'])

    ax = f.axes[0]
    ax.get_legend().set_visible(False)
    sns.despine(left=True, bottom=True)
    f.tight_layout()

    return f


def describe(summaries, out_dir=None, tablefmt='simple'):
    overall_summary = {}

    for task, task_res in summaries.items():
        for version, version_res in task_res.items():
            name = f'{task} ({version})'
            pad_line = '#' * (80 + len(name) % 2)
            pad = '#' * (39 - len(name) // 2)

            print(f'\n{pad_line}\n{pad} {name} {pad}\n{pad_line}')

            acc_table = {}
            weights_table = {}
            center_table = {}

            for agg, agg_res in version_res.items():
                for model, model_res in agg_res.items():
                    col_name = (model, agg)

                    acc_table[col_name] = [None] * 13
                    weights_table[col_name] = [None] * 13
                    center_table[col_name] = [None] * 13

                    for layer, result in model_res.items():
                        center = None
                        acc = result['accuracy']

                        if 'weights' in result:
                            weights, center = get_weights(result['weights'])
                            weights_table[col_name][layer] = weights
                            center_table[col_name][layer] = center
                            # expect_table[col_name][layer] = expected_layer

                        acc_table[col_name][layer] = acc

            print('\nAccuracies:\n')
            print(tabulate(acc_table, headers='keys', showindex='always', floatfmt='.3f', tablefmt=tablefmt))

            print('\n\nAccuracy deltas:\n')
            delta_acc_table = {key: [None] + [accs[i] - accs[i - 1]
                                              for i in range(1, len(accs))] for key, accs in acc_table.items()}
            print(tabulate(delta_acc_table, headers='keys', showindex='always', floatfmt='.3f', tablefmt=tablefmt))

            print('\n\nExpected layers:\n')
            expected = {}
            for key, deltas in delta_acc_table.items():
                expected_layer = sum([i * deltas[i] for i in range(1, len(deltas))]) / sum(deltas[1:])
                expected[key] = expected_layer
                print(f'{key}\t{expected_layer}')

            # expect_table = {key: [acc - max(acc_table[key]) for i in range(1, len(deltas))]
            #                 for key, deltas in delta_acc_table.items()}
            # print(tabulate(expect_table, headers='keys', showindex='always', floatfmt='.3f', tablefmt=tablefmt))

            print('\n\nGravitational centers:\n')
            print(tabulate(center_table, headers='keys', showindex='always', floatfmt='.3f', tablefmt=tablefmt))

            for model in ['bertje', 'mbert']:
                if model not in overall_summary:
                    overall_summary[model] = {}

                overall_summary[model][task] = {
                    'expected': expected[(model, 'mix')],
                    'center': center_table[(model, 'mix')][12],
                    'weights': weights_table[(model, 'mix')][12]
                }

            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)

                val_min, val_max = 1, 0
                if task.endswith('-pos'):
                    val_min, val_max = -0.004, 0.014
                if task.endswith('-dep'):
                    val_min, val_max = -0.004, 0.014

                f = plot_accuracies(acc_table, task, agg='mix', val_min=val_min, val_max=val_max)
                f.savefig(os.path.join(out_dir, f'{task}-{version}-mix-accuracies.png'))
                plt.close(f)

                val_min, val_max = 1, 0
                if task.endswith('-pos'):
                    val_min, val_max = -0.004, 0.014
                if task.endswith('-dep'):
                    val_min, val_max = -0.006, 0.014

                f = plot_accuracies(acc_table, task, agg='single', val_min=val_min, val_max=val_max)
                f.savefig(os.path.join(out_dir, f'{task}-{version}-single-accuracies.png'))
                plt.close(f)

                # plot_weights(weights_table, task).savefig(os.path.join(out_dir, f'{task}-{version}-weights.png'))

                f = plot_weight_curves(weights_table, task)
                f.savefig(os.path.join(out_dir, f'{task}-{version}-weight-curves.png'))
                plt.close(f)

    if out_dir is not None:
        plot_overall(overall_summary).savefig(os.path.join(out_dir, 'overview.png'))


def main():
    parser = ArgumentParser(description='Summarize exported results')
    parser.add_argument('--path', default='data')
    parser.add_argument('--task', default=None)
    parser.add_argument('--version', default=None)
    parser.add_argument('--agg', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--layer', default=None)
    parser.add_argument('--tablefmt', default='simple')
    parser.add_argument('-o', default=None, help='Plot export path')
    args = parser.parse_args()

    summaries = load_summaries(args.path, args.task, args.version, args.agg, args.model, args.layer)

    describe(summaries, out_dir=args.o, tablefmt=args.tablefmt)


if __name__ == '__main__':
    main()
