import os
import json
from subprocess import run
from sklearn.metrics import f1_score, classification_report

from utils import reset_config, config


def conlleval(filepath):
    with open(filepath) as f:
        txt = str.encode(f.read())

    process = run(['./conlleval', '-d', '\t'], input=txt, capture_output=True)
    out = process.stdout.decode('utf-8').split('\n')[1]
    return float(out[-5:])


def accuracy(filepath):
    n_total, n_correct = 0, 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            _, true, pred = line.split('\t')
            n_total += 1
            n_correct += true == pred
    return n_correct / n_total


def groupacc(filepath):
    groups = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = line.split('\t')
            group, true, pred = parts[:3]
            if group not in groups:
                groups[group] = []

            p = float(parts[3]) if len(parts) > 3 else 1
            groups[group].append(p if true == pred else 1 - p)

    n_correct = sum([round(sum(preds) / len(preds)) for preds in groups.values()])
    return n_correct / len(groups)


def label_set(filepath):
    labels = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            _, true, pred = line.split('\t')
            labels.add(true)
            labels.add(pred)
    return labels


def catlabels(filepath):
    y_true, y_pred = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            _, true, pred = line.split('\t')
            y_true.append(true.split('-', maxsplit=1)[-1])
            y_pred.append(pred.split('-', maxsplit=1)[-1])
    return y_true, y_pred


def microcat(filepath):
    y_true, y_pred = catlabels(filepath)
    labels = list(set(y_true) - {'O'})
    return f1_score(y_true, y_pred, labels=labels, average='micro')


def reportcat(filepath):
    y_true, y_pred = catlabels(filepath)
    labels = list(set(y_true) - {'O'})
    print(classification_report(y_true, y_pred, labels, digits=4))


def finalize_checkpoint(path, name, stats, labels):
    # Link checkpoint
    ckpt_name = f'checkpoint-{name}'
    model_path = os.path.join(path, f'model')
    if os.path.exists(model_path) and os.path.islink(model_path):
        os.remove(model_path)
    os.symlink(ckpt_name, model_path)

    # Save stats
    with open(os.path.join(path, ckpt_name, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Fix config
    cfg_path = os.path.join(path, ckpt_name, 'config.json')
    with open(cfg_path) as f:
        cfg = json.load(f)

    if len(labels) != cfg['num_labels']:
        print(f'WARNING: not {cfg["num_labels"]} labels in {path} but {len(labels)}')
        cfg['num_labels'] = len(labels)

    if 'output_attentions' in cfg:
        del cfg['output_attentions']
        del cfg['output_hidden_states']
        del cfg['output_past']
        del cfg['pruned_heads']
        del cfg['torchscript']
        del cfg['use_bfloat16']
    cfg['id2label'] = {str(i): lab for i, lab in enumerate(labels)}
    cfg['label2id'] = {lab: i for i, lab in enumerate(labels)}
    os.rename(cfg_path, cfg_path + '.old')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)


def score_file(filepath):
    if config.summary.method == "microcat":
        score = microcat(filepath)
    elif config.summary.method == "conlleval":
        score = conlleval(filepath)
    elif config.summary.method == "accuracy":
        score = accuracy(filepath)
    elif config.summary.method == "groupacc":
        score = groupacc(filepath)
    else:
        print('Warning: unknown evaluation method "{}". Falling back to accuracy'.format(config.summary.method))
        score = accuracy(filepath)
    return score


def summarize(output_path):
    suffix = '-{}.tsv'.format(config.summary.type)
    dev_outputs = sorted([f for f in os.listdir(config.get_path('output')) if f.endswith(suffix)])

    if config.summary.method == "count":
        last = dev_outputs[-1] if len(dev_outputs) > 0 else 'does not exist'
        print(last)
        return

    best_filename = None

    scores = []
    max_score = 0
    for filename in dev_outputs:
        filepath = os.path.join(output_path, filename)
        score = score_file(filepath)

        suffix = 'BEST' if score >= max_score else ''
        print('{}: {:2.4f} {}'.format(filename, score, suffix))

        if score >= max_score:
            max_score = score
            best_filename = filename
        scores.append((filename, score))

    # if config.summary.method.endswith('cat'):
    if best_filename is not None:
        best_filepath = os.path.join(output_path, best_filename)
        print('\nBEST: {} [{}]\n'.format(best_filepath, max_score))
        if config.summary.method != "groupacc":
            reportcat(best_filepath)

        labels = sorted(label_set(os.path.join(output_path, best_filename)))
        test_score = score_file(os.path.join(output_path, best_filename.replace('-dev', '-test')))
        stats = {'best_checkpoint': int(best_filename[:3]), 'dev_score': max_score,
                 'test_score': test_score, 'dev_scores': scores}
        finalize_checkpoint(output_path, best_filename[:3], stats, labels)

    # for filename, score in scores:
    #     suffix = 'BEST' if score == max_score else ''
    #     print('{}: {:2.2f} {}'.format(filename, score, suffix))


def main():
    data_cfgs = config.data.cfgs if config.data.name is None else [None]
    model_cfgs = config.model.cfgs if config.model.name is None else [None]

    for data_cfg in data_cfgs:
        for model_cfg in model_cfgs:
            print('\nExporting predictions for {}'.format(model_cfg))
            reset_config()
            config.add('util/export')
            if model_cfg is not None:
                config.add(model_cfg)
            if data_cfg is not None:
                config.add(data_cfg)

            out_path = config.get_path('output')
            if os.path.exists(out_path):
                final_path = os.path.join(out_path, 'model')
                if not config.force and os.path.exists(final_path) and os.path.islink(final_path):
                    print(f'\n{final_path} already exists. skipping')
                    continue
                summarize(out_path)
            else:
                print(out_path, 'does not exist')
            print('')


if __name__ == '__main__':
    main()
