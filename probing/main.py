import os
import json
import multiprocessing as mp
import time
import shutil

import torch.cuda

from utils import config
from data import data_path, load_embeddings
from probe import train, summarize, report


def run_layer(train_data, dev_data, test_data, task_name, model_name, mode, layer):
    task_dir = os.path.join(task_name, 'probes', config.name, mode)
    probe_path = data_path(task_dir, 'train', 'pt', model_name, layer)

    if config.train:
        print('\nstart training layer {} of {} with task {} with config {}'.format(
            layer, model_name, task_name, config.name))

        logdir = os.path.join(config.data.logdir, task_name, config.name, model_name, mode, str(layer).zfill(2))

        if os.path.exists(probe_path):
            print('skipping, {} already exists'.format(probe_path))
        else:
            train(train_data, dev_data, mode, layer, logdir, probe_path + '.tmp')
            shutil.move(probe_path + '.tmp', probe_path)

    if config.export:
        if not os.path.exists(probe_path):
            print('skipping, {} does not exist'.format(probe_path))
        else:
            summary_dir = os.path.join(task_name, 'summaries', config.name, mode)
            summary_path = data_path(summary_dir, mode, 'json', model_name, layer)

            summary, labels, preds = summarize(mode, layer, probe_path, test_data)
            with open(summary_path, 'w') as f:
                json.dump(summary, f)

            return labels, preds

    return None, None


def run_task(task_name, task_data, task_format, model_name, model_id):
    mode = config.layer_mode
    train_data, dev_data, test_data = None, None, None

    if config.train:
        dev_data = load_embeddings(task_name, task_data, task_format, 'dev', model_name, model_id, config.label_map)
        train_data = dev_data if config.sample else load_embeddings(
            task_name, task_data, task_format, 'train', model_name, model_id, config.label_map)

    if config.export:
        test_data = load_embeddings(task_name, task_data, task_format, 'test', model_name, model_id, config.label_map)

    if config.dry_run:
        print('loaded data, now stopping dry run')
        return

    layers_labels = []

    n_workers = config.num_workers if config.num_workers > 0 else 1
    if n_workers == 1:
        for layer in range(*config.layer_range):
            layer_labels, layer_preds = run_layer(train_data, dev_data, test_data, task_name, model_name, mode, layer)
            if config.export and layer_labels is not None:
                if len(layers_labels) == 0:
                    layers_labels.append(layer_labels)
                layers_labels.append(layer_preds)
    else:
        procs_queue, procs_running = [], []
        for layer in range(*config.layer_range):
            p = mp.Process(target=run_layer, args=(train_data, dev_data,
                                                   test_data, task_name, model_name, mode, layer))
            procs_queue.append(p)
            # run_layer(train_data, dev_data, test_data, task_name, model_name, mode, layer)

        for p in procs_queue:
            while len(procs_running) >= n_workers:
                time.sleep(1)
                for i in range(n_workers):
                    if not procs_running[i].is_alive():
                        procs_running.pop(i)
                        break

            procs_running.append(p)
            p.start()

        for p in procs_running:
            p.join()

    if len(layers_labels) > 0:
        preds_dir = os.path.join(task_name, 'predictions', config.name)
        preds_path = data_path(preds_dir, mode, 'json', model_name)

        with open(preds_path, 'w') as f:
            for labels in zip(*layers_labels):
                labels = [config.label_map[lab] for lab in labels]
                f.write('\t'.join(labels) + '\n')

        print(f'Saved layer-wise predictions to {preds_path}')

    if config.report:
        summaries = []

        for layer in range(*config.layer_range):
            summary_dir = os.path.join(task_name, 'summaries', config.name, mode)
            summary_path = data_path(summary_dir, mode, 'json', model_name, layer)

            if not os.path.exists(summary_path):
                print('skipping, {} does not exist'.format(summary_path))
            else:
                with open(summary_path) as f:
                    summary = json.load(f)
                summaries.append((layer, summary))

        report(summaries)


def main():
    mp.set_start_method('spawn')

    config.show()

    task_names = config.task_name if type(config.task_name) == list else [config.task_name]
    task_datas = config.task_data if type(config.task_data) == list else [config.task_data]
    task_formats = config.task_format if type(config.task_format) == list else [config.task_format]

    models = config.model if type(config.model) == list else [config.model]
    model_names = config.model_name if type(config.model_name) == list else [config.model_name]
    assert len(models) == len(model_names)

    print('CUDA available' if torch.cuda.is_available() else 'Running on CPU')

    for task_name, task_data, task_format in zip(task_names, task_datas, task_formats):
        for model_id, model_name in zip(models, model_names):
            print('\nStarting task {} with model {}'.format(task_name, model_name))
            run_task(task_name, task_data, task_format, model_name, model_id)


if __name__ == '__main__':
    main()
