import os
import argparse

from tqdm import tqdm

from sklearn.metrics import classification_report

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import AdamW


def read_examples(path, add_labels=True):
    examples = [[]]
    label_set = set()

    # Read examples from file
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                examples.append([])
                continue

            token, label = line.split('\t')
            if add_labels:
                examples[-1].append((token, label))
                label_set.add(label)
            else:
                examples[-1].append(token)

    if len(examples[-1]) == 0:
        del examples[-1]

    if add_labels:
        return examples, label_set
    return examples


def load_data(input_path, filename, tokenizer, label_map=None, cache_dir=None,
              max_seq_length=512, allow_clipping=True):

    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, filename + '.pkl')
        if os.path.exists(cache_path):
            print(' ➤ Loading cached data from {}'.format(cache_path))
            return torch.load(cache_path)

    print(' ➤ Loading data from {}'.format(filename))
    examples, label_set = read_examples(os.path.join(input_path, filename))

    # Convert examples to features
    if label_map is None:
        label_set.add('O')
        label_map = {label: i for i, label in enumerate(sorted(label_set))}

    all_input_ids, all_input_masks, all_segment_ids, all_label_ids, all_label_masks = [], [], [], [], []
    for example in examples:
        input_ids, input_masks, label_ids, label_masks = [], [], [], []

        for token, label in example:
            token_ids = tokenizer.encode(token)
            input_ids.extend(token_ids)
            label_ids.extend([label_map[label]] + [label_map['O']] * (len(token_ids) - 1))
            label_masks.extend([1] + [0] * (len(token_ids) - 1))

        if len(input_ids) > 512 and allow_clipping:
            input_ids = input_ids[:512]
            label_ids = label_ids[:512]
            label_masks = label_masks[:512]

        padding_length = max_seq_length - len(input_ids)

        input_masks = [1] * len(input_ids) + [0] * padding_length
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        label_masks.extend([0] * padding_length)
        label_ids.extend([label_map['O']] * padding_length)

        all_input_ids.append(input_ids)
        all_input_masks.append(input_masks)
        all_segment_ids.append([0] * max_seq_length)
        all_label_ids.append(label_ids)
        all_label_masks.append(label_masks)

    # Return as data set
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
    all_label_masks = torch.tensor(all_label_masks, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids, all_label_masks)

    if cache_dir is not None:
        torch.save((dataset, label_map), cache_path)
        print(' ➤ Cached data in {}'.format(cache_path))

    return dataset, label_map


def save_model(model, output_path, name='model'):
    task_model_path = os.path.join(output_path, name)
    os.makedirs(task_model_path, exist_ok=True)
    model.save_pretrained(task_model_path)
    print(' ➤ Task model is saved to: {}'.format(task_model_path))


def load_model(output_path, device, name='model'):
    model_path = os.path.join(output_path, name)
    if not os.path.exists(model_path):
        raise ValueError('Could not find model at: {}. You first need to train a task model.'.format(model_path))

    print(' ➤ Loading fully trained model from: {}'.format(model_path))
    model = BertForTokenClassification.from_pretrained(model_path)
    model.to(device)
    return model


def train_model(model, device, train_data, dev_data, label_map, output_path, xla_model=None,
                num_epochs=4, batch_size=8, lr=5e-5, eps=1e-8, max_grad_norm=1.0, seed=4327):

    if type(model) == str:
        # Load pretrained model
        print(' ➤ Loading base model: {}'.format(model))
        config = BertConfig.from_pretrained(model, num_labels=len(label_map),
                                            hidden_dropout_prob=0.2, attention_dropout_prob=0.2)
        model = BertForTokenClassification.from_pretrained(model, config=config)
    model.to(device)
    model.train()

    # Initialize training
    print(' ➤ Start training')
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=128)

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.zero_grad()
    for epoch in range(1, num_epochs + 1):
        print('  • Start epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(tqdm(train_dataloader), 1):
            model.train()

            input_ids, input_mask, token_type_ids, true_labels, label_mask = [b.to(device) for b in batch]

            inputs = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': token_type_ids,
                'labels': true_labels,
            }
            outputs = model(**inputs)
            loss, out = outputs[:2]
            loss.backward()

            if xla_model is None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            else:
                xla_model.optimizer_step(optimizer, barrier=True)

            model.zero_grad()

            # Evaluation
            pred_labels = out.argmax(2)
            assert pred_labels.shape == true_labels.shape

            epoch_loss += loss.item()
            n_correct += (label_mask * (pred_labels == true_labels)).sum().item()
            n_total += label_mask.sum().item()

            n_steps = 500
            if step % n_steps == 0:
                dev_acc = eval_model(model, device, dev_dataloader, return_acc=True)
                args = (epoch, num_epochs, step, len(train_dataloader),
                        epoch_loss / n_steps, n_correct / n_total, dev_acc)
                print('    ~ Epoch {}/{}, batch {}/{}: Loss={:.3f}, Acc={:.3f}, Dev Acc={:.3f}'.format(*args))
                epoch_loss, n_correct, n_total = 0.0, 0, 0

        save_model(model, output_path, 'model-{}'.format(epoch))
        show_model_results(model, device, dev_loader=dev_dataloader, label_map=label_map)
    model.eval()
    return model


def eval_model(model, device, dataloader, return_acc=False, return_labels=False):
    model.eval()

    if not return_acc and not return_labels:
        raise ValueError('Set at least one of return_acc or return_labels to true')

    batched_true, batched_preds, batched_sent_ids = [], [], []
    n_correct, n_total = 0, 0
    sent_i = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, input_mask, token_type_ids, true_labels, label_mask = [b.to(device) for b in batch]
            inputs = {
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': token_type_ids
            }

            outputs = model(**inputs)
            out = outputs[0]
            pred_labels = out.argmax(2)

            if return_acc:
                n_correct += (label_mask * (pred_labels == true_labels)).sum().item()
                n_total += label_mask.sum().item()

            if return_labels:
                mask_idx = label_mask.flatten().nonzero()

                batch_size = true_labels.shape[0]

                true_labels = true_labels.flatten()[mask_idx].flatten()
                pred_labels = pred_labels.flatten()[mask_idx].flatten()
                sent_ids = torch.tensor(range(sent_i, sent_i + batch_size)).reshape(-1, 1).repeat(1, 512)
                sent_ids = sent_ids.flatten()[mask_idx].flatten()
                sent_i += batch_size

                batched_true.append(true_labels)
                batched_preds.append(pred_labels)
                batched_sent_ids.append(sent_ids)

    total_acc = 0
    if return_acc:
        total_acc = n_correct / n_total

    if return_labels:
        all_true = torch.cat(batched_true, 0).cpu()
        all_preds = torch.cat(batched_preds, 0).cpu()
        all_sent_ids = torch.cat(batched_sent_ids, 0).cpu()
        return (total_acc, all_true, all_preds, all_sent_ids) if return_acc else (all_true, all_preds, all_sent_ids)

    return total_acc


def show_model_results(model, device, train_loader=None, dev_loader=None, test_loader=None, label_map=None):
    target_names = sorted(label_map, key=label_map.get)
    labels = list(range(len(target_names)))

    out = []

    if train_loader is not None:
        print(' ➤ Train results:')
        train_true, train_pred, sent_ids = eval_model(model, device, train_loader, return_labels=True)
        train_report = classification_report(train_true, train_pred, target_names=target_names, labels=labels)
        out.append((train_true, train_pred, sent_ids))
        print(train_report)

    if dev_loader is not None:
        print(' ➤ Dev results:')
        dev_true, dev_pred, sent_ids = eval_model(model, device, dev_loader, return_labels=True)
        dev_report = classification_report(dev_true, dev_pred, target_names=target_names, labels=labels)
        out.append((dev_true, dev_pred, sent_ids))
        print(dev_report)

    if test_loader is not None:
        print(' ➤ Test results:')
        test_true, test_pred, sent_ids = eval_model(model, device, test_loader, return_labels=True)
        test_report = classification_report(test_true, test_pred, target_names=target_names, labels=labels)
        out.append((test_true, test_pred, sent_ids))
        print(test_report)

    return out


def export_predictions(in_dir, in_name, out_dir, out_name, label_ids, sent_ids, label_map):
    label_names = sorted(label_map, key=label_map.get)
    sents = read_examples(os.path.join(in_dir, in_name), add_labels=False)

    out_path = os.path.join(out_dir, out_name)
    print(' ➤ Writing predictions for {} sentences to {}'.format(len(sents), out_path))

    if len(sents) != max(sent_ids) + 1:
        raise ValueError('number of original sents {} does not match number of sent predictions {}'.format(
            len(sents), max(sent_ids) + 1))

    sents_labels = [['O' for _ in sent] for sent in sents]

    prev_sent_i, token_i = 0, -1
    for label_id, sent_i in zip(label_ids, sent_ids):
        sent_i = sent_i.item()
        if sent_i > prev_sent_i:
            token_i = -1
            prev_sent_i, token_i = sent_i, -1
        token_i += 1
        if token_i >= len(sents_labels[sent_i]):
            print(sents[sent_i], sents_labels[sent_i], sent_i, token_i, list(label_ids)[:100], list(sent_ids)[:100])
        sents_labels[sent_i][token_i] = label_names[label_id]

    with open(out_path, 'w') as f:
        for sent, labels in zip(sents, sents_labels):
            for token, label in zip(sent, labels):
                f.write('{}\t{}\n'.format(token, label))
            f.write('\n')


def run_task(model, input_path, output_path, device, model_file='model',
             do_train=True, do_eval=False, cache_dir=None, xla_model=None):
    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, model.replace('/', '-'), input_path.replace('/', '-'))
        os.makedirs(cache_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(model, do_basic_tokenize=False)
    train_data, label_map = load_data(input_path, 'train.tsv', tokenizer, cache_dir=cache_dir, allow_clipping=True)
    dev_data, _ = load_data(input_path, 'dev.tsv', tokenizer, label_map, cache_dir=cache_dir, allow_clipping=True)

    print(' ➤ Data loading finished')

    if model_file != 'model':
        print(' ➤ Using model checkpoint {}'.format(model_file))
        model = load_model(output_path, device, name=model_file)

    if do_train:
        model = train_model(model, device, train_data, dev_data, label_map=label_map,
                            output_path=output_path, xla_model=xla_model)
        save_model(model, output_path)
    elif type(model) == str:
        model = load_model(output_path, device, name=model_file)

    if do_eval:
        test_data, _ = load_data(input_path, 'test.tsv', tokenizer, label_map, cache_dir=cache_dir)

        train_loader = DataLoader(train_data, batch_size=8)
        dev_loader = DataLoader(dev_data, batch_size=8)
        test_loader = DataLoader(test_data, batch_size=8)

        train, dev, test = show_model_results(model, device, train_loader, dev_loader, test_loader, label_map)

        export_predictions(input_path, 'train.tsv', output_path, 'train.true.tsv', train[0], train[2], label_map)
        export_predictions(input_path, 'train.tsv', output_path, 'train.pred.tsv', train[1], train[2], label_map)
        export_predictions(input_path, 'dev.tsv', output_path, 'dev.true.tsv', dev[0], dev[2], label_map)
        export_predictions(input_path, 'dev.tsv', output_path, 'dev.pred.tsv', dev[1], dev[2], label_map)
        export_predictions(input_path, 'test.tsv', output_path, 'test.true.tsv', test[0], test[2], label_map)
        export_predictions(input_path, 'test.tsv', output_path, 'test.pred.tsv', test[1], test[2], label_map)


def list_tasks(input_path, output_path, whitelist):
    input_paths = input_path.split(',')
    whitelist = False if whitelist == 'all' else whitelist.split(',')
    tasks = []

    for path in input_paths:
        if path is None:
            continue

        if not os.path.exists(path):
            print('Path [{}] does not exist'.format(path))
            exit(-1)

        data_tasks = [task for task in os.listdir(path) if os.path.isdir(os.path.join(path, task))]
        if whitelist:
            data_tasks = [task for task in data_tasks if task in whitelist]

        out_dir = os.path.join(output_path, path.split('/')[-1])
        tasks.extend([(os.path.join(path, task), os.path.join(out_dir, task)) for task in data_tasks])
    return tasks


def main():
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="Comma seperated list of data dir paths (Use prepare-*.py scripts for preprocessing.)")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="Path to pretrained model or shortcut name (bert-base-multilingual-cased)")

    # Optional arguments
    parser.add_argument("--model_file", default='model', type=str, help="Filename of task model")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--force', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument("--tasks", default='all', type=str,
                        help="Comma seperated list of tasks. Names are subdirs of data dirs. Default: all")
    parser.add_argument("--cache_dir", default='cache', type=str, help="Path to cache directory")

    # TPU training
    parser.add_argument("--use_tpu", action='store_true',
                        help="Whether to use a TPU. (Make sure that the environement variables TPU_NAME,\
                              TPU_IP_ADDRESS and XRT_TPU_CONFIG are set)")

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        print('Specify --do_train and/or --do_eval')
        exit(-1)

    if args.do_train and not args.force and os.path.exists(args.output_dir):
        print('Output path already exists')
        exit(-1)

    tasks = list_tasks(args.input_dir, args.output_dir, args.tasks)
    if len(tasks) == 0:
        print('No (whitelisted) tasks found')
        exit(-1)

    print('Starting benchmark tasks!')

    print('\n▶ Arguments:')
    for key in vars(args):
        print(' ➤ {:15}: {}'.format(key, getattr(args, key)))

    print('\n▶ Device:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xla_model = None

    if args.use_tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        xla_model = xm

    print(' ➤ Using device: {}'.format(device.type))

    print('\n▶ Scheduling to run {} tasks:'.format(len(tasks)))
    for task_in, task_out in tasks:
        print(' ➤ {} [Output: {}]'.format(task_in, task_out))

    print('\n' + '#' * 80)

    for i, (task_in, task_out) in enumerate(tasks, 1):
        print('\n▶ Task {}/{} [{}]:'.format(i, len(tasks), task_in))
        run_task(args.model, task_in, task_out, device, args.model_file, args.do_train,
                 args.do_eval, cache_dir=args.cache_dir, xla_model=xla_model)
        print(' ➤ Finished!')


if __name__ == '__main__':
    main()
