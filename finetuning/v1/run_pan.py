import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

import numpy as np


def read_example_file(path):
    example_map = {}

    # Read examples from file
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            label, sent = line.split('\t', maxsplit=1)
            if label not in example_map:
                example_map[label] = []
            example_map[label].append(sent)
    return example_map


def prepare_examples(path, seed=6784):
    example_map = read_example_file(path)

    np.random.seed(seed)

    # Generate sentence pairs
    authors = sorted(example_map.keys())
    examples = []
    for author in authors:
        for sent1 in example_map[author]:
            label = np.random.randint(2) if len(example_map[author]) > 1 else 1
            if label == 0:  # Same author
                sent2 = sent1
                while sent2 == sent1:
                    sent2 = np.random.choice(example_map[author])
            else:  # Different author
                author2 = author
                while author2 == author:
                    author2 = np.random.choice(authors)
                sent2 = np.random.choice(example_map[author2])
            examples.append((sent1, sent2, label))

    print('Generated {} pairs'.format(len(examples)))

    return examples


def prepare_eval_examples(train_path, test_path, n_comps=3, seed=4232):
    example_map = read_example_file(train_path)

    test_examples = {}

    np.random.seed(seed)

    # Read examples from file
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            authorlabel, sent = line.split('\t', maxsplit=1)
            author, label = authorlabel.split('|')
            label = 0 if label == 'Y' else 1
            if author not in test_examples:
                test_examples[author] = []

            for n in range(n_comps):
                sent2 = np.random.choice(example_map[author])
                test_examples[author].append((sent, sent2, label))

    return test_examples


def wrap_examples(examples, tokenizer):
    all_input_ids, all_token_type_ids, all_input_masks, all_labels = [], [], [], []
    for sent1, sent2, label in examples:
        tokenized = tokenizer.encode_plus(sent1, sent2, max_length=512, add_special_tokens=True,
                                          pad_to_max_length=True)

        all_input_ids.append(tokenized['input_ids'])
        all_token_type_ids.append(tokenized['token_type_ids'])
        all_input_masks.append(tokenized['attention_mask'])
        all_labels.append(label)

    # Return as data set
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
    all_labels = torch.tensor(all_labels)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_input_masks, all_labels)
    return dataset


def load_data(input_path, filename, tokenizer, cache_dir=None,
              max_seq_length=512, allow_clipping=True):

    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, filename + '.pkl')
        if os.path.exists(cache_path):
            print(' ➤ Loading cached data from {}'.format(cache_path))
            return torch.load(cache_path)

    print(' ➤ Loading data from {}'.format(filename))
    examples = prepare_examples(os.path.join(input_path, filename))

    print(' ➤ Wrapping dataset')
    dataset = wrap_examples(examples, tokenizer)

    if cache_dir is not None:
        torch.save(dataset, cache_path)
        print(' ➤ Cached data in {}'.format(cache_path))

    return dataset


def load_eval_data(input_path, train_file, test_file, tokenizer, cache_dir=None):
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, test_file + '.pkl')
        if os.path.exists(cache_path):
            print(' ➤ Loading cached data from {}'.format(cache_path))
            return torch.load(cache_path)

    train_path = os.path.join(input_path, train_file)
    test_path = os.path.join(input_path, test_file)
    print(' ➤ Loading evaluation data from {}'.format(test_path))
    example_map = prepare_eval_examples(train_path, test_path)
    print(' ➤ Wrapping examples')
    datasets = {author: wrap_examples(examples, tokenizer) for author, examples in example_map.items()}

    if cache_dir is not None:
        torch.save(datasets, cache_path)
        print(' ➤ Cached data in {}'.format(cache_path))

    return datasets


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
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    return model


def train_model(model, device, train_data, test_data, output_path,
                num_epochs=4, batch_size=8, lr=5e-5, eps=1e-8, max_grad_norm=1.0, seed=4327):

    if type(model) == str:
        # Load pretrained model
        print(' ➤ Loading base model: {}'.format(model))
        model = BertForSequenceClassification.from_pretrained(model)
    model.to(device)
    model.train()

    # Initialize training
    print(' ➤ Start training')
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    # dev_dataloader = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=128)

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.zero_grad()
    for epoch in range(1, num_epochs + 1):
        print('  • Start epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(tqdm(train_dataloader), 1):
            model.train()

            input_ids, token_type_ids, input_masks, true_labels = [b.to(device) for b in batch]

            inputs = {
                'input_ids': input_ids,
                'attention_mask': input_masks,
                'token_type_ids': token_type_ids,
                'labels': true_labels,
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            model.zero_grad()

            # Evaluation
            pred_labels = logits.argmax(1)
            assert pred_labels.shape == true_labels.shape

            epoch_loss += loss.item()
            n_correct += (pred_labels == true_labels).sum().item()
            n_total += len(true_labels)

            n_steps = 50
            if step % n_steps == 0:
                _, test_acc = predict(model, test_data, device, return_acc=True)
                args = (epoch, num_epochs, step, len(train_dataloader),
                        epoch_loss / n_steps, n_correct / n_total, test_acc)
                print('    ~ Epoch {}/{}, batch {}/{}: Loss={:.3f} Acc={:.3f} TAcc={:.3f}'.format(*args))
                epoch_loss, n_correct, n_total = 0.0, 0, 0

        save_model(model, output_path, 'model-{}'.format(epoch))
    model.eval()
    return model


def predict(model, test_data, device, return_acc=False, batch_size=64):
    model.eval()

    authors = sorted(test_data.keys())
    preds = []
    n_correct = 0
    with torch.no_grad():
        for author in tqdm(authors, 'Evaluating'):
            logits = []

            data = test_data[author]
            dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size)

            true_label = None
            for batch in dataloader:
                input_ids, token_type_ids, input_masks, true_labels = [b.to(device) for b in batch]

                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': input_masks,
                    'token_type_ids': token_type_ids,
                }
                outputs = model(**inputs)
                logits.extend(list(outputs[0].argmax(1).cpu().numpy()))

                if true_label is None:
                    true_label = true_labels[0].cpu().numpy()

            pred_label = int(np.round(np.mean(logits)))
            if pred_label == true_label:
                n_correct += 1
            pred = 'Y' if pred_label == 0 else 'N'
            preds.append((author, pred))
    if return_acc:
        return preds, n_correct / len(authors)
    return preds


def run_task(model, input_path, output_path, device, model_file='model',
             do_train=True, do_eval=False, cache_dir=None):

    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, model.replace('/', '-'), input_path.replace('/', '-'))
        os.makedirs(cache_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(model, do_basic_tokenize=False)
    print(' ➤ Loading training data')
    train_data = load_data(input_path, 'train.tsv', tokenizer, cache_dir=cache_dir, allow_clipping=True)
    print(' ➤ Loading validation data')
    test_data = load_eval_data(input_path, 'train.tsv', 'test.tsv', tokenizer, cache_dir=cache_dir)

    print(' ➤ Data loading finished')

    if model_file != 'model':
        print(' ➤ Using model checkpoint {}'.format(model_file))
        model = load_model(output_path, device, name=model_file)

    if do_train:
        model = train_model(model, device, train_data, test_data,
                            output_path=output_path)
        save_model(model, output_path)
    elif type(model) == str:
        model = load_model(output_path, device, name=model_file)

    if do_eval:
        predictions = predict(model, test_data, device)
        with open(os.path.join(output_path, 'predictions.txt'), 'w') as f:
            for problem_name, pred in predictions:
                f.write('{}\t{}\n'.format(problem_name, pred))


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
    parser.add_argument("--num_threads", type=int, default=30)

    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)

    if not args.do_train and not args.do_eval:
        print('Specify --do_train and/or --do_eval')
        exit(-1)

    if args.do_train and not args.force and os.path.exists(args.output_dir):
        print('Output path already exists')
        exit(-1)

    print('\n▶ Arguments:')
    for key in vars(args):
        print(' ➤ {:15}: {}'.format(key, getattr(args, key)))

    print('\n▶ Device:')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' ➤ Using device: {}'.format(device.type))

    run_task(args.model, args.input_dir, args.output_dir, device, args.model_file, args.do_train,
             args.do_eval, cache_dir=args.cache_dir)


if __name__ == '__main__':
    main()
