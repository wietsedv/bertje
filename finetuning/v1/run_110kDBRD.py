import os
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW


def read_example_file(path):
    examples = []

    # Read examples from file
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue

            label, text = line.split('\t', maxsplit=1)
            label = 1 if label == 'pos' else 0
            examples.append((label, text))
    return examples


def wrap_examples(examples, tokenizer):
    all_input_ids, all_input_masks, all_labels = [], [], []
    for label, text in examples:
        tokenized = tokenizer.encode_plus(text, max_length=512, add_special_tokens=True,
                                          pad_to_max_length=True, return_token_type_ids=False)

        all_input_ids.append(tokenized['input_ids'])
        all_input_masks.append(tokenized['attention_mask'])
        all_labels.append(label)

    # Return as data set
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
    all_labels = torch.tensor(all_labels)

    dataset = TensorDataset(all_input_ids, all_input_masks, all_labels)
    return dataset


def load_data(input_path, filename, tokenizer, cache_dir=None):
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, filename + '.pkl')
        if os.path.exists(cache_path):
            print(' ➤ Loading cached data from {}'.format(cache_path))
            return torch.load(cache_path)

    print(' ➤ Loading data from {}'.format(filename))
    examples = read_example_file(os.path.join(input_path, filename))
    print(' ➤ Wrapping dataset')
    dataset = wrap_examples(examples, tokenizer)

    if cache_dir is not None:
        torch.save(dataset, cache_path)
        print(' ➤ Cached data in {}'.format(cache_path))

    return dataset


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


def train_model(model, device, train_dataloader, dev_dataloader, output_path,
                num_epochs=4, lr=2e-5, eps=1e-8, max_grad_norm=1.0, seed=4327):

    if type(model) == str:
        # Load pretrained model
        print(' ➤ Loading base model: {}'.format(model))
        model = BertForSequenceClassification.from_pretrained(model)
    model.to(device)
    model.train()

    # Initialize training
    print(' ➤ Start training')

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.zero_grad()
    for epoch in range(1, num_epochs + 1):
        print('  • Start epoch {}/{}'.format(epoch, num_epochs))
        epoch_loss, n_correct, n_total = 0.0, 0, 0

        for step, batch in enumerate(tqdm(train_dataloader), 1):
            model.train()

            input_ids, input_mask, true_labels = [b.to(device) for b in batch]

            outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=true_labels)
            loss, logits = outputs[:2]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
                args = (epoch, num_epochs, step, len(train_dataloader),
                        epoch_loss / n_steps, n_correct / n_total)
                print('    ~ Epoch {}/{}, batch {}/{}: Loss={:.3f}, Acc={:.3f}'.format(*args))
                epoch_loss, n_correct, n_total = 0.0, 0, 0

        _, test_acc = predict(model, dev_dataloader, device, return_acc=True)
        print('    ~ Epoch {}/{}, Test Acc={:.3f}'.format(epoch, num_epochs, test_acc))
        save_model(model, output_path, 'model-{}'.format(epoch))
    model.eval()
    return model


def predict(model, dataloader, device, return_acc=False, batch_size=64):
    model.eval()

    trues, preds = [], []
    n_correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids, input_mask, true_labels = batch

            outputs = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
            out_labels = outputs[0].argmax(1).cpu()

            trues.extend(list(true_labels.numpy()))
            preds.extend(list(out_labels.numpy()))
            n_correct += int((true_labels == out_labels).sum())

    if return_acc:
        return preds, n_correct / len(preds)
    return preds


def run_task(model, input_path, output_path, device, model_file='model',
             do_train=True, do_eval=False, cache_dir=None,
             batch_size=8):

    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, model.replace('/', '-'), input_path.replace('/', '-'))
        os.makedirs(cache_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(model, do_basic_tokenize=True, do_lower_case=False)
    print(' ➤ Loading training data')
    train_data = load_data(input_path, 'train.tsv', tokenizer, cache_dir=cache_dir)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    print(' ➤ Loading validation data')
    test_data = load_data(input_path, 'test.tsv', tokenizer, cache_dir=cache_dir)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=128)

    if model_file != 'model':
        print(' ➤ Using model checkpoint {}'.format(model_file))
        model = load_model(output_path, device, name=model_file)

    if do_train:
        model = train_model(model, device, train_dataloader, test_dataloader,
                            output_path=output_path)
        save_model(model, output_path)
    elif type(model) == str:
        model = load_model(output_path, device, name=model_file)

    if do_eval:
        predictions = predict(model, test_dataloader, device)
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
