import os
from glob import glob

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForTokenClassification, BertForSequenceClassification, BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForTokenClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from utils import config
from data import load_data, read_examples


def prepare_optimizer(model, dataloader):
    t_total = len(dataloader) // config.train.gradient_accumulation_steps * config.train.max_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.optimizer.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(params, lr=config.optimizer.learning_rate, eps=config.optimizer.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.optimizer.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, global_step):
    out_path = config.get_path('output')
    checkpoint_dir = os.path.join(out_path, 'checkpoint-{}'.format(str(epoch).zfill(3)))
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.save_pretrained(checkpoint_dir)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'random_state': torch.random.get_rng_state()
    }, os.path.join(checkpoint_dir, "state.pt"))

    # Delete previous state
    if epoch > 0:
        prev_state_path = os.path.join(out_path, 'checkpoint-{}'.format(str(epoch - 1).zfill(3)), "state.pt")
        os.remove(prev_state_path)


def load_model():
    model_path = config.model.name
    state = None

    if config.model.checkpoint is not False:
        checkpoint_path = os.path.join(config.get_path('output'), 'checkpoint-*')
        checkpoints = sorted(glob(checkpoint_path))
        if len(checkpoints) > 0:
            n = config.model.checkpoint
            model_path = checkpoints[-1] if n < 0 else checkpoint_path.replace('*', str(n).zfill(3))
            print('Loading checkpoint from "{}"'.format(model_path))

            state_path = os.path.join(model_path, "state.pt")
            if os.path.exists(state_path):
                try:
                    state = torch.load(state_path, map_location=config.model.device)
                except Exception:
                    print('WARNING: could not load state dict')
            elif config.model.do_train:
                raise Exception('attempting to resume training from {}, but state.pt is missing'.format(model_path))

    if config.model.type == 'roberta':
        clf = RobertaForTokenClassification if config.data.token_level else RobertaForSequenceClassification
    else:
        clf = BertForTokenClassification if config.data.token_level else BertForSequenceClassification

    model = clf.from_pretrained(model_path, num_labels=config.data.num_labels,
                                attention_probs_dropout_prob=config.train.attention_dropout,
                                hidden_dropout_prob=config.train.hidden_dropout)
    model.to(config.model.device)
    return model, state


def prepare_batch(batch):
    batch_len = batch[1].sum(1).max()

    input_ids, input_mask, token_type_ids, true_labels, label_mask = [
        b[:, :batch_len].contiguous().to(config.model.device) for b in batch[:5]]

    inputs = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'labels': true_labels,
    }

    if config.data.token_level:
        inputs['attention_mask'] = input_mask

    if len(batch) > 5:
        labels_valid = batch[5].to(config.model.device)
        return inputs, true_labels, label_mask, labels_valid

    return inputs, true_labels, label_mask


def train(model, train_dataset, dev_dataset=None, state=None):
    writer = SummaryWriter(config.get_path('logs'))

    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset), batch_size=config.train.batch_size)

    if dev_dataset is not None:
        dev_dataloader = DataLoader(dev_dataset,
                                    sampler=SequentialSampler(dev_dataset), batch_size=config.eval.batch_size)

    optimizer, scheduler = prepare_optimizer(model, train_dataloader)
    torch.random.manual_seed(config.train.seed)

    model.zero_grad()

    # Get step intervals
    gradient_steps = config.train.gradient_accumulation_steps
    logging_steps = config.train.logging_steps
    if logging_steps < 1:
        logging_steps = int(len(train_dataloader) // gradient_steps * logging_steps)
    if logging_steps == 0:
        logging_steps = 1
    eval_steps = config.train.eval_steps
    if eval_steps < 1:
        eval_steps = int(len(train_dataloader) // gradient_steps * eval_steps)

    print('Global step intervals: Logging={} Eval={}'.format(logging_steps, eval_steps))

    current_epoch = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    # Restore previous checkpoint
    if state is not None:
        torch.random.set_rng_state(state['random_state'].cpu())
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        current_epoch = state['epoch'] + 1
        global_step = state['global_step']

    print('Starting at epoch {}'.format(current_epoch))

    for epoch in range(current_epoch, config.train.max_epochs):
        print(' > Start epoch {}/{}'.format(epoch, config.train.max_epochs))

        n_correct, n_total = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Batch", disable=not config.verbose)):
            model.train()

            inputs, true_labels, label_mask = prepare_batch(batch)

            outputs = model(**inputs)
            loss, out = outputs[:2]
            loss.backward()

            tr_loss += loss.item()

            pred_labels = out.argmax(-1)
            pred_labels = pred_labels.reshape(*true_labels.shape)

            n_correct += (label_mask * (pred_labels == true_labels)).sum().item()
            n_total += label_mask.sum().item() if config.data.token_level else true_labels.shape[0]

            # print(n_correct, n_total, true_labels.shape, pred_labels.shape, (pred_labels == true_labels).shape)
            assert n_correct <= n_total

            if (step + 1) % gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    lr = scheduler.get_lr()[0]
                    loss = (tr_loss - logging_loss) / logging_steps
                    acc = n_correct / n_total

                    tqdm.write('Epoch={} Step={} lr={:.9f} loss={:.3f} acc={:.3f}'.format(epoch, global_step,
                                                                                          lr, loss, acc))

                    writer.add_scalar("Learning Rate", lr, global_step)
                    writer.add_scalar("Loss/Train", loss, global_step)
                    writer.add_scalar("Accuracy/Train", acc, global_step)
                    logging_loss = tr_loss

        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, global_step)

        # Evaluation
        if dev_dataset is not None:
            eval_result = evaluate(model, dev_dataloader)
            writer.add_scalar("Loss/Eval", eval_result['loss'], epoch)
            writer.add_scalar("Accuracy/Eval", eval_result['acc'], epoch)
            tqdm.write('Evaluation: Epoch={} loss={:.3f} acc={:.3f}'.format(epoch,
                                                                            eval_result['loss'], eval_result['acc']))

    writer.close()


def evaluate(model, dataloader, return_acc=True, return_labels=False, return_probs=False):
    batched_true, batched_preds, batched_sent_ids = [], [], []
    batched_probs = []
    n_correct, n_total = 0, 0
    sent_i = 0

    result = {
        'loss': 0
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            model.eval()

            batch = prepare_batch(batch)
            inputs, true_labels, label_mask = batch[:3]

            labels_valid = batch[3] if len(batch) > 3 else None

            outputs = model(**inputs)
            loss, out = outputs[:2]

            pred_labels = out.argmax(-1)
            pred_labels = pred_labels.reshape(*true_labels.shape)

            result['loss'] += loss.item()

            if labels_valid is not None:
                pred_labels = pred_labels == true_labels
                true_labels = labels_valid == 1

            if return_acc:
                n_correct += (label_mask * (pred_labels == true_labels)).sum().item()
                n_total += label_mask.sum().item() if config.data.token_level else true_labels.shape[0]

            mask_idx = label_mask.flatten().nonzero()
            if return_labels:
                batch_size = true_labels.shape[0]
                batch_len = true_labels.shape[1]

                true_labels = true_labels.flatten()[mask_idx].flatten()
                pred_labels = pred_labels.flatten()[mask_idx].flatten()
                sent_ids = torch.tensor(range(sent_i, sent_i + batch_size)).reshape(-1, 1).repeat(1, batch_len)
                sent_ids = sent_ids.flatten()[mask_idx].flatten()
                sent_i += batch_size

                batched_true.append(true_labels)
                batched_preds.append(pred_labels)
                batched_sent_ids.append(sent_ids)

            if return_probs:
                # print(out.shape, out.argmax(-1), out.sum(-1))
                probs = torch.softmax(out, -1).max(-1)[0]
                probs = probs.flatten()[mask_idx].flatten()
                batched_probs.append(probs)

    if return_acc:
        result['acc'] = n_correct / n_total

    if return_labels:
        result['labels_true'] = torch.cat(batched_true, 0).cpu()
        result['labels_pred'] = torch.cat(batched_preds, 0).cpu()
        result['sent_ids'] = torch.cat(batched_sent_ids, 0).cpu()

    if return_probs:
        result['pred_probs'] = torch.cat(batched_probs, 0).cpu()

    return result


def export(model, dataset, label_map, filename):
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=config.eval.batch_size)
    result = evaluate(model, dataloader, return_acc=False, return_labels=True, return_probs=config.summary.probs)

    groups = None
    if config.summary.groups:
        grouped_sents, _ = read_examples(os.path.join(config.data.input, filename), add_labels=2)
        sents = [[s[0] for s in ex] for ex in grouped_sents]
        groups = [[s[1] for s in ex] for ex in grouped_sents]
    else:
        sents = read_examples(os.path.join(config.data.input, filename), add_labels=False)

    label_names = sorted(label_map, key=label_map.get)

    labels_true = result['labels_true']
    labels_pred = result['labels_pred']
    sent_ids = result['sent_ids']
    pred_probs = result['pred_probs'] if config.summary.probs else [None] * len(sent_ids)

    if config.model.checkpoint >= 0:
        filename = str(config.model.checkpoint).zfill(3) + '-' + filename

    out_path = os.path.join(config.get_path('output'), filename)
    with open(out_path, 'w') as f:
        prev_sent_id = 0
        token_id = 0
        for label_true, label_pred, sent_id, pred_prob in zip(labels_true, labels_pred, sent_ids, pred_probs):
            if sent_id != prev_sent_id:
                if config.data.token_level:
                    f.write('\n')
                prev_sent_id = sent_id
                token_id = 0

            true, pred = label_names[label_true], label_names[label_pred]
            if token_id >= len(sents[sent_id]):
                print('skipping sent={} token={} true={} pred={}'.format(sent_id, token_id, true, pred))
                continue

            token = sents[sent_id][token_id] if groups is None else groups[sent_id][token_id]
            out = [token, true, pred]
            if config.summary.probs:
                out.append(str(pred_prob.item()))
            f.write('\t'.join(out) + '\n')

            token_id += 1

    print('Predictions are exported to {}'.format(out_path))


def main():
    config.show()

    if config.model.name is None:
        print('provide config model name')
        exit(0)

    if config.data.input is None:
        print('provide task data')
        exit(0)

    print('Loading tokenizer "{}"'.format(config.model.name))
    Tokenizer = RobertaTokenizer if config.model.type == 'roberta' else BertTokenizer
    tokenizer = Tokenizer.from_pretrained(config.model.name, do_lower_case=False)

    cache_dir = config.get_path('cache')

    train_dataset, label_map = load_data(config.data.input, 'train.tsv', tokenizer,
                                         cfg=config.data, cache_dir=cache_dir)
    print('Train data: {} examples, {} labels: {}'.format(len(train_dataset), len(label_map), list(label_map.keys())))

    dev_dataset = None
    if config.data.dev:
        dev_dataset, _ = load_data(config.data.input, 'dev.tsv', tokenizer, label_map,
                                   cfg=config.data, cache_dir=cache_dir)
        print('Dev data: {} examples'.format(len(dev_dataset)))

    print('Loading model "{}"'.format(config.model.name))
    model, state = load_model()

    if config.model.do_train:
        print('Start training')
        train(model, train_dataset, dev_dataset, state)

    if config.model.do_export:
        test_dataset, _ = load_data(config.data.input, 'test.tsv', tokenizer, label_map,
                                    cfg=config.data, cache_dir=cache_dir)

        # print('\nExporting train:')
        # export(model, train_dataset, label_map, 'train.tsv')
        if dev_dataset is not None:
            print('Exporting dev')
            export(model, dev_dataset, label_map, 'dev.tsv')

        print('Exporting test')
        export(model, test_dataset, label_map, 'test.tsv')

    print('\nDone!')


def test():
    print(config.model.name, config.model.checkpoint)


if __name__ == '__main__':
    main()
