from typing import List

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm

from utils import config
from scalar_mix import ScalarMix


class Probe(nn.Module):
    def __init__(self, n_embeds=1):
        super(Probe, self).__init__()

        self.n_embeds = n_embeds
        self.scalar_mix = ScalarMix(n_embeds) if n_embeds > 1 else None

        n_input = config.hparams.n_input
        n_hidden = config.hparams.n_hidden
        n_labels = len(config.label_map)

        if not config.hparams.out_layer:
            n_hidden = n_labels

        self.in_dropout = nn.Dropout(config.hparams.in_dropout)

        rnn_size = n_hidden // 2 if config.hparams.bidirectional else n_hidden
        rnn_model = nn.LSTM if config.hparams.type == 'lstm' else nn.RNN if config.hparams.type == 'rnn' else None
        self.rnn = rnn_model(n_input, rnn_size, num_layers=config.hparams.num_layers,
                             dropout=config.hparams.rnn_dropout,
                             bidirectional=config.hparams.bidirectional, batch_first=True)

        self.hidden_dropout = nn.Dropout(config.hparams.hidden_dropout)
        self.fc1 = nn.Linear(n_hidden * config.spans, n_labels) if config.hparams.out_layer else None

        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs: List[torch.Tensor], span_sizes, labels=None):
        assert len(inputs) == self.n_embeds

        embed = self.scalar_mix(inputs) if self.scalar_mix is not None else inputs[0]

        spans = []
        for span_i in range(embed.shape[1]):
            span_embed = embed[:, span_i]
            sizes = span_sizes[:, span_i]

            span_embed = self.in_dropout(span_embed)

            span = pack_padded_sequence(span_embed, sizes, batch_first=True, enforce_sorted=False)
            span, _ = self.rnn(span)
            span, _ = pad_packed_sequence(span, batch_first=True)

            spans.append(span[torch.arange(len(sizes)), sizes - 1])

        out = torch.cat(spans, 1)

        if self.fc1 is not None:
            out = self.hidden_dropout(out)
            out = self.fc1(out)

        loss = self.loss(out, labels) if labels is not None else None
        return out, loss


def prepare_batch(batch, mode, layer):
    embeddings, span_sizes, labels = batch

    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
        span_sizes = span_sizes.cuda()
        labels = labels.cuda()

    if mode == 'single':
        inputs = [embeddings.select(2, layer)]
    elif mode == 'mix':
        start, stop = config.skip_embed, layer + 1
        if config.reverse_layers:
            start, stop = layer, embeddings.shape[2]
        inputs = [embeddings.select(2, i) for i in range(start, stop)]
    else:
        raise ValueError('unsupported layer mode "{}"'.format(mode))

    # mask = span_sizes > 0
    # return [t[mask] for t in inputs], span_sizes[mask], labels[mask]
    return inputs, span_sizes, labels


def init_model(mode, layer, path=None) -> Probe:
    cumulative_length = config.layer_range[1] - layer if config.reverse_layers else layer + 1 - config.skip_embed
    model = Probe(n_embeds=1 if mode == 'single' else cumulative_length)

    if torch.cuda.is_available():
        model.cuda()

    if path is not None:
        loc = None if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(path, map_location=loc))

    return model


def train(train_data, dev_data, mode, layer, logdir, out_path):
    writer = SummaryWriter(logdir)

    model = init_model(mode, layer)

    if config.hparams.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.hparams.lr, momentum=0.9)
    elif config.hparams.optimizer == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=config.hparams.lr, weight_decay=config.hparams.weight_decay)
    else:
        raise Exception(f'unknown optimizer {config.hparams.optimizer}')

    train_loader = DataLoader(train_data, batch_size=config.hparams.batch_size, shuffle=not config.lazy)
    dev_loader = DataLoader(dev_data, batch_size=config.hparams.eval_batch_size)

    best_dev_loss = None
    dev_strikes = 0

    global_iter = 0
    running_loss, running_correct, running_iter = 0.0, 0, 0

    stop_early = False

    for epoch in range(1, 51):
        if stop_early:
            break

        # Train epoch
        for batch in tqdm(train_loader, disable=not config.pbar):
            model.train()
            optimizer.zero_grad()

            inputs, span_sizes, labels = prepare_batch(batch, mode, layer)
            out, loss = model(inputs, span_sizes, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (out.argmax(-1) == labels).sum().cpu().item()
            running_iter += len(labels)
            global_iter += 1

            # Log current status
            if global_iter % config.steps.log_interval == 0:
                curr_loss = running_loss / running_iter
                curr_acc = running_correct / running_iter

                writer.add_scalar('Loss/train', curr_loss, global_iter)
                writer.add_scalar('Accuracy/train', curr_acc, global_iter)
                tqdm.write('Layer={:>2} Step={:,} Loss={:.4f} Acc={:.3f}'.format(
                    layer, global_iter, curr_loss, curr_acc))

                if model.scalar_mix is not None:
                    weights = F.softmax(torch.tensor(model.scalar_mix.scalar_parameters), 0)
                    tqdm.write(f'Weights={[round(w.cpu().item(), 3) for w in weights]}')

                running_loss, running_correct, running_iter = 0.0, 0, 0

            # Validation
            if global_iter % config.steps.val_interval == 0:
                dev_loss, dev_correct, dev_iter = 0.0, 0, 0

                for batch in tqdm(dev_loader, disable=not config.pbar):
                    model.eval()

                    inputs, span_sizes, labels = prepare_batch(batch, mode, layer)
                    out, loss = model(inputs, span_sizes, labels)

                    dev_loss += loss.item()
                    dev_correct += (out.argmax(-1) == labels).sum().cpu().item()
                    dev_iter += len(labels)

                avg_loss = dev_loss / dev_iter
                avg_acc = dev_correct / dev_iter
                loss_diff = dev_loss if best_dev_loss is None else best_dev_loss - dev_loss

                writer.add_scalar('Loss/valid', avg_loss, epoch)
                writer.add_scalar('Accuracy/valid', avg_acc, epoch)
                tqdm.write('Epoch={} Valid Loss={:.4f} Acc={:.3f} Loss Diff={:.4f}'.format(epoch, avg_loss,
                                                                                           avg_acc, loss_diff))

                if best_dev_loss is None or dev_loss < best_dev_loss:
                    dev_strikes = 0
                    best_dev_loss = dev_loss
                    tqdm.write('Saving improved probe to {}'.format(out_path))
                    torch.save(model.state_dict(), out_path)
                else:
                    dev_strikes += 1
                    tqdm.write('Not improved. Strike {}/{}'.format(dev_strikes, config.steps.patience))

                if dev_strikes == config.steps.patience:
                    tqdm.write('Not improved for {} validations, stopping early'.format(config.steps.patience))
                    stop_early = True
                    break


def summarize(mode, layer, path, test_data=None):
    model = init_model(mode, layer, path)
    model.eval()

    summary = {}
    labels = []
    preds = []

    if mode == 'mix' and layer > 0 and model.scalar_mix is not None:
        scalar_weights = [p.item() for p in model.scalar_mix.scalar_parameters]
        summary['weights'] = scalar_weights

    if test_data is not None:
        test_loader = DataLoader(test_data, batch_size=64)

        n_correct, n_total = 0, 0
        for batch in tqdm(test_loader, disable=not config.pbar, desc=path):
            inputs, span_sizes, batch_labels = prepare_batch(batch, mode, layer)

            if len(span_sizes) == 0:
                continue

            model.eval()
            out, _ = model(inputs, span_sizes)
            batch_preds = out.argmax(-1)
            n_correct += (batch_preds == batch_labels).sum().item()
            n_total += len(out)
            labels.extend(list(batch_labels.cpu().numpy()))
            preds.extend(list(batch_preds.cpu().numpy()))

        acc = n_correct / n_total
        summary['accuracy'] = acc

    return summary, labels, preds


def report(summaries):
    prev_acc = 0

    for layer, summary in summaries:
        out = 'Layer {:>2}:'.format(layer)

        if 'accuracy' in summary:
            prefix = '+' if summary['accuracy'] > prev_acc else ''
            out += ' Acc={:.3f} ({}{:.3f}),'.format(summary['accuracy'], prefix, summary['accuracy'] - prev_acc)
            prev_acc = summary['accuracy']

        if 'weights' in summary:
            weights = F.softmax(torch.tensor(summary['weights']), 0)
            center = (weights * torch.arange(1, len(weights) + 1)).sum().item()
            out += ' Center-of-gravity={:.2f},'.format(center)
            out += ' Weights={}'.format([round(w.item(), 3) for w in weights])

        print(out)
