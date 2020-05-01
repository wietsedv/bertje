
import os
import json
import torch
from tqdm import tqdm
import random

import torch.cuda
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import torch.nn.functional as F

from utils import config

random.seed(68732)


def data_path(task_data, fold, extension, model_name=None, layer=None):
    filename = '{}.{}'.format(fold, extension)
    if layer is not None:
        filename = '{}.{}'.format(str(layer).zfill(2), filename)
    if model_name is not None:
        filename = '{}.{}'.format(model_name, filename)

    dir_path = os.path.join(config.data.path, task_data)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


def load_original(task_name, task_data, task_format, fold):
    path = data_path(task_data, fold, 'tsv')

    if task_format not in ['token', 'token-bio', 'conllu', 'conllu-edges', 'conll-2012']:
        raise Exception(f'unsupported task format {task_format} for task {task_name} with data {task_data}')

    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                if len(data) > 0:
                    yield data
                    data = []
                continue

            if task_format.startswith('conll') and line[0] == '#':
                continue

            parts = line.split('\t')
            if task_format == 'conll-2012':
                doc, token, label = parts[0], parts[2], parts[3]
            elif task_format.startswith('conllu'):
                token = parts[config.token_column]
                label = parts[config.label_column]
            else:
                token, label = parts

            item = {'token': token, 'label': label.lower()}

            if task_format == 'conll-2012':
                item['doc'] = doc
            if task_format == 'conllu-edges':
                item['id'] = parts[config.id_column]
                item['head'] = parts[config.head_column]

            data.append(item)

    if len(data) > 0:
        yield data


def extract_token_spans(sent, task_format, tokenizer):
    pieces = []
    spans = []

    distractor_span = 0

    for t in sent:
        tok_pieces = tokenizer.tokenize(t['token'])
        start, end = len(pieces), len(pieces) + len(tok_pieces)
        pieces.extend(tok_pieces)

        label = t['label']
        span_start = True

        if task_format == 'token-bio':
            if label[0].lower() == 'b':
                span_start = True
                distractor_span = 0
            elif label[0].lower() == 'i':
                span_start = False
                distractor_span = 0
            elif label[0].lower() == 'o':
                if distractor_span > 0:
                    span_start = False
                    distractor_span -= 1
                elif random.uniform(0, 1) <= config.outside_span_prob:
                    span_start = True
                    distractor_span = random.randint(*config.outside_span_range) - 1 if span_start else 0
                else:
                    continue

            if len(label) > 1 and label[1] == '-':
                label = label[2:]

        if span_start:
            spans.append(((start, end), label))
        else:
            spans[-1] = ((spans[-1][0][0], end), label)

    return {'tokens': pieces, 'spans': spans}


def extract_edge_spans(sent, tokenizer):
    edges = {}
    token_spans = {}
    pieces = []

    for t in sent:
        tok_pieces = tokenizer.tokenize(t['token'])
        start, end = len(pieces), len(pieces) + len(tok_pieces)
        pieces.extend(tok_pieces)

        label = t['label']
        token_id = t['id']
        head_id = t['head']

        if head_id not in edges:
            edges[head_id] = set()
        edges[head_id].add(token_id)
        token_spans[token_id] = ((start, end), label)

    # Resolve spans
    spans = []
    for head_id, token_ids in edges.items():
        if head_id not in token_spans:
            continue

        head_span = token_spans[head_id][0]

        for token_id in token_ids:
            (start, end), label = token_spans[token_id]

            if token_id in edges:
                for child_id in edges[token_id]:
                    child_start, child_end = token_spans[child_id][0]

                    if child_start < start:
                        end = child_start
                    if child_end > end:
                        start = child_end

            spans.append(((start, end), head_span, label))

    # Resolve pairs
    # span_pairs = []
    # for head_id, children in edges.items():
    #     if head_id not in spans:
    #         continue

    #     head_span = spans[head_id][0]

    #     for child_id in children:
    #         child_span, label = spans[child_id]
    #         span_pairs.append((child_span, head_span, label))

    return {'tokens': pieces, 'spans': spans}


def extract_coref_spans(sent, tokenizer):
    pieces = []
    ent_spans = {}

    for t in sent:
        doc, token, label = t['doc'], t['token'], t['label']

        tok_pieces = tokenizer.tokenize(token)
        start, end = len(pieces), len(pieces) + len(tok_pieces)
        pieces.extend(tok_pieces)

        if label == '-':
            continue

        for cid in reversed(label.split('|')):
            first, last = False, False
            if cid[0] == '(':
                first = True
                cid = cid[1:]
            if cid[-1] == ')':
                last = True
                cid = cid[:-1]
            cid = int(cid)

            ent_id = (doc, cid)

            if first:
                if ent_id not in ent_spans:
                    ent_spans[ent_id] = []
                ent_spans[ent_id].append((start, None))
            if last:
                ent_spans[ent_id][-1] = (ent_spans[ent_id][-1][0], end)

    ent_spans = {ent: [(start, end) for start, end in spans if end is not None] for ent, spans in ent_spans.items()}

    spans = [(span, ent) for ent, spans in ent_spans.items() for span in spans]
    return {'tokens': pieces, 'spans': spans}


def ent_samples(ents):
    spans = []

    ent_set = {}
    for span, ent in ents:
        if ent not in ent_set:
            ent_set[ent] = []
        ent_set[ent].append(span)

    for ent1 in ent_set:
        if len(ent_set[ent1]) < 2:
            continue

        n_spans = len(ent_set[ent1])

        # positive samples
        # span1, span2_pos = random.sample(ent_set[ent1], 2)
        random.shuffle(ent_set[ent1])
        for i in range(n_spans - 1):
            spans.append((ent_set[ent1][i], ent_set[ent1][i+1], 'true'))

        # negative samples
        for span1 in ent_set[ent1]:
            random.shuffle(ents)
            for span2_neg, ent2 in ents:
                if ent2 != ent1:
                    spans.append((span1, span2_neg, 'false'))
                    break

        # for span1 in ent_set[ent]:
        #     for span2 in ent_set[ent]:
        #         if span1 == span2:
        #             continue
        #         spans.append((span1, span2, 'true'))

    return spans


# def negative_samples(ents):
#     spans = []

#     for span1, ent1 in ents:
#         neg_spans = [span2 for span2, ent2 in ents if ent2 != ent1]
#         for span2 in random.sample(neg_spans, k=min(len(neg_spans), 2)):
#             spans.append((span1, span2, 'false'))

#     return spans


def coref_negative_sampling(data):
    new_data = []

    pbar = tqdm(total=len(data))

    i = 0
    doc, tokens, ents = None, [], []
    while i < len(data):
        sent_tokens = data[i]['tokens']
        sent_ents = data[i]['spans']

        if len(sent_ents) == 0 or len(sent_tokens) > 510:
            pbar.update()
            i += 1
            continue

        sent_doc = sent_ents[0][1][0]

        too_long = len(tokens) + len(sent_tokens) > 510

        if doc is None:
            doc = sent_doc

        if sent_doc != doc or too_long or i == len(data) - 1:
            new_data.append({'tokens': tokens, 'spans': ent_samples(ents)})
            doc, tokens, ents = sent_doc, [], []

        if too_long:
            if len(data[i - 1]['tokens']) + len(sent_tokens) <= 510:
                pbar.update(-1)
                i -= 1
            continue

        # Shift span indices
        sent_ents = [((start + len(tokens), end + len(tokens)), cid) for (start, end), cid in sent_ents]

        tokens.extend(sent_tokens)
        ents.extend(sent_ents)

        pbar.update()
        i += 1

    pbar.clear()

    return new_data


def load_prepared(task_name, task_data, task_format, fold, model_name, tokenizer, force=False):
    path = data_path(task_name, fold, 'json', model_name)

    # Load already existing data
    if not force and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print('Loaded prepared data from', path)
        return data

    data = []
    label_set = set()
    for sent in tqdm(load_original(task_name, task_data, task_format, fold)):
        if task_format == 'conll-2012':
            spans = extract_coref_spans(sent, tokenizer)
        elif task_format.endswith('-edges'):
            spans = extract_edge_spans(sent, tokenizer)
        else:
            spans = extract_token_spans(sent, task_format, tokenizer)

        data.append(spans)
        label_set.update([span[-1] for span in spans['spans']])

    print('prepared spans')

    if task_format == 'conll-2012':
        print('resampling sentences')
        data = coref_negative_sampling(data)
    else:
        print('Prepared dataset. Labels:', sorted(label_set))

    with open(path, 'w') as f:
        json.dump(data, f)

    return data


model, tokenizer = None, None


# def to_dataset(embeds, span_sizes, labels):
#     n_items = len(embeds)
#     n_spans = config.spans
#     n_tokens = config.max_pieces
#     n_layers = 13
#     n_features = 768

#     t_embeds = torch.zeros(n_items, n_spans, n_tokens, n_layers, n_features, dtype=torch.float32)
#     t_span_sizes = torch.zeros(n_items, n_spans, dtype=torch.long)

#     for item_i, item in enumerate(embeds):
#         for span_i, span in enumerate(item):
#             t_span_sizes[item_i, span_i] = span_sizes[item_i][span_i]

#             for layer_i, layer in enumerate(span):
#                 t_embeds[item_i, span_i, :len(layer), layer_i, :] = layer

#     t_labels = torch.tensor(labels)
#     dataset = TensorDataset(t_embeds, t_span_sizes, t_labels)
#     return dataset


class ProbeDataset(Dataset):
    def __init__(self, embeds, span_sizes, labels):
        print('prepared probe dataset')
        self.embeds = embeds
        self.span_sizes = span_sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_i):
        # embeds = torch.stack([F.pad(torch.stack(t), (0, 0, 0, config.max_pieces - len(t[0])))
        #                       for t in self.embeds[item_i]])
        # span_sizes = torch.tensor(self.span_sizes[item_i])
        # label = torch.tensor(self.labels[item_i])
        return self.embeds[item_i], self.span_sizes[item_i], self.labels[item_i]


class IterProbeDataset(IterableDataset):
    """ Load IterProbe iterable data set """

    def __init__(self, sentences, model, tokenizer, label_map, shuffle=False):
        self.sentences = sentences
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.shuffle = shuffle
        self.items = []

    def __len__(self):
        return sum([len(s['spans']) for s in self.sentences])

    def __iter__(self):
        if len(self.items) == len(self.sentences):
            if self.shuffle:
                random.shuffle(self.items)
            for item in self.items:
                yield item

        n_spans = config.spans
        n_tokens = config.max_pieces

        if self.shuffle:
            random.shuffle(self.sentences)

        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        for sent in self.sentences:
            input_ids = self.tokenizer.convert_tokens_to_ids(sent['tokens'])
            model_input = self.tokenizer.prepare_for_model(input_ids, max_length=512, return_tensors='pt')

            if torch.cuda.is_available():
                model_input = {key: t.cuda() for key, t in model_input.items()}

            _, layers = self.model(**model_input)

            for item in sent['spans']:
                spans = item[:-1]
                label = item[-1]

                embeds = [[] for _ in range(n_spans)]
                span_sizes = []
                label = self.label_map[label]

                skip = False
                for span_i, (start, end) in enumerate(spans):
                    if start >= 510 or end > 510:
                        skip = True
                        break
                    if end <= start or end - start > n_tokens:
                        skip = True
                        break

                    span_sizes.append(end - start)

                    for layer_i, layer in enumerate(layers):
                        layer_embeds = layer[0, start+1:end+1].cpu()
                        embeds[span_i].append(layer_embeds)

                if not skip:
                    self.items.append((embeds, span_sizes, label))

                    embeds = torch.stack([F.pad(torch.stack(t), (0, 0, 0, config.max_pieces - len(t[0])))
                                          for t in embeds])
                    yield embeds, torch.tensor(span_sizes), label


def load_embeddings(task_name, task_data, task_format, fold, model_name, model_id, idx2label):
    global model, tokenizer

    print(f'loading embeddings for {task_name} {fold} from {task_data}')

    label2idx = {l: i for i, l in enumerate(idx2label)}

    path = data_path(task_name, fold, 'pt', model_name)
    if not config.lazy and os.path.exists(path):
        print(f'preparing existing data from {path}')
        return ProbeDataset(*torch.load(path))

    if tokenizer is None:
        print('Loading tokenizer for "{}"'.format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    sentences = load_prepared(task_name, task_data, task_format, fold, model_name, tokenizer)

    if model is None:
        print('Loading model for "{}"'.format(model_name))
        model_config = AutoConfig.from_pretrained(model_id, output_hidden_states=True)
        if config.random_weights:
            model = AutoModelForTokenClassification.from_config(model_config)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_id, config=model_config)

        if torch.cuda.is_available():
            model.cuda()

    print('loading lazily' if config.lazy else 'loading eagerly')
    dataiter = IterProbeDataset(sentences, model, tokenizer, label2idx)
    if config.lazy:
        return dataiter

    embeds, span_sizes, labels = [], [], []
    for emb, spa, lab in tqdm(dataiter):
        embeds.append(emb)
        span_sizes.append(spa)
        labels.append(lab)

    torch.save((embeds, span_sizes, labels), path)
    print(f'saved embeddings to {path}')
    return ProbeDataset(embeds, span_sizes, labels)
