import os
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def read_examples(path, add_labels=True):
    examples = [[]]
    label_set = set()

    if add_labels is True:
        add_labels = 1

    # Read examples from file
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                examples.append([])
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                print(line)
            txt = parts[0]  # if len(parts) == 2 else tuple(parts[:-1])
            if add_labels:
                label = parts[add_labels]
                # if len(parts) > 2:
                #     valid = parts[2].lower() == 'y'
                #     examples[-1].append((txt, label, valid))
                # else:
                examples[-1].append((txt, label))
                label_set.add(label)
            else:
                examples[-1].append(txt)

    if len(examples[-1]) == 0:
        del examples[-1]

    if len(examples) == 1:
        new_examples = []
        for example in examples[0]:
            new_examples.append([example])
        examples = new_examples

    if add_labels:
        return examples, label_set
    return examples


def clip_pad(input_ids, label_ids, label_masks, tokenizer, clip_start, token_level, pad_label):
    # Clip length
    if len(input_ids) > 510:
        if clip_start:
            input_ids = input_ids[-510:]
            label_ids = label_ids[-510:]
            label_masks = label_masks[-510:]
        else:
            input_ids = input_ids[:510]
            label_ids = label_ids[:510]
            label_masks = label_masks[:510]

    # Add special tokens
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    if token_level:
        label_ids = [pad_label] + label_ids + [pad_label]
        label_masks = [0] + label_masks + [0]

    # Pad to maximum length
    padding_length = 512 - len(input_ids)
    input_masks = [1] * len(input_ids) + [0] * padding_length
    input_ids.extend([tokenizer.pad_token_id] * padding_length)

    assert len(input_ids) == 512
    assert len(input_masks) == 512

    if token_level:
        label_ids.extend([pad_label] * padding_length)
        label_masks.extend([0] * padding_length)

        assert len(label_ids) == 512
        assert len(label_masks) == 512
    else:
        assert len(label_ids) == 1
        assert len(label_masks) == 1

    return input_ids, input_masks, label_ids, label_masks


def merge_examples(examples, merge):
    new_examples = []

    i = 0
    while i < len(examples):
        example = []

        for j in range(i, i + merge):
            if j >= len(examples):
                break
            if len(example) > 0 and example[0][-1] != examples[j][0][-1]:
                break
            example.extend(examples[j])

        if len({parts[-1] for parts in example}) != 1:
            print(example)
            raise ValueError('not all labels for merged sentences are equal')

        new_examples.append(example)

        i += 1

    return new_examples


def load_data(input_path, filename, tokenizer, label_map=None, cfg=None, cache_dir=None):
    if cache_dir is not None:
        cache_path = os.path.join(cache_dir, filename + '.pkl')
        if os.path.exists(cache_path):
            print(' ➤ Loading cached data from {}'.format(cache_path))
            return torch.load(cache_path)

    print(' ➤ Loading data from {}'.format(filename))
    examples, label_set = read_examples(os.path.join(input_path, filename))

    if not cfg.token_level and cfg.merge is not None:
        examples = merge_examples(examples, cfg.merge)

    # print(examples[0], label_set)

    # Convert examples to features
    if label_map is None:
        if cfg.token_level:
            label_set.add('O')
        label_map = {label: i for i, label in enumerate(sorted(label_set))}

    print(f'   dataset has {len(label_set)} labels')

    pad_label = label_map['O'] if 'O' in label_map else tokenizer.unk_token_id

    all_input_ids, all_input_masks, all_segment_ids, all_label_ids, all_label_masks = [], [], [], [], []
    for example in tqdm(examples):
        input_ids, label_ids, label_masks = [], [], []

        segment_b_length = 0
        for i, parts in enumerate(example):
            txt, label = parts[:2]

            if not cfg.token_level and type(txt) == tuple:
                token_ids = tokenizer.encode(txt[0], add_special_tokens=False, max_length=254)
                token_ids.append(tokenizer.sep_token_id)
                segment_b = tokenizer.encode(txt[1], add_special_tokens=False, max_length=254)
                token_ids.extend(segment_b)
                segment_b_length = len(segment_b) + 1
            else:
                token_ids = tokenizer.encode(txt, add_special_tokens=False, max_length=510)

            if len(token_ids) == 0:
                continue
            if not cfg.token_level and len(input_ids) > 0:
                input_ids.append(tokenizer.sep_token_id)

            input_ids.extend(token_ids)

            if cfg.token_level:
                label_ids.extend([label_map[label] if label in label_map else tokenizer.unk_token_id] +
                                 [pad_label] * (len(token_ids) - 1))
                label_masks.extend([1] + [0] * (len(token_ids) - 1))
            elif i == 0:
                label_ids.append(label_map[label])
                label_masks.append(1)

        input_ids, input_masks, label_ids, label_masks = clip_pad(
            input_ids, label_ids, label_masks, tokenizer, cfg.clip_start, cfg.token_level, pad_label)

        all_input_ids.append(input_ids)
        all_input_masks.append(input_masks)
        all_segment_ids.append([0] * (512 - segment_b_length) + [1] * segment_b_length)
        all_label_ids.append(label_ids)
        all_label_masks.append(label_masks)

    # Return as data set
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_masks = torch.tensor(all_input_masks, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
    all_label_masks = torch.tensor(all_label_masks, dtype=torch.long)

    data = [all_input_ids, all_input_masks, all_segment_ids, all_label_ids, all_label_masks]
    dataset = TensorDataset(*data)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save((dataset, label_map), cache_path)
        print(' ➤ Cached data in {}'.format(cache_path))

    return dataset, label_map
