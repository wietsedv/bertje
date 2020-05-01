import os
import sys
import io

from zstandard import ZstdDecompressor
from tqdm import tqdm
from tokenizer import BasicTokenizer


if len(sys.argv) < 3:
    print('Usage: python3 prepare-nlwiki.py source-path dest-path')

source_dir = sys.argv[1]
dest_dir = sys.argv[2]

tokenizer = BasicTokenizer()

os.makedirs(dest_dir, exist_ok=True)


def parse_id(full_id):
    docs, paras, sents = full_id.split('|')
    doc_id = docs.split(':')[1]
    para_id = paras.split(':')[1]
    sent_id = sents.split(':')[1]
    return doc_id, para_id, sent_id


def convert_file(stream):
    doc_id, para_id, _ = '', '', ''
    out = ''
    sent = ''

    for line in stream:
        fields = line.rstrip().split('\t')

        if len(fields) > 5:
            word = fields[1]

            if sent:
                sent = sent + ' ' + word
            else:
                sent = word

            new_doc_id, new_para_id, new_sent_id = parse_id(fields[5])
            if doc_id and new_doc_id != doc_id:
                out += '\n'

            doc_id, para_id, _ = new_doc_id, new_para_id, new_sent_id
        else:
            if not sent:
                continue
            if para_id == '0':
                continue

            out += ' '.join(tokenizer.tokenize(sent)) + '\n'
            sent = ''

    if sent:
        out += ' '.join(tokenizer.tokenize(sent)) + '\n'

    return out


dctx = ZstdDecompressor()

for group in tqdm(os.listdir(source_dir), ncols=80, position=0):
    group_dir = os.path.join(source_dir, group)
    group_outs = []

    for filename in tqdm(os.listdir(group_dir), ncols=80, desc=group, position=1):
        filepath = os.path.join(group_dir, filename)

        with open(filepath, 'rb') as f:
            with dctx.stream_reader(f) as r:
                text_stream = io.TextIOWrapper(r, encoding='utf-8')
                out = convert_file(text_stream)
        group_outs.append(out)

    out = '\n'.join(group_outs)
    out_path = os.path.join(dest_dir, '{}.txt'.format(group))
    with open(out_path, 'w') as f:
        f.write(out)
