import gzip
import os
import sys
from tqdm import tqdm
from tokenizer import BasicTokenizer


if len(sys.argv) < 3:
    print('Usage: python3 prepare-twnc.py source-path dest-path')

source_dir = sys.argv[1]
dest_dir = sys.argv[2]

source_path = os.path.join(source_dir, 'SUITES')

tokenizer = BasicTokenizer()

os.makedirs(dest_dir, exist_ok=True)

group_outs, i = [], 0
for filename in tqdm(os.listdir(source_path), ncols=80):
    if not filename.endswith('.sents.gz'):
        continue

    with gzip.open(os.path.join(source_path, filename), 'r') as f:
        out = ''
        doc_id = ''
        for line in f.readlines():
            line = str(line, encoding='utf-8')
            if '|' not in line:
                continue
            full_id, sent = line.split('|', 1)
            new_doc_id = tuple(full_id.split('-')[:2])

            if doc_id and new_doc_id != doc_id:
                out += '\n'

            out += ' '.join(tokenizer.tokenize(sent.strip())) + '\n'
            doc_id = new_doc_id
        group_outs.append(out)

    if len(group_outs) == 100:
        dest_path = dest_dir + '/{}.txt'.format(i)
        with open(dest_path, 'w') as f:
            f.write('\n'.join(group_outs))
        group_outs = []
        i += 1

if len(group_outs) > 0:
    dest_path = dest_dir + '/{}.txt'.format(i)
    with open(dest_path, 'w') as f:
        f.write('\n'.join(group_outs))
