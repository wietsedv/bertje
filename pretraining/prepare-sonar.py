from glob import glob
import gzip
import os
import sys
from tqdm import tqdm
from tokenizer import BasicTokenizer

if len(sys.argv) < 3:
    print('Usage: python3 prepare-sonar.py source-path dest-path')

source_dir = sys.argv[1]  # /net/corpora/LassyLarge
dest_dir = sys.argv[2]

BLACKLIST_PATH = '/net/corpora/LassySmall/Suites/W*-*-*.sents'

tokenizer = BasicTokenizer()

os.makedirs(dest_dir, exist_ok=True)

id_blacklist = {f.split('/')[-1][:-6] for f in glob(BLACKLIST_PATH)}  # { 'WR-P-P-H-0000000020' }

print('Blacklist: {} documents'.format(len(id_blacklist)))

IGNORE_GROUPS = {
    'WR-P-E-A',  # flemish chats
    'WR-P-E-J',  # wikipedia
    'WR-P-E-L',  # tweets
    'WR-U-E-A',  # chats
    'WR-U-E-D',  # sms
}

ignored_groups = 0
ignored_docs = 0

for group in tqdm(os.listdir(source_dir), ncols=80, position=0):
    group_dir = os.path.join(source_dir, group, 'SUITES')

    if group[0] != 'W':
        continue
    if group in IGNORE_GROUPS:
        ignored_groups += 1
        continue

    group_outs = []

    for filename in tqdm(os.listdir(group_dir), ncols=80, desc=group, position=1):
        filepath = os.path.join(group_dir, filename)

        with gzip.open(filepath, 'r') as f:
            out = ''
            doc_id = ''
            for line in f.readlines():
                line = str(line, encoding='utf-8')
                if '|' not in line:
                    continue
                full_id, sent = line.split('|', 1)
                new_doc_id = full_id[:full_id.index('.')]

                if new_doc_id in id_blacklist:
                    ignored_docs += 1
                    continue
                if doc_id and new_doc_id != doc_id:
                    out += '\n'

                out += ' '.join(tokenizer.tokenize(sent.strip())) + '\n'
                doc_id = new_doc_id
        group_outs.append(out)

    out = '\n'.join(group_outs)
    with open(os.path.join(dest_dir, '{}.txt'.format(group)), 'w') as f:
        f.write(out)

print('Ignored {} groups and {} documents'.format(ignored_groups, ignored_docs))
