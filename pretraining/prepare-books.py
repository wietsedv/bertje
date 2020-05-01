import os
import sys

from tqdm import tqdm

import nltk.data
from tokenizer import BasicTokenizer

if len(sys.argv) < 3:
    print('Usage: python3 prepare-books.py source-path dest-path')

source_dir = sys.argv[1]
dest_dir = sys.argv[2]

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
tokenizer = BasicTokenizer()

os.makedirs(dest_dir, exist_ok=True)

file_names = list(os.listdir(source_dir))
out, i, j = '', 0, 0
for filename in tqdm(file_names, ncols=80):
    source_path = os.path.join(source_dir, filename)

    if out:
        out += '\n'

    with open(source_path, 'r') as f:
        lines = list(f.readlines())
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue

            # Sentence and word tokenization
            sents = sent_detector.tokenize(line)
            sents = [' '.join(tokenizer.tokenize(s)) for s in sents]
            out += '\n'.join(sents) + '\n'

    i += 1
    if i == 100:
        dest_path = dest_dir + '/{}.txt'.format(j)
        with open(dest_path, 'w') as f:
            f.write(out)
        out, i = '', 0
        j += 1

if out:
    dest_path = dest_dir + '/{}.txt'.format(j)
    with open(dest_path, 'w') as f:
        f.write(out)
