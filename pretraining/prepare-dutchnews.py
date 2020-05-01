import os
import sys

from tqdm import tqdm

import nltk.data
from tokenizer import BasicTokenizer

if len(sys.argv) < 3:
    print('Usage: python3 prepare-dutchnews.py source-path dest-path')

source_dir = sys.argv[1]
dest_dir = sys.argv[2]

sent_detector = nltk.data.load('tokenizers/punkt/dutch.pickle')
tokenizer = BasicTokenizer()

os.makedirs(dest_dir, exist_ok=True)

for filename in sorted(os.listdir(source_dir)):
    source_path = os.path.join(source_dir, filename)
    dest_path = os.path.join(dest_dir, filename)

    out = ''
    with open(source_path, 'r') as f:
        lines = list(f.readlines())
        for line in tqdm(lines, desc=filename, ncols=80):
            line = line.strip()
            if len(line) == 0:
                out += '\n'
                continue

            # Simple cleanup steps
            line = line.replace("''", '"')

            # Sentence and word tokenization
            sents = sent_detector.tokenize(line)
            sents = [' '.join(tokenizer.tokenize(s)) for s in sents]
            out += '\n'.join(sents) + '\n'

    with open(dest_path, 'w') as f:
        f.write(out)
