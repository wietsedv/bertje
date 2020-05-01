import sys


if len(sys.argv) < 3:
    print('Usage: python3 vocab.py source-path dest-path')

source_path = sys.argv[1]
dest_path = sys.argv[2]


vocab = []

with open(source_path, 'r') as f:
    for l in f.readlines():
        t, _ = l.split('\t')
        if t[0] == '▁':
            t = t[1:]
        elif t[0] != '[' or t[-1] != ']':
            t = '##' + t
        if len(t) > 0:
            vocab.append(t)

vocab_special = vocab[:5]
vocab_single_chars = sorted([t for t in vocab[5:] if len(t) == 1]) + sorted([t for t in vocab[5:] if len(t) == 3 and t[:2] == '##'])
vocab_prefix = sorted([t for t in vocab[5:] if t[:2] != '##' and len(t) > 1])
vocab_suffix = sorted([t for t in vocab[5:] if t[:2] == '##' and len(t) > 3])

vocab = vocab_special + vocab_single_chars + vocab_prefix + vocab_suffix

with open(dest_path, 'w') as f:
    f.write('\n'.join(vocab))

print('Saved WordPiece vocabulary to {}'.format(dest_path))
print(f'special={len(vocab_special)}, single_chars={len(vocab_single_chars)}, prefixes={len(vocab_prefix)}, suffixes={len(vocab_suffix)}, total={len(vocab)}')
