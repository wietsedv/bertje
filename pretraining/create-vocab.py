from glob import glob

from sentencepiece import SentencePieceTrainer


NUM_THREADS = 24
VOCABSIZE = 30_001
NUM_SENTS = 100_000_000


SOURCE_PATH = '/home/s2971992/Bertje/clean-data-v2/*/*.txt'
# SOURCE_PATH = '/Volumes/Data/Corpora/DutchWebNews/clean/*.txt'

input_paths = list(glob(SOURCE_PATH))
input_path = ','.join(input_paths)

print('Total number of files: {}'.format(len(input_paths)))

cmd = '--input={} --vocab_size={} --num_threads={} --input_sentence_size={} --shuffle_input_sentence=true --model_type=unigram --split_by_number=false --split_by_unicode_script=false --model_prefix=dutch --bos_piece=[CLS] --eos_piece=[SEP] --unk_piece=[UNK] --control_symbols=[PAD],[MASK]'.format(
    input_path, VOCABSIZE, NUM_THREADS, NUM_SENTS
)
trainer = SentencePieceTrainer.Train(cmd)
