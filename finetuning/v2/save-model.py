import os
import sys
from transformers import ModelCard, AutoTokenizer, BertForTokenClassification, BertForSequenceClassification, TokenClassificationPipeline, TextClassificationPipeline

if len(sys.argv) < 4:
    print('usage: "python save-model.py basename name type task" where "basename" is the original model name ("bert-base-dutch-cased"), "name" is the dir name in "output" and type is "token" or "seq"')
    exit(1)

base_name = sys.argv[1]
name = sys.argv[2]
typ = sys.argv[3]

if typ not in ['token', 'seq']:
    print('type must be token or seq')
    exit(1)

src_path = os.path.join('output', name, 'model')
if not os.path.exists(src_path):
    print(src_path + ' does not exist')
    exit(1)

name = base_name + '-finetuned-' + '-'.join(name.split('-')[:-1])
print(name)

dst_path = f'models/{name}'
os.makedirs(dst_path, exist_ok=True)

# Load model
model = BertForTokenClassification.from_pretrained(
    src_path) if typ == 'token' else BertForSequenceClassification.from_pretrained(src_path)
tokenizer = AutoTokenizer.from_pretrained(base_name)


modelcard = ModelCard(model_details="""This model does not have a specific model card yet.

You can possibly find more information about model comparison and labels at [the Github page](https://github.com/wietsedv/bertje).""")

# Save pipeline
pipeline = TokenClassificationPipeline if typ == 'token' else TextClassificationPipeline
pipe = pipeline(model, tokenizer, modelcard=modelcard)
pipe.save_pretrained(dst_path)
