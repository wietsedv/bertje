# BERTje: A Dutch BERT model

BERTje is a Dutch pre-trained BERT model developed at the University of Groningen.

<img src="/bertje.png" height="250">

For details, check out our paper on arxiv: https://arxiv.org/abs/1912.09582


## Transformers

BERTje is the default Dutch BERT model in [Transformers](https://github.com/huggingface/transformers)! You can start using it with the following snippet:

```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-dutch-cased")
model = BertModel.from_pretrained("bert-base-dutch-cased")
```

That's all! Check out the [Transformers documentation](https://huggingface.co/transformers/model_doc/bert.html) for further instructions.

## Benchmarks

The Arxiv paper lists benchmarks. Here are a couple of comparisons between BERTje, multilingual BERT, BERT-NL and RobBERT that were done after writing the paper. Unlike some other comparisons, the fine-tuning procedures for these benchmarks are identical for each pre-trained model. You may be able to achieve higher scores for individual models by optimizing fine-tuning procedures.

More experimental results will be added to this page when they are finished. Technical details about how a fine-tuned these models will be published later as well as downloadable fine-tuned checkpoints.

All of the tested models are *base* sized (12) layers with cased tokenization.

### Named Entity Recognition

| Model  | [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/) | [SoNaR-1](https://ivdnt.org/downloads/taalmaterialen/tstc-sonar-corpus) |
| --- | --- | --- |
| **BERTje** | **90.24** | **84.93** |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)   | 88.61 | 84.19 |
| [BERT-NL](http://textdata.nl) | 85.05 | 80.45 |
| [RobBERT](https://github.com/iPieter/RobBERT) | 84.72 | 81.98 |

### Part-of-speech tagging

| Model  | [UDv2.5 LassySmall](https://universaldependencies.org/treebanks/nl_lassysmall/index.html) |
| --- | --- |
| **BERTje** | 96.48 |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md)   | **96.49** |
| [BERT-NL](http://textdata.nl) | 96.10 |
| [RobBERT](https://github.com/iPieter/RobBERT) | 95.91 |


## Download
Download the model here:

 - BERT-base, cased (12-layer, 768-hidden, 12-heads, 110M parameters)
   - [`bert-base-dutch-cased.zip`](https://bertje.s3.eu-central-1.amazonaws.com/v1/bert-base-dutch-cased.zip) (1.5GB) ([`vocab.txt`](https://bertje.s3.eu-central-1.amazonaws.com/v1/vocab.txt) • [`config.json`](https://bertje.s3.eu-central-1.amazonaws.com/v1/config.json))

The model is fully compatible with [Transformers](https://github.com/huggingface/transformers) and interchangable with [original](https://github.com/google-research/bert#pre-trained-models) BERT checkpoints.


## Acknowledgements
Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).


## Citation

Do you use BERTje for a publication? Please use the following citation:

```
@misc{vries2019bertje,
    title={BERTje: A Dutch BERT Model},
    author={Wietse de Vries and Andreas van Cranenburgh and Arianna Bisazza and Tommaso Caselli and Gertjan van Noord and Malvina Nissim},
    year={2019},
    eprint={1912.09582},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
