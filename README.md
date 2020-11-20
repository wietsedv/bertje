# BERTje: A Dutch BERT model

BERTje is a Dutch pre-trained BERT model developed at the University of Groningen.

<img src="/bertje.png" height="250">

For details, check out our paper on arxiv: https://arxiv.org/abs/1912.09582


## Publications with BERTje

  - [BERTje: A Dutch BERT Model](https://arxiv.org/abs/1912.09582)
  - [What's so special about BERT's layers? A closer look at the NLP pipeline in monolingual and multilingual models](https://www.aclweb.org/anthology/2020.findings-emnlp.389)


## Transformers

BERTje is the default Dutch BERT model in [Transformers](https://github.com/huggingface/transformers)! You can start using it with the following snippet:

```python
from transformers import BertTokenizer, BertModel, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
model = AutoModel.from_pretrained("wietsedv/bert-base-dutch-cased")  # PyTorch
model = TFAutoModel.from_pretrained("wietsedv/bert-base-dutch-cased")  # Tensorflow
```

That's all! Check out the [Transformers documentation](https://huggingface.co/transformers/model_doc/bert.html) for further instructions.

## Benchmarks

The Arxiv paper lists benchmarks. Here are a couple of comparisons between BERTje, multilingual BERT, BERT-NL and RobBERT that were done after writing the paper. Unlike some other comparisons, the fine-tuning procedures for these benchmarks are identical for each pre-trained model. You may be able to achieve higher scores for individual models by optimizing fine-tuning procedures.

More experimental results will be added to this page when they are finished. Technical details about how a fine-tuned these models will be published later as well as downloadable fine-tuned checkpoints.

All of the tested models are *base* sized (12) layers with cased tokenization.

Headers in the tables below link to original data sources. Scores link to the model pages that corresponds to that specific fine-tuned model. These tables will be updated when more simple fine-tuned models are made available.


### Named Entity Recognition


| Model                                                                        | [CoNLL-2002](https://www.clips.uantwerpen.be/conll2002/ner/)                                  | [SoNaR-1](https://ivdnt.org/downloads/taalmaterialen/tstc-sonar-corpus)                   | spaCy UD LassySmall                                                                             |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **BERTje**                                                                   | [**90.24**](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-conll2002-ner)    | [**84.93**](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-sonar-ner)    | [86.10](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-udlassy-ner)            |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | [88.61](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-conll2002-ner) | [84.19](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-sonar-ner) | [**86.77**](https://huggingface.co/wietsedv/bert-base-multilingual-cased-finetuned-udlassy-ner) |
| [BERT-NL](http://textdata.nl)                                                | 85.05                                                                                         | 80.45                                                                                     | 81.62                                                                                           |
| [RobBERT](https://github.com/iPieter/RobBERT)                                | 84.72                                                                                         | 81.98                                                                                     | 79.84                                                                                           |

### Part-of-speech tagging

| Model                                                                        | [UDv2.5 LassySmall](https://universaldependencies.org/treebanks/nl_lassysmall/index.html) |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **BERTje**                                                                   | **96.48**                                                                                 |
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | 96.20                                                                                     |
| [BERT-NL](http://textdata.nl)                                                | 96.10                                                                                     |
| [RobBERT](https://github.com/iPieter/RobBERT)                                | 95.91                                                                                     |


## Download
The recommended download method is using the Transformers library. The model is available at the [model hub](https://huggingface.co/wietsedv/bert-base-dutch-cased).

You can manually download the model files here:

 - BERT-base, cased (12-layer, 768-hidden, 12-heads, 110M parameters)
   - [`pytorch_model.bin`](https://cdn.huggingface.co/wietsedv/bert-base-dutch-cased/pytorch_model.bin) (1.5GB) ([`vocab.txt`](https://cdn.huggingface.co/wietsedv/bert-base-dutch-cased/vocab.txt) • [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/wietsedv/bert-base-dutch-cased/config.json))

The model is fully compatible with [Transformers](https://github.com/huggingface/transformers) and interchangable with [original](https://github.com/google-research/bert#pre-trained-models) BERT checkpoints.

Thanks to HuggingFace for hosting the model files.


## Code

The main code that is used for pretraining data preparation, finetuning and probing are given in the appropriate directies. Do *not* expect the code to be fully functional, complete or documented since this is research code that has been written and collected in the course of multiple months. Nevertheless, the code can be useful for reference.


## Acknowledgements
Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC).


## Citation

Please use the following citation if you use BERTje or our fine-tuned models:

```bibtex
@misc{devries2019bertje,
	title = {{BERTje}: {A} {Dutch} {BERT} {Model}},
	shorttitle = {{BERTje}},
	author = {de Vries, Wietse  and  van Cranenburgh, Andreas  and  Bisazza, Arianna  and  Caselli, Tommaso  and  Noord, Gertjan van  and  Nissim, Malvina},
	year = {2019},
	month = dec,
	howpublished = {arXiv:1912.09582},
	url = {http://arxiv.org/abs/1912.09582},
}
```

Use the following citation if you use anything from the probing classifiers:

```bibtex
@inproceedings{devries2020bertlayers,
	title = {What's so special about {BERT}'s layers? {A} closer look at the {NLP} pipeline in monolingual and multilingual models},
	author = {de Vries, Wietse  and  van Cranenburgh, Andreas  and  Nissim, Malvina},
	year = {2020},
	booktitle = {Findings of EMNLP},
	pages = {4339--4350},
	url = {https://www.aclweb.org/anthology/2020.findings-emnlp.389},
}
```
