# BERTje: A Dutch BERT model

BERTje is a Dutch pre-trained BERT model developed at the University of Groningen.

<img src="/bertje.png" height="250">

For details, check out our paper on arxiv: https://arxiv.org/abs/1912.09582


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
