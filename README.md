# EmpatheticDialogues

PyTorch original implementation of Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset (https://arxiv.org/pdf/1811.00207.pdf).

## Dependencies

Versions given are what the code has been tested on.

### Required
- [numpy](https://www.numpy.org/) (1.14.3)
- [PyTorch](https://pytorch.org/) (1.0.1.post2)
- [tqdm](https://tqdm.github.io/) (4.19.7)

### Optional
- [fairseq](https://fairseq.readthedocs.io/en/latest/) (0.6.2; for BLEU calculation in `retrieval_eval_bleu.py`)
- [fastText](https://fasttext.cc/) (0.8.22; for Prepend models)
- [pandas](https://pandas.pydata.org/) (0.22.0; for DailyDialog dataset)
- [ParlAI](https://parl.ai/) ([commit used](https://github.com/facebookresearch/ParlAI/commit/471db18c47d322d814f4e1bba6e35d9da6ac31ff); for BERT model)
- [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) (0.5.1; for BERT model)

## Dataset

To download the EmpatheticDialogues dataset:

```
wget https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
```

## Commands

### Transformer-based retrieval

[ADD COMMANDS: pretraining, fine-tuning]

### BERT-based retrieval

[ADD COMMANDS: pretraining, fine-tuning]

### [OTHER COMMANDS]

[ADD COMMANDS]

## References

Please cite [[1]](https://arxiv.org/pdf/1811.00207.pdf) if you found the resources in this repository useful.

### Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset

[1] H. Rashkin, E. M. Smith, M. Li, Y. Boureau [*Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset*](https://arxiv.org/pdf/1811.00207.pdf)

```
@inproceedings{rashkin2019towards,
  title = {Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset},
  author = {Hannah Rashkin and Eric Michael Smith and Margaret Li and Y-Lan Boureau},
  booktitle = {ACL},
  year = {2019},
}
```

## License

See the LICENSE file in the root repo folder for more details.
