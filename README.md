# EmpatheticDialogues

PyTorch original implementation of Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset (https://arxiv.org/abs/1811.00207).

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

#### Pretraining
```
python retrieval_train.py \
--batch-size 512 \
--cuda \
--dataset-name reddit \
--dict-max-words 250000 \
--display-iter 250 \
--embeddings ${REDDIT_EMBEDDINGS_PATH} \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learn-embeddings \
--learning-rate 8e-4 \
--model transformer \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name model \
--n-layers 4 \
--num-epochs 10000 \
--optimizer adamax \
--reddit-folder ${REDDIT_DATA_FOLDER} \
--transformer-dim 300 \
--transformer-n-heads 6
```

#### Fine-tuning
```
python retrieval_train.py \
--batch-size 512 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 250 \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learn-embeddings \
--learning-rate 8e-4 \
--load-checkpoint ${PRETRAINED_MODEL_PATH} \
--max-hist-len 4 \
--model transformer \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name model \
--n-layers 4 \
--num-epochs 10 \
--optimizer adamax \
--reddit-folder ${REDDIT_DATA_FOLDER} \
--transformer-dim 300 \
--transformer-n-heads 6
```

#### Evaluation
```
# P@1,100
python retrieval_train.py \
--batch-size 512 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 250 \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model transformer \
--model-dir ${EVAL_SAVE_FOLDER} \
--model-name model \
--n-layers 4 \
--optimizer adamax \
--pretrained ${TRAIN_SAVE_FOLDER}/model.mdl \
--reactonly \
--transformer-dim 300 \
--transformer-n-heads 6

# Self-BLEU (EmpatheticDialogues context/candidates)
python retrieval_eval_bleu.py \
--empchat-cands \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model ${TRAIN_SAVE_FOLDER}/model.mdl \
--name model \
--output-folder ${EVAL_SAVE_FOLDER} \
--reactonly \
--task empchat
```

### BERT-based retrieval

#### Pretraining
```
python retrieval_train.py \
--batch-size 256 \
--bert-dim 300 \
--cuda \
--dataset-name reddit \
--dict-max-words 250000 \
--display-iter 100 \
--embeddings None \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learning-rate 6e-5 \
--model bert \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name model \
--num-epochs 10000 \
--optimizer adamax \
--reddit-folder ${BERT_TOKENIZED_REDDIT_DATA_FOLDER}
```

#### Fine-tuning
```
python retrieval_train.py \
--batch-size 256 \
--bert-dim 300 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 100 \
--embeddings None \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--learning-rate 1e-5 \
--load-checkpoint ${PRETRAINED_MODEL_PATH} \
--max-hist-len 4 \
--model bert \
--model-dir ${TRAIN_SAVE_FOLDER} \
--model-name model \
--num-epochs 100 \
--optimizer adamax \
--stop-crit-num-epochs 10
```

#### Evaluation
```
# P@1,100
python retrieval_train.py \
--batch-size 256 \
--bert-dim 300 \
--cuda \
--dataset-name empchat \
--dict-max-words 250000 \
--display-iter 100 \
--embeddings None \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model bert \
--model-dir ${EVAL_SAVE_FOLDER} \
--model-name model \
--optimizer adamax \
--pretrained ${TRAIN_SAVE_FOLDER}/model.mdl \
--reactonly

# Self-BLEU (EmpatheticDialogues context/candidates)
python retrieval_eval_bleu.py \
--bleu-dict ${PATH_TO_MODEL_WITH_TRANSFORMER_DICT} \
--empchat-cands \
--empchat-folder ${EMPATHETIC_DIALOGUES_DATA_FOLDER} \
--max-hist-len 4 \
--model ${TRAIN_SAVE_FOLDER}/model.mdl \
--name model \
--output-folder ${EVAL_SAVE_FOLDER} \
--reactonly \
--task empchat
```

Note: we pass in a separate dictionary (`--bleu-dict`) when calculating the self-BLEU of BERT models in order to match tokenization between Transformer and BERT models.

#### EmoPrepend-1

Add the following flags when calling `retrieval_train.py`:
```
--fasttext 1 \
--fasttext-path ${PATH_TO_TRAINED_FASTTEXT_MODEL} \
--fasttext-type emo
```

## References

Please cite [[1]](https://arxiv.org/abs/1811.00207) if you found the resources in this repository useful.

### Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset

[1] H. Rashkin, E. M. Smith, M. Li, Y. Boureau [*Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset*](https://arxiv.org/abs/1811.00207)

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
