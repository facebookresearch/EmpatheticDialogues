We forked the original codebase from [here](https://github.com/facebookresearch/EmpatheticDialogues). The
original [readme](./README_OLD.md) has been moved.

Download the models trained by us from [here](https://drive.google.com/file/d/1fTF8xa3EC2yPQksAXmspPJOJTt3d6Veg/view)

Demonstration video for the project: [YouTube](https://www.youtube.com/watch?v=5JLwE3JxIaU)

# Table of contents

1. [Setup steps](#setup-steps)

   1.1 [Environment](#environment)

   1.2 [Data & Models](#data--models)

2. [Code Structure](#code-structure)
3. [Running code](#running-code)

   3.1 [Training Emotion Classifiers](#training-emotion-classifiers)

   3.2 [Training Retrieval w/ BERT](#training-retrieval-w-bert)

4. [Technical Difficulties](#technical-difficulties)

<hr/>

## Setup steps

### Environment

We will setup two conda environments, one for running the code in the original paper and the other for running the code
in the new environment.

> Environment 1 : empathy01

```bash
conda create -n empathy01 python=3.6

conda activate empathy01

conda install numpy=1.14.3

conda install pytorch-cpu==1.0.1 torchvision-cpu==0.2.2 cpuonly -c pytorch

pip install tqdm==4.19.7

conda install pandas=0.22.0

pip install fairseq==0.6.2

pip install git+git://github.com/facebookresearch/ParlAI.git@471db18c47d322d814f4e1bba6e35d9da6ac31ff

pip install pytorch-pretrained-bert==0.5.1

pip install fasttext==0.9.1
```

> Environment 2 : empathy02

```bash
pip install tqdm==4.19.7

pip install tensorflow==2.4.1

pip install keras==2.4.3

pip install pytorch-pretrained-bert==0.5.1

pip install transformers==4.12.5
```

<hr/>

### Data & Models

- Download the dataset as mentioned in the [original codebase](./README_OLD.md#dataset) and extract under `data/`
  directory.
- Run [data/transform_labels.py](./data/transform_labels.py) to generate the 8 label variant of dataset from the 32
  label dataset.
- Download the pre-trained models as provided in the [original codebase](./README_OLD.md#models) and extract
  under `models/`
  directory.

<hr/>

## Code Structure

Apart from the original code, the files that we introduced reside in [empchat/classifiers](./empchat/classifiers) folder
and [data](./data) folder.

[empchat/classifiers](./empchat/classifiers) folder contains the model training and evaluation files for the Emotion
classifiers.

<hr/>

## Running code

### Training Emotion Classifiers

Activate the `empathy02` environment, and then you can train these models.

1. BiLSTM model - [model file](./empchat/classifiers/model_lstm.py)

   To train **32** labels model -

      ```bash
      python -m empchat.classifiers.model_lstm
      ```

   To train **8** labels model -

      ```bash
      LABEL_SUFFIX=_8 python -m empchat.classifiers.model_lstm
      ```

2. BiLSTM w/ Attention model - [model file](./empchat/classifiers/model_lstm_attention.py)

   To train **32** labels model -

      ```bash
      python -m empchat.classifiers.model_lstm_attention
      ```

   To train **8** labels model -

      ```bash
      LABEL_SUFFIX=_8 python -m empchat.classifiers.model_lstm_attention
      ```

2. BERT model - [model file](./empchat/classifiers/model_transformer.py)

   To train **32** labels model -

      ```bash
      python -m empchat.classifiers.model_transformer
      ```

   To train **8** labels model -

      ```bash
      LABEL_SUFFIX=_8 python -m empchat.classifiers.model_transformer
      ```

### Training Retrieval w/ BERT

Activate the `empathy01` environment, and then you can train these models.
The training and evaluation scripts reside in `cmd/` directory.

1. BiLSTM model -
   ```bash
   # train 32 labels
   bash cmd/lstm.sh
   # evaluate 32 labels
   bash cmd/eval_lstm.sh
   
   # train 8 labels
   bash cmd/lstm_8.sh
   # evaluate 8 labels
   bash cmd/eval_lstm_8.sh
   ```

2. BiLSTM w/ Attention model -
   ```bash
   # train 32 labels
   bash cmd/lstm_attn.sh
   # evaluate 32 labels
   bash cmd/eval_lstm_attn.sh
   
   # train 8 labels
   bash cmd/lstm_attn_8.sh
   # evaluate 8 labels
   bash cmd/eval_lstm_attn_8.sh
   ```

3. Replicating fasttext model -
    ```bash
   # train 32 labels
   bash cmd/fast.sh
   # evaluate 32 labels
   bash cmd/eval_fast.sh
   ```

**Notes -**

- EMO classifier model is selected by specifiying ENV variable `EMO_MODEL=fast(default)|lstm|attn|trans`

<hr/>

## Technical Difficulties

- Replicating the original environment was cumbersome. Since, the libraries used are a bit outdated it took some time to
  make it work with GPUs and resolving conflicts.
- Adding `Attention` and `BERT` to the older libraries without updating the versions seemed impossible. Hence, we
  created 2 separate environments to make things work without conflicts.
- We couldn't replicate/train the custom transformer based retrieval proposed in the original code as a
  crucial [word dictionary](./empchat/datasets/loader.py#L35) is missing. Hence, we only replicate/train retrieval w/
  BERT models.
- Training the retrieval with BERT models required huge ~90GB of GPU memory! We consumed all the credits in setup and
  training BiLSTM models. So, we have just added the model code.
