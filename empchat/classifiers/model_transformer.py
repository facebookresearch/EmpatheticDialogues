from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx, bert_predict_and_save_json
from empchat.classifiers.data_loader import EmotionDataset

import numpy as np
import os

from pytorch_pretrained_bert import BertTokenizer
import tensorflow as tf
import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from transformers import TFAutoModelForSequenceClassification, BertTokenizerFast, IntervalStrategy
from transformers import TFTrainer, TFTrainingArguments


def EmotionClassifierModel(label2idx, filepath):
    # model_name = "bert-base-cased"
    model_name = "distilbert-base-cased"

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2idx))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        optimizer='adam'
    )
    # model.compile(loss=model.compute_loss, metrics=["accuracy"], optimizer='adam')

    # define the checkpoint
    # checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', patience=15)
    callbacks_list = [
        # checkpoint,
        es
    ]

    return model, callbacks_list


if __name__ == "__main__":

    # TODO 5: set from CMD
    BATCH_SIZE = 64
    # N_EMB = 100
    N_SEQ = 512
    # HIDDEN_DIM = 64
    N_EPOCHS = 100
    SEED = 42
    TRAIN = True
    LABEL_SUFFIX = "_8"  # or ""
    filepath = "models/bert%s.h5" % LABEL_SUFFIX
    # model_name = "bert-base-cased"
    model_name = "distilbert-base-cased"

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased",
        do_lower_case=False,
        never_split=(
                ["[CLS]", "[MASK]"]
                + list(get_bert_token_mapping(None).values())
        ),
    )

    # load data
    train_dataset = EmotionDataset("data/train%s.csv" % LABEL_SUFFIX, True, tokenizer.tokenize)
    valid_dataset = EmotionDataset("data/valid%s.csv" % LABEL_SUFFIX, False, tokenizer.tokenize,
                                   label2idx=train_dataset.label2idx)
    test_dataset = EmotionDataset("data/test%s.csv" % LABEL_SUFFIX, False, tokenizer.tokenize,
                                  label2idx=train_dataset.label2idx)

    idx2labels, label2idx = build_label_idx(
        train_dataset.insts
    )

    # Encode input words and labels
    x_train = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_train = []  # [label2idx[label] for label in labels]
    np.random.shuffle(train_dataset.insts)
    for inst in train_dataset.insts:
        # ids_label = []
        # ids_label.append(label2idx[inst.label])
        # y_train.append(ids_label)
        y_train.append(label2idx[inst.label])
        x_train.append(inst.ori_sentence)

    # Encode input words and labels
    x_valid = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_valid = []  # [label2idx[label] for label in labels]
    for inst in valid_dataset.insts:
        # ids_label = []
        # ids_label.append(label2idx[inst.label])
        # y_valid.append(ids_label)
        y_valid.append(label2idx[inst.label])
        x_valid.append(inst.ori_sentence)

    # Encode input words and labels
    x_test = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_test = []  # [label2idx[label] for label in labels]
    for inst in test_dataset.insts:
        # ids_label = []
        # ids_label.append(label2idx[inst.label])
        # y_test.append(ids_label)
        y_test.append(label2idx[inst.label])
        x_test.append(inst.ori_sentence)

    # Apply Padding to X
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    x_train = tokenizer(x_train, truncation=True, padding=True, max_length=N_SEQ, return_tensors="tf")
    x_valid = tokenizer(x_valid, truncation=True, padding=True, max_length=N_SEQ, return_tensors="tf")
    x_test = tokenizer(x_test, truncation=True, padding=True, max_length=N_SEQ, return_tensors="tf")

    # Convert Y to numpy array
    # y_train = keras.utils.to_categorical(y_train, num_classes=len(label2idx), dtype='float32')
    # y_valid = keras.utils.to_categorical(y_valid, num_classes=len(label2idx), dtype='float32')

    train_ds = tf.data.Dataset.from_tensor_slices((
        dict(x_train),
        y_train
    ))

    valid_ds = tf.data.Dataset.from_tensor_slices((
        dict(x_valid),
        y_valid
    ))

    test_ds = tf.data.Dataset.from_tensor_slices((
        dict(x_test),
        y_test
    ))

    # from IPython import embed
    # 
    # embed()
    # Train model

    # model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2idx))
    # training_args = TFTrainingArguments(
    #     num_train_epochs=1,
    #     per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     weight_decay=0.01,  # strength of weight decay
    #     load_best_model_at_end=True,
    #     logging_steps=1,
    #     evaluation_strategy=IntervalStrategy.STEPS,
    #     output_dir="models/bert"
    # )
    # model = TFTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds)
    # model.train()

    if TRAIN:
        model, callbacks_list = EmotionClassifierModel(label2idx, filepath)
        # Train model
        model.fit(train_ds.batch(BATCH_SIZE), validation_data=valid_ds.batch(BATCH_SIZE), batch_size=BATCH_SIZE,
                  epochs=1, callbacks=callbacks_list)
        model.save_pretrained(filepath)

    model = TFAutoModelForSequenceClassification.from_pretrained(filepath)

    os.makedirs("data/trans/", exist_ok=True)

    bert_predict_and_save_json(model, train_dataset.insts, train_ds, idx2labels,
                               "data/trans/train%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, valid_dataset.insts, valid_ds, idx2labels,
                               "data/trans/valid%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, test_dataset.insts, test_ds, idx2labels, "data/trans/test%s.json" % LABEL_SUFFIX,
                               BATCH_SIZE)
