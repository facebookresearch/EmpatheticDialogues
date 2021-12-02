from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx, bert_predict_and_save_json, create_bert_ds
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
    LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "")  # or ""
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

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Encode input words and labels
    train_ds = create_bert_ds(train_dataset.insts, N_SEQ, tokenizer, label2idx, True)
    valid_ds = create_bert_ds(valid_dataset.insts, N_SEQ, tokenizer, label2idx)
    test_ds = create_bert_ds(test_dataset.insts, N_SEQ, tokenizer, label2idx)

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
                  epochs=N_EPOCHS, callbacks=callbacks_list)
        model.save_pretrained(filepath)

    model = TFAutoModelForSequenceClassification.from_pretrained(filepath)

    print("Train: ", model.evaluate(train_ds))
    print("Valid: ", model.evaluate(valid_ds))
    print("Test: ", model.evaluate(test_ds))

    os.makedirs("data/trans/", exist_ok=True)

    bert_predict_and_save_json(model, train_dataset.insts, train_ds, idx2labels,
                               "data/trans/train%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, valid_dataset.insts, valid_ds, idx2labels,
                               "data/trans/valid%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, test_dataset.insts, test_ds, idx2labels,
                               "data/trans/test%s.json" % LABEL_SUFFIX, BATCH_SIZE)

    # history 4 predictions
    train_ds = create_bert_ds(train_dataset.hist_insts, N_SEQ, tokenizer, label2idx)
    valid_ds = create_bert_ds(valid_dataset.hist_insts, N_SEQ, tokenizer, label2idx)
    test_ds = create_bert_ds(test_dataset.hist_insts, N_SEQ, tokenizer, label2idx)

    bert_predict_and_save_json(model, train_dataset.hist_insts, train_ds, idx2labels,
                               "data/trans/train%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, valid_dataset.hist_insts, valid_ds, idx2labels,
                               "data/trans/valid%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
    bert_predict_and_save_json(model, test_dataset.hist_insts, test_ds, idx2labels,
                               "data/trans/test%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
