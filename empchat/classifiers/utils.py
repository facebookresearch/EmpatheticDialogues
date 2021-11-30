from typing import List, Tuple, Dict
from .instance import Instance

import numpy as np
import json
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

PAD = "<PAD>"
UNK = "<UNK>"


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build the mapping from label to index and index to labels.
    :param insts: list of instances.
    :return:
    """
    label2idx = {}
    idx2labels = []
    for inst in insts:
        if inst.label not in label2idx:
            idx2labels.append(inst.label)
            label2idx[inst.label] = len(label2idx)

    label_size = len(label2idx)
    print("#labels: {}".format(label_size))
    print("label 2idx: {}".format(label2idx))
    return idx2labels, label2idx


def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
    for inst in insts:
        if inst.label not in label2idx:
            raise ValueError(
                f"The label {inst.label} does not exist in label2idx dict. The label might not appear in the training "
                f"set. "
            )


def build_word_idx(trains: List[Instance], devs: List[Instance], tests: List[Instance]) -> Tuple[
    Dict, List, Dict, List]:
    """
    Build the vocab 2 idx for all instances
    :param train_insts:
    :param dev_insts:
    :param test_insts:
    :return:
    """
    word2idx = dict()
    idx2word = []
    word2idx[PAD] = 0
    idx2word.append(PAD)
    word2idx[UNK] = 1
    idx2word.append(UNK)

    char2idx = {}
    idx2char = []
    char2idx[PAD] = 0
    idx2char.append(PAD)
    char2idx[UNK] = 1
    idx2char.append(UNK)

    # extract char on train, dev, test
    for inst in trains + devs + tests:
        for word in inst.words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    # extract char only on train (doesn't matter for dev and test)
    for inst in trains:
        for word in inst.words:
            for c in word:
                if c not in char2idx:
                    char2idx[c] = len(idx2char)
                    idx2char.append(c)
    return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
    for obj in objs:
        if obj is not None:
            return False
    return [None] * len(objs)


def predict_and_save_json(model, insts: List[Instance], word2idx, idx2labels, N_SEQ, file_name, batch_size):
    encoded_samples = []  # encoded_samples = [[word2idx[word] for word in valid_dataset]]
    for inst in insts:
        ids_word = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        encoded_samples.append(ids_word)

    # Apply Padding
    encoded_samples = pad_sequences(encoded_samples, N_SEQ, value=word2idx[PAD])

    # Convert to numpy array
    encoded_samples = np.array(encoded_samples)

    # Make predictions
    label_probs = model.predict(encoded_samples, batch_size=batch_size)

    emotions_dict = dict()
    for i in range(len(label_probs)):
        idx = np.argmax(label_probs[i])
        emotion_final = idx2labels[idx]
        emotions_dict[insts[i].ori_sentence] = emotion_final

    json.dump(emotions_dict, open(file_name, "w"))


def bert_predict_and_save_json(model, insts: List[Instance], encoded_samples, idx2labels, file_name, batch_size):
    # Make predictions
    label_probs = model.predict(encoded_samples, batch_size=batch_size)

    emotions_dict = dict()
    for i in range(len(label_probs)):
        idx = np.argmax(label_probs[i])
        emotion_final = idx2labels[idx]
        emotions_dict[insts[i].ori_sentence] = emotion_final

    json.dump(emotions_dict, open(file_name, "w"))


def create_x_y_lstm(insts: List[Instance], max_length, word2idx, label2idx, shuffle=False):
    if shuffle:
        np.random.shuffle(insts)
    x = []
    y = []
    for inst in insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            if word in word2idx:
                ids_word.append(word2idx[word])
            else:
                ids_word.append(word2idx[UNK])
        ids_label.append(label2idx[inst.label])
        x.append(ids_word)
        y.append(ids_label)

    x = pad_sequences(x, max_length, value=word2idx[PAD])
    x = np.array(x)
    from keras.utils import to_categorical
    y = to_categorical(y, num_classes=len(label2idx), dtype='float32')
    return x, y


def create_bert_ds(insts: List[Instance], max_length, tokenizer, label2idx, shuffle=False):
    if shuffle:
        np.random.shuffle(insts)
    x = []
    y = []
    for inst in insts:
        # ids_label = []
        # ids_label.append(label2idx[inst.label])
        # y.append(ids_label)
        y.append(label2idx[inst.label])
        x.append(inst.ori_sentence)

    x = tokenizer(x, truncation=True, padding=True, max_length=max_length, return_tensors="tf")

    ds = tf.data.Dataset.from_tensor_slices((dict(x), y))

    return ds
