from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx, PAD, build_word_idx, predict_and_save_json
from empchat.classifiers.data_loader import EmotionDataset

import numpy as np
from tqdm import tqdm
import os

from pytorch_pretrained_bert import BertTokenizer
import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


def EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix, filepath):
    # Define input tensor
    # Replace max-words,
    sequence_input = keras.Input(shape=(N_SEQ,), dtype='int32')

    # Word embedding layer
    embedded_inputs = Embedding(len(word2idx),
                                N_EMB,
                                weights=[embedding_matrix],
                                input_length=N_SEQ)(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = keras.layers.wrappers.Bidirectional(
        keras.layers.LSTM(N_EMB, return_sequences=False)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(N_EMB, activation='relu')(lstm_outs)
    output = keras.layers.Dense(len(label2idx), activation='softmax')(fc)

    # Finally building model
    model = keras.Model(inputs=[sequence_input], outputs=output)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')

    # Print model summary
    model.summary()

    # define the checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', patience=15)
    callbacks_list = [checkpoint, es]

    return model, callbacks_list


if __name__ == "__main__":

    # TODO 5: set from CMD
    BATCH_SIZE = 64
    GLOVE_FILE = "data/glove.6B.100d.txt"
    N_EMB = 100
    N_SEQ = 160
    HIDDEN_DIM = 64
    N_EPOCHS = 100
    SEED = 42
    TRAIN = True
    filepath = "models/lstm_v1_trained.h5"

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
    train_dataset = EmotionDataset("data/train.csv", True, tokenizer.tokenize)
    valid_dataset = EmotionDataset("data/valid.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx)
    test_dataset = EmotionDataset("data/test.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx)

    word2idx, idx2word, char2idx, idx2char = build_word_idx(
        train_dataset.insts, valid_dataset.insts, test_dataset.insts
    )

    idx2labels, label2idx = build_label_idx(
        train_dataset.insts
    )

    # Maps each word in the embeddings vocabulary to it's embedded representation
    embeddings_index = {}
    with open(GLOVE_FILE, "r") as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    N_vocab = len(word2idx)

    # Maps each word in our vocab to it's embedded representation, if the word is present in the GloVe embeddings
    embedding_matrix = np.zeros((N_vocab, N_EMB))
    n_match = 0

    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            n_match += 1
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(0, 1, (N_EMB,))
    print("Vocabulary match: ", n_match)

    # Encode input words and labels
    x_train = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_train = []  # [label2idx[label] for label in labels]
    np.random.shuffle(train_dataset.insts)
    for inst in train_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        ids_label.append(label2idx[inst.label])
        x_train.append(ids_word)
        y_train.append(ids_label)

    # Encode input words and labels
    x_valid = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_valid = []  # [label2idx[label] for label in labels]
    for inst in valid_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        ids_label.append(label2idx[inst.label])
        x_valid.append(ids_word)
        y_valid.append(ids_label)

    # Encode input words and labels
    x_test = []  # [word2idx[word] for word in sentence] for sentence in train_dataset]
    y_test = []  # [label2idx[label] for label in labels]
    for inst in test_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        ids_label.append(label2idx[inst.label])
        x_test.append(ids_word)
        y_test.append(ids_label)

    # Apply Padding to X
    x_train = pad_sequences(x_train, N_SEQ)
    x_valid = pad_sequences(x_valid, N_SEQ)
    x_test = pad_sequences(x_test, N_SEQ)

    # Convert X to numpy array
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    x_test = np.array(x_test)

    # Convert Y to numpy array
    y_train = keras.utils.to_categorical(y_train, num_classes=len(label2idx), dtype='float32')
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(label2idx), dtype='float32')
    y_test = keras.utils.to_categorical(y_test, num_classes=len(label2idx), dtype='float32')

    model, callbacks_list = EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix, filepath)

    if TRAIN:
        # Train model
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE,
                  epochs=N_EPOCHS, callbacks=callbacks_list)

    model = load_model(filepath, compile=False)

    os.makedirs("data/lstm/test_lstm.json", exist_ok=True)

    predict_and_save_json(model, train_dataset.insts, word2idx, idx2labels, N_SEQ, "data/lstm/train.json", BATCH_SIZE)
    predict_and_save_json(model, valid_dataset.insts, word2idx, idx2labels, N_SEQ, "data/lstm/valid.json", BATCH_SIZE)
    predict_and_save_json(model, test_dataset.insts, word2idx, idx2labels, N_SEQ, "data/lstm/test.json", BATCH_SIZE)
