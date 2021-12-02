from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx, create_x_y_lstm, build_word_idx, predict_and_save_json
from empchat.classifiers.data_loader import EmotionDataset

import numpy as np
from tqdm import tqdm
import os

from pytorch_pretrained_bert import BertTokenizer
import tensorflow as tf
import keras
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
        keras.layers.LSTM(N_EMB, return_sequences=True)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    input_dim = int(lstm_outs.shape[2])
    permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
    attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = keras.layers.Reshape((N_SEQ,))(attention_vector)
    attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = keras.layers.Dense(N_EMB, activation='relu')(attention_output)
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
    BATCH_SIZE = 16
    GLOVE_FILE = "data/glove.6B.100d.txt"
    N_EMB = 100
    N_SEQ = 512
    HIDDEN_DIM = 64
    N_EPOCHS = 100
    SEED = 42
    TRAIN = True
    LABEL_SUFFIX = os.getenv("LABEL_SUFFIX", "")  # or ""
    filepath = "models/lstm_attn%s.h5" % LABEL_SUFFIX

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
    x_train, y_train = create_x_y_lstm(train_dataset.insts, N_SEQ, word2idx, label2idx, True)
    x_valid, y_valid = create_x_y_lstm(valid_dataset.insts, N_SEQ, word2idx, label2idx)
    x_test, y_test = create_x_y_lstm(test_dataset.insts, N_SEQ, word2idx, label2idx, )

    model, callbacks_list = EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix, filepath)

    if TRAIN:
        # Train model
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE,
                  epochs=N_EPOCHS, callbacks=callbacks_list)

    model = load_model(filepath)

    print("Train: ", model.evaluate(x_train, y_train))
    print("Valid: ", model.evaluate(x_valid, y_valid))
    print("Test: ", model.evaluate(x_test, y_test))

    os.makedirs("data/attn/", exist_ok=True)

    predict_and_save_json(model, train_dataset.insts, word2idx, idx2labels, N_SEQ,
                          "data/attn/train%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    predict_and_save_json(model, valid_dataset.insts, word2idx, idx2labels, N_SEQ,
                          "data/attn/valid%s.json" % LABEL_SUFFIX, BATCH_SIZE)
    predict_and_save_json(model, test_dataset.insts, word2idx, idx2labels, N_SEQ,
                          "data/attn/test%s.json" % LABEL_SUFFIX, BATCH_SIZE)

    # history 4 predictions
    predict_and_save_json(model, train_dataset.hist_insts, word2idx, idx2labels, N_SEQ,
                          "data/lstm/train%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
    predict_and_save_json(model, valid_dataset.hist_insts, word2idx, idx2labels, N_SEQ,
                          "data/lstm/valid%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
    predict_and_save_json(model, test_dataset.hist_insts, word2idx, idx2labels, N_SEQ,
                          "data/lstm/test%s-4.json" % LABEL_SUFFIX, BATCH_SIZE)
