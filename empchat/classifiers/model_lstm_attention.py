from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx

import time
import datetime
import numpy as np
from tqdm import tqdm

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

def EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix):
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

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    # define the checkpoint
    filepath = "model_weights-improvement-{epoch:02d}-{val_acc:.6f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

if __name__ == "__main__":
    # start_time = time.time()

    from .data_loader import EmotionDataset
    from .utils import build_word_idx
    from pytorch_pretrained_bert import BertTokenizer

    # TODO 5: set from CMD
    BATCH_SIZE = 16
    GLOVE_FILE = "data/glove.6B.100d.txt"
    N_EMB = 100
    N_SEQ = 50
    HIDDEN_DIM = 64
    N_EPOCHS = 100

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased",
        do_lower_case=False,
        never_split=(
                ["[CLS]", "[MASK]"]
                + list(get_bert_token_mapping(None).values())
        ),
    )

    # load data
    train_dataset = EmotionDataset("data/train.csv", True, tokenizer.tokenize, N_SEQ)
    valid_dataset = EmotionDataset("data/valid.csv", False, tokenizer.tokenize, N_SEQ,
                                   label2idx=train_dataset.label2idx)
    test_dataset = EmotionDataset("data/test.csv", False, tokenizer.tokenize, N_SEQ, label2idx=train_dataset.label2idx)

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
    x_train = [] #[word2idx[word] for word in sentence] for sentence in train_dataset]
    y_train = [] #[label2idx[label] for label in labels]
    for inst in train_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        ids_label.append(label2idx[inst.label])
        x_train.append(ids_word)
        y_train.append(ids_label)

    # Encode input words and labels
    x_valid= [] #[word2idx[word] for word in sentence] for sentence in train_dataset]
    y_valid = [] #[label2idx[label] for label in labels]
    for inst in valid_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        ids_label.append(label2idx[inst.label])
        x_valid.append(ids_word)
        y_valid.append(ids_label)

    # Apply Padding to X
    x_train = pad_sequences(x_train, N_SEQ)
    x_valid = pad_sequences(x_valid, N_SEQ)

    # Convert X to numpy array
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)

    # Convert Y to numpy array
    y_train = keras.utils.to_categorical(y_train, num_classes=len(label2idx), dtype='float32')
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(label2idx), dtype='float32')

    model, callbacks_list = EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix)

    # # Train model
    # model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE,
    #           epochs=1)  # add callbacks=callbacks_list as another parameter later

    # model.save("models/lstm_attention_v1_trained.h5")

    model = load_model("models/lstm_attention_v1_trained.h5", compile=False)

    # end_time = time.time()
    # print("Time taken to train the model", (end_time - start_time))

    # Re-create the model to get attention vectors as well as label prediction
    model_with_attentions = keras.Model(inputs=model.input,
                                        outputs=[model.output,
                                                 model.get_layer('attention_vec').output])

    encoded_samples = [] # encoded_samples = [[word2idx[word] for word in valid_dataset]]
    for inst in valid_dataset.insts:
        ids_word = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        encoded_samples.append(ids_word)

    # Apply Padding
    encoded_samples = pad_sequences(encoded_samples, N_SEQ)

    # Convert to numpy array
    encoded_samples = np.array(encoded_samples)

    # Make predictions
    label_probs, attentions = model_with_attentions.predict(encoded_samples)

    emotions_final = []
    for i in range(len(label_probs)):
        idx = np.argmax(label_probs[i])
        emotion_final = idx2labels[idx]
        emotions_final.append(emotion_final)

    print(emotions_final)
    print(len(emotions_final))