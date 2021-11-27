from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx

import numpy as np
from tqdm import tqdm
import time
import datetime
import random
import math
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding

from sklearn.preprocessing import LabelEncoder
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import time
# import torch.optim as optim



# class EmotionClassifierModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
#         super().__init__()
#         # Embedding
#         self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=True)
#         # BiLSTM
#         self.lstm = nn.LSTM(
#             embedding_dim,
#             hidden_dim,
#             num_layers=1,
#             bidirectional=True,
#             # dropout = dropout, # adds dropout on the connections between hidden states in one layer to hidden states in the next layer.
#             batch_first=True
#         )
#         # Multihead attention:
#         # TODO 1: probably MHA was not available for that version
#         #  if so should switch to keras ASAP
#         self.mha = nn.MultiheadAttention(2 * hidden_dim, num_heads=8)
#         # Flatten into [batch_size, 2*N_HIDDEN*N_SEQ]
#         self.flatten = nn.Flatten()
#         # Fully connected classifier
#         self.fc1 = nn.Linear(
#             N_SEQ *
#             2 * hidden_dim, 1024)  # As bidirectional
#         self.dropout = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(1024, 256)
#         self.dropout = nn.Dropout(dropout)
#         self.fc3 = nn.Linear(256, 32)
#         self.fc4 = nn.Linear(32, output_dim)
#
#     def forward(self, text):
#         # Embedding of the given "text" represented as a vector
#         embedded = self.embedding(text)  # [batch size, sent len, emb dim]
#         print(len(embedded[0]))
#         # LSTM output
#         lstm_output, (ht, cell) = self.lstm(embedded)  # [batch size, sent len, hid dim], [ batch size, 1, hid dim]
#         print(lstm_output.size())
#         # Compute attention:
#         attn_output, attn_output_weights = self.mha(lstm_output, lstm_output, lstm_output)
#         print(attn_output.size())
#         # Flatten:
#         x = self.flatten(attn_output)
#         print(x.size())
#         # Classifer:
#         # Layer 1
#         x = self.fc1(x)
#         x = F.relu(x)
#         # Dropout
#         x = self.dropout(x)
#         # Layer 2
#         x = self.fc2(x)
#         x = F.relu(x)
#         # Layer 3
#         x = self.fc3(x)
#         x = F.relu(x)
#         # Output layer
#         output = self.fc4(x)
#
#         return output  # No need for sigmoid, our loss function will apply that for us
#
#
# # TODO 3: change to multiclass, they probably have just binary labels
# def binary_accuracy(logits, y):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """
#     # Round predictions to the closest integer
#     # TODO 3: mostly this F.log_softmax()
#     preds = torch.sigmoid(logits)
#     rounded_preds = torch.round(preds)
#     correct = (rounded_preds == y).float()  # convert into float for division
#     acc = correct.sum() / len(correct)
#     return acc
#
#
# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs
#
#
# def train(model, iterator, optimizer, criterion):
#     """ Trains the model on the given training set """
#     epoch_loss = 0
#     epoch_acc = 0
#
#     model.train()  # Tells your model that you are training the model
#     for text, seq_len, labels in iterator:
#         # https://discuss.pytorch.org/t/how-to-add-to-attribute-to-dataset/86468
#         text = text.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()  # Zero the previous gradients
#
#         logits = model(text)
#         labels = labels.type_as(logits)
#
#         loss = criterion(logits, labels)
#         acc = binary_accuracy(logits, labels)
#         # TODO 4: support unpacking using seq_len??
#
#         loss.backward()  # Compute gradients
#
#         optimizer.step()  # Make the updates
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
#
#
# def evaluate(model, iterator, criterion):
#     """ Evaluates the model on the validation set """
#     epoch_loss = 0
#     epoch_acc = 0
#
#     model.eval()  # Tells the model that we are currently evaluating the model
#     #
#     # with torch.no_grad():  # Temporarily set all the requires_grad flag to false
#
#     for text, seq_len, labels in iterator:
#         text = text.to(device)
#         labels = labels.to(device)
#
#         logits = model(text)
#         labels = labels.type_as(logits)
#
#         loss = criterion(logits, labels)
#         acc = binary_accuracy(logits, labels)
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix):
    # Define input tensor
    # Replace max-words,
    sequence_input = keras.Input(shape=(N_SEQ,), dtype='int32')

    # Word embedding layer
    embedded_inputs = Embedding(len(word2idx),
                                             N_EMB,
                                             weights=[embedding_matrix],
                                             input_length=N_SEQ)(sequence_input)

    # embedding_layer = Embedding(len(word2idx) + 1, N_EMB, weights=[embedding_matrix],
    #                             input_length=N_SEQ, trainable=False)

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
    from .data_loader import EmotionDataset
    # from torch.utils.data import DataLoader
    from .utils import build_word_idx
    from pytorch_pretrained_bert import BertTokenizer

    # TODO 5: set from CMD
    BATCH_SIZE = 16
    GLOVE_FILE = "data/glove.6B.100d.txt"
    N_EMB = 100
    N_SEQ = 50
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(len(word2idx))
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            n_match += 1
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(0, 1, (N_EMB,))
    print("Vocabulary match: ", n_match)
    print(len(embedding_matrix))

    model, callbacks_list = EmotionClassifierModel(N_EMB, N_SEQ, word2idx, label2idx, embedding_matrix)

    # Encode input words and labels
    x_train = [] #[word2idx[word] for word in sentence] for sentence in ]
    y_train = [] #[label2idx[label] for label in labels]
    for inst in train_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        for label in inst.label:
            ids_label.append(label2idx[label])
        x_train.append(ids_word)
        y_train.append(ids_label)

    # Encode input words and labels
    x_valid= [] #[word2idx[word] for word in sentence] for sentence in ]
    y_valid = [] #[label2idx[label] for label in labels]
    for inst in valid_dataset.insts:
        ids_word = []
        ids_label = []
        for word in inst.words:
            ids_word.append(word2idx[word])
        for label in inst.label:
            ids_label.append(label2idx[label])
        x_valid.append(ids_word)
        y_valid.append(ids_label)

    # # Apply Padding to X
    # from keras.preprocessing.sequence import pad_sequences
    #
    # X = pad_sequences(X, max_words)

    # Convert Y to numpy array
    y_train = keras.utils.to_categorical(y_train, num_classes=len(label2idx), dtype='float32')

    # Convert Y to numpy array
    y_valid = keras.utils.to_categorical(y_valid, num_classes=len(label2idx), dtype='float32')

    # Train model
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE,
              epochs=N_EPOCHS, callbacks=callbacks_list)

    # Re-create the model to get attention vectors as well as label prediction
    model_with_attentions = keras.Model(inputs=model.input,
                                        outputs=[model.output,
                                                 model.get_layer('attention_vec').output])
    encoded_samples = [[word2idx[word] for word in valid_dataset]]
    # Make predictions
    label_probs, attentions = model_with_attentions.predict(encoded_samples)
    label_probs = {idx2labels[_id]: prob for (label, _id), prob in zip(label2idx.items(), label_probs[0])}
    emotions = [label for label, _ in label_probs.items()]
    scores = [score for _, score in label_probs.items()]


    # Convert to torch tensor to be used directly in the embedding layer:
    # embeddings_tensor = torch.LongTensor(embedding_matrix).to(device)
    # embeddings_tensor = torch.tensor(embedding_matrix, requires_grad=True).to(device)

    # # convert to tensors
    # train_dataset.convert_instances_to_feature_tensors(word2idx)
    # valid_dataset.convert_instances_to_feature_tensors(word2idx)
    # test_dataset.convert_instances_to_feature_tensors(word2idx)
    #
    # # create loaders
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=12,
    #     # see if this works
    #     collate_fn=train_dataset.batchify,
    #     pin_memory=False,
    # )
    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=12,
    #     # see if this works
    #     collate_fn=valid_dataset.batchify,
    #     pin_memory=False,
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=12,
    #     # see if this works
    #     collate_fn=test_dataset.batchify,
    #     pin_memory=False,
    # )
    #
    # model = EmotionClassifierModel(N_vocab, N_EMB, HIDDEN_DIM, len(train_dataset.label2idx))
    # print(model)
    #
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # # TODO 3: change to multiclass, they probably have just binary labels
    # criterion = nn.CrossEntropyLoss()  # Binary crossentropy: This computes sigma(logits) too, much more numerically stable
    #
    # model = model.to(device)
    # criterion = criterion.to(device)
    #
    # best_valid_loss = float("inf")
    #
    # history = {
    #     "t_loss": [],
    #     "v_loss": [],
    #     "t_acc": [],
    #     "v_acc": []
    # }
    #
    # for epoch in range(N_EPOCHS):
    #     start_time = time.time()
    #
    #     train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    #     valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    #
    #     end_time = time.time()
    #
    #     history["t_loss"].append(train_loss)
    #     history["v_loss"].append(valid_loss)
    #     history["t_acc"].append(train_acc)
    #     history["v_acc"].append(valid_acc)
    #
    #     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    #
    #     # Saves best only
    #     if valid_loss < best_valid_loss:
    #         best_valid_loss = valid_loss
    #         torch.save(model.state_dict(), f"model_{epoch + 1}.pt")
    #
    #     # Print details about each epoch:
    #     print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    #     print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
    #     print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%")
