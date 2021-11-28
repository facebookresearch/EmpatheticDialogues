from empchat.datasets.tokens import get_bert_token_mapping
from empchat.classifiers.utils import build_label_idx

import time
import datetime
import numpy as np

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments

def EmotionClassifierModel(label2idx):
    model_name = "bert-base-uncased"

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2idx))

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    # define the checkpoint
    filepath = "model_weights-improvement-{epoch:02d}-{val_acc:.6f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

if __name__ == "__main__":
    start_time = time.time()

    from .data_loader import EmotionDataset
    # from torch.utils.data import DataLoader
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

    model, callbacks_list = EmotionClassifierModel(label2idx)

    # Train model

    training_args = TrainingArguments(
        num_train_epochs=10,
        per_device_train_batch_size=16,  # batch size per device during training
        weight_decay=0.01,  # strength of weight decay
        load_best_model_at_end=True,
        logging_steps=200,
        evaluation_strategy="steps"
    )

    model = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    model.train()

    model.save("models/bert_v1_trained.h5")

    # model = load_model("models/bert_v1_trained.h5", compile=False)

    end_time = time.time()
    print("Time taken to train the model", (end_time - start_time))