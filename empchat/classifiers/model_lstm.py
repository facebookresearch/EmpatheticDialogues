from empchat.datasets.tokens import get_bert_token_mapping

##CHANGED FROM HERE
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
##TILL HERE

if __name__ == "__main__":
    from data_loader import EmotionDataset
    from torch.utils.data import DataLoader
    from utils import build_word_idx, PAD
    from empchat.datasets.loader import pad
    from pytorch_pretrained_bert import BertTokenizer

    # TODO: provide tokenization function
    ##CHANGED FROM HERE
    tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-cased",
                    do_lower_case=False,
                    never_split=(
                        ["[CLS]", "[MASK]"]
                        + list(get_bert_token_mapping(None).values())
                    ),
                )
    ##TILL HERE

    # load data
    train_dataset = EmotionDataset("../../data/train.csv", True, tokenizer.tokenize)
    valid_dataset = EmotionDataset("../../data/valid.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx) ##CHANGED TRUE/FALSE
    test_dataset = EmotionDataset("../../data/test.csv", False, tokenizer.tokenize, label2idx=train_dataset.label2idx) ##CHANGED TRUE/FALSE

    word2idx, idx2word, char2idx, idx2char = build_word_idx(
        train_dataset.insts, valid_dataset.insts, test_dataset.insts
    )

    # convert to tensors
    train_dataset.convert_instances_to_feature_tensors(word2idx)
    valid_dataset.convert_instances_to_feature_tensors(word2idx)
    test_dataset.convert_instances_to_feature_tensors(word2idx)


    def batchify(batch):
        input_list = list(zip(*batch))
        contexts, next_ = [
            pad(ex, word2idx[PAD]) for ex in [input_list[0], input_list[1]]
        ]
        return contexts, next_


    ##CHANGED FROM HERE
    # create loaders
    train_loader = DataLoader(
        train_dataset,
        # TODO: set from CMD
        batch_size=16,
        shuffle=True,
        num_workers=12,
        # see if this works
        collate_fn=batchify,
        pin_memory=False,
    )

# Maps each word in the embeddings vocabulary to it's embedded representation
embeddings_index = {}
with open('glove.6B.100d.txt', "r", errors="ignore") as f:
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

N_vocab = len(word2idx)
N_EMB = 200
N_SEQ = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Convert to torch tensor to be used directly in the embedding layer:
embeddings_tensor = torch.FloatTensor(embedding_matrix).to(device)

class Model(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding.from_pretrained(embeddings_tensor, freeze=True)
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            # dropout = dropout, # adds dropout on the connections between hidden states in one layer to hidden states in the next layer.
            batch_first=True
        )
        # Multihead attention:
        self.mha = nn.MultiheadAttention(2 * hidden_dim, num_heads=8)
        # Flatten into [batch_size, 2*N_HIDDEN*N_SEQ]
        self.flatten = nn.Flatten()
        # Fully connected classifer
        self.fc1 = nn.Linear(N_SEQ * 2 * hidden_dim, 1024)  # As bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, text):
        # Embedding of the given "text" represented as a vector
        embedded = self.embedding(text)  # [batch size, sent len, emb dim]
        # LSTM output
        lstm_output, (ht, cell) = self.lstm(embedded)  # [batch size, sent len, hid dim], [ batch size, 1, hid dim]
        # Compute attention:
        attn_output, attn_output_weights = self.mha(lstm_output, lstm_output, lstm_output)
        # Flatten:
        x = self.flatten(attn_output)
        # Classifer:
        # Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        # Dropout
        x = self.dropout(x)
        # Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        # Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        # Output layer
        output = self.fc4(x)

        return output  # No need for sigmoid, our loss function will apply that for us
##CHANGED TILL HERE