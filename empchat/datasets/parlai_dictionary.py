# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# I don't really want to import all of ParlAI for just a dictionary
# This very simple dictionary mimics what the one of ParlAI does,
# without the long chain of dependency.
# Thanks to the fact that python is not typed whatsoever, this should work

from empchat.datasets.tokens import tokenize


class ParlAIDictionary:
    def __init__(self, file_path=None):
        """
        Initializes the dictionary with the same type of file that ParlAI's
        dictionary uses: tab separated dics
        """
        self.tok2ind = {}
        self.ind2tok = {}
        self.freq = {}
        print(f"Loading dictionary from {file_path}")
        if file_path is not None:
            with open(file_path, "r") as f:
                counter = 0
                for line in f:
                    splited = line[0:-1].split("\t")
                    if splited[0] not in self.tok2ind:
                        self.tok2ind[splited[0]] = counter
                        self.ind2tok[counter] = splited[0]
                        self.freq[splited[0]] = int(splited[1])
                        counter += 1
            self.null_token = self.ind2tok[counter - 1]
            self.unk_token = self.ind2tok[counter - 2]

    def vec2txt(self, vec):
        raw = " ".join(self.ind2tok[idx] for idx in vec)
        return (
            raw.replace("__END__", "")
            .replace(" . ", ". ")
            .replace(" ! ", "! ")
            .replace(" , ", ", ")
            .replace(" ? ", "? ")
        )

    def txt2vec(self, text):
        return [
            self.tok2ind.get(token, self.tok2ind.get(self.unk_token, None))
            for token in tokenize(text)
        ]

    def as_reddit_style_dict(self):
        """
        Turned out reddit dataset also has a weird style dict. Convert this one
        to this style so that later in the code they speak the same language.
        """
        words = self.tok2ind
        iwords = []
        for i in range(len(self.tok2ind)):
            iwords.append(self.ind2tok[i])
        res = {"words": words, "iwords": iwords, "wordcounts": self.freq}
        if hasattr(self, "bert_tokenizer"):
            res["bert_tokenizer"] = self.bert_tokenizer
        return res

    @staticmethod
    def create_from_reddit_style(reddit_style_dic):
        res = ParlAIDictionary()
        for w in reddit_style_dic["words"].keys():
            res.tok2ind[w] = reddit_style_dic["words"][w]
        for i in range(len(reddit_style_dic["iwords"])):
            res.ind2tok[i] = reddit_style_dic["iwords"][i]
        res.null_token = "<PAD>"  # res.ind2tok[len(res.ind2tok)-1]
        res.unk_token = "<UNK>"  # res.ind2tok[len(res.ind2tok)-2]
        if "bert_tokenizer" in reddit_style_dic:
            res.bert_tokenizer = reddit_style_dic["bert_tokenizer"]
        return res

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.unk_token)
        elif type(key) == str:
            return self.tok2ind.get(key, self.tok2ind.get(self.unk_token, None))

    def __len__(self):
        return len(self.tok2ind)
