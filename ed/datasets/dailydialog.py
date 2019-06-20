# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def getmode(array, removenone=False):
    counts = dict()
    for a in array:
        if not removenone or a.lower() != "none":
            counts[a] = counts.get(a, 0) + 1.0
    if len(counts) == 0:
        return []
    max_c = max(counts.values())
    modes = []
    for a in counts:
        if counts[a] == max_c:
            modes.append(a)
    return modes


def multifeel_to_one(emo_context):
    maxemo = getmode(emo_context, removenone=True)
    if len(maxemo) > 1:
        np.random.shuffle(maxemo)
        return maxemo[0]
    elif len(maxemo) < 1:
        return "none"
    else:
        return maxemo[0]


def sentence_to_tensor(dic, sentence):
    """
    simply convert a sentence to a torch tensor.
    """
    indexes = dic.txt2vec(sentence)
    return torch.LongTensor(indexes)


class DDDataset(Dataset):
    def __init__(self, splitname, dic, data_folder, maxlen=100, history_len=1):
        df = self.read_dailydialog_data(data_folder, splitname)
        self.max_hist_len = history_len
        self.data = []
        for i in range(df.shape[0]):
            row = df.iloc[i]
            sent = row["line"]
            history = row["context"]
            if len(history) == 0:
                continue
            prev_str = " <s> ".join(history[-self.max_hist_len :])
            contextt = sentence_to_tensor(dic, prev_str)[:maxlen]
            label = sentence_to_tensor(dic, sent)[:maxlen]
            self.data.append((contextt, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def read_dailydialog_data(floc, traindevtest):
        emolookup = {
            0: "none",
            1: "anger",
            2: "disgust",
            3: "fear",
            4: "happiness",
            5: "sadness",
            6: "surprise",
        }
        split_name = {"train": "train", "valid": "validation", "test": "test"}[
            traindevtest
        ]
        conversations = open(
            os.path.join(floc, split_name, "dialogues_" + split_name + ".txt")
        ).readlines()
        totemot = open(
            os.path.join(floc, split_name, "dialogues_emotion_" + split_name + ".txt")
        ).readlines()
        datarows = []
        for i in range(len(conversations)):
            lines = conversations[i].strip().split("__eou__")
            emotions = totemot[i].strip().split(" ")
            prev_context = []
            prev_emot_context = []
            if len(lines) - 1 != len(emotions):
                print("error")
            for j in range(len(lines) - 1):
                item = [
                    i,
                    prev_context.copy(),
                    multifeel_to_one(prev_emot_context.copy()),
                    lines[j],
                    emolookup[int(emotions[j])],
                ]
                datarows.append(item)
                prev_context.append(lines[j])
                prev_emot_context.append(emolookup[int(emotions[j])])
        return pd.DataFrame(
            datarows, columns=["convid", "context", "emo", "line", "nextemo"]
        )
