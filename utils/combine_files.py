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

raw_sentences_file_path = 'uniquesentences_unstripped.txt'
parsed_sentences_path = 'final_50_batch_constituency_trees_0_149518.txt'
SPLT_TOKEN = "("
ROOT_TAG_INDEX = 1


def read_all_lines(path):
    all_lines = None
    with open(path) as f:
        all_lines = f.readlines()
    return all_lines

raw_sentences = read_all_lines(raw_sentences_file_path)
parsed_sentences = read_all_lines(parsed_sentences_path)

combined = []

for i in range(len(raw_sentences)):
    sentence = raw_sentences[i]
    root = parsed_sentences[i].split(SPLT_TOKEN)[ROOT_TAG_INDEX]
    row = sentence.rstrip()+","+root
    combined.append(row)


with open('look_up.txt', 'w') as filehandle:
    filehandle.writelines("%s\n" % sentence for sentence in combined)


