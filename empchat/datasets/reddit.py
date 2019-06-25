# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import os.path

import torch
from torch.utils.data import Dataset

from empchat.datasets.tokens import BERT_ID, START_OF_COMMENT, UNK_TOKEN


class RedditDataset(Dataset):
    def __init__(
        self,
        data_folder,
        chunk_id,
        dict_,
        max_len=100,
        rm_long_sent=False,
        max_hist_len=1,
        rm_long_contexts=False,
        rm_blank_sentences=False,
    ):
        data_path = os.path.join(data_folder, f"chunk{chunk_id}.pth")
        logging.info(f"Loading reddit dataset from {data_path}")
        data = torch.load(data_path)
        self.dictionary = dict_["words"]  # w -> i
        self.iwords = dict_["iwords"]  # i -> w
        self.words = data["w"]
        self.starts = data["cstart"]
        self.ends = data["cend"]
        self.uids = data["uid"]
        self.p2c = data["p2c"]
        self.unk_index = self.dictionary[UNK_TOKEN]
        if "bert_tokenizer" in dict_:
            self.using_bert = True
            assert BERT_ID == "bert-base-cased"
            deleted_uid = -1
        else:
            self.using_bert = False
            deleted_uid = 6
            # This was the deleted uid in the original Reddit binary chunks
        self.max_hist_len = max_hist_len
        self.max_len = max_len
        if self.max_hist_len > 1:
            self.start_of_comment = torch.LongTensor(
                [self.dictionary[START_OF_COMMENT]]
            )
        valid_comments = (self.uids != deleted_uid) * (self.p2c != -1)
        parent_ids = self.p2c.clamp(min=0)
        valid_comments *= (
            self.uids[parent_ids] != deleted_uid
        )  # Remove comments whose parent has been deleted
        if rm_long_sent:
            valid_comments *= (self.ends - self.starts) <= max_len
        if rm_blank_sentences:
            valid_comments *= (self.ends - self.starts) > 0
            # In retrieve.py, length-0 sentences are removed when retrieving
            # candidates; rm_blank_sentences allows length-0 sentence removal
            # when retrieving contexts as well
        if rm_long_contexts:
            valid_comments *= (
                self.ends[parent_ids] - self.starts[parent_ids]
            ) <= max_len
        valid_comment_index = valid_comments.nonzero()
        self.datasetIndex2wIndex = valid_comment_index
        self.n_comments = self.datasetIndex2wIndex.numel()
        logging.info(
            f"Loaded {self.p2c.numel()} comments from {data_path}"
            f" of which {self.n_comments} are valid"
        )

    def __len__(self):
        return self.n_comments

    def __getitem__(self, index):
        i = self.datasetIndex2wIndex[index]
        hist = []
        parent_index = self.p2c[i]
        if self.max_hist_len <= 1:
            context = self.get_words(parent_index)
        else:
            while len(hist) < 2 * self.max_hist_len and parent_index != -1:
                hist.append(self.get_words(parent_index))
                hist.append(self.start_of_comment)
                parent_index = self.p2c[parent_index]
            hist.reverse()
            context = torch.cat(hist)
        pos = self.get_words(i)
        return context, pos

    def get_words(self, index):
        start = self.starts[index]
        end = min(self.ends[index], start + self.max_len)
        selected_words = self.words[start:end]
        if self.using_bert:
            return selected_words
        else:
            return selected_words.clamp(-1, self.unk_index)
