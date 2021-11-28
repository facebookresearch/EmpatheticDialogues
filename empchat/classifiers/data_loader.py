from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
import collections
import re

from .utils import UNK, PAD
from .utils import build_label_idx, check_all_labels_in_dict
from .instance import Instance

Feature = collections.namedtuple('Feature',
                                 'words seq_len label')
Feature.__new__.__defaults__ = (None,) * 6

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class EmotionDataset(Dataset):
    def __init__(self, file: str,
                 is_train: bool,
                 tokenizer,
                 replace_digits=False,
                 label2idx: Dict[str, int] = None):
        """
        Read the dataset into Instance
        """
        # read all the instances. sentences and labels
        self.tokenizer = tokenizer
        insts = self.read_txt(file, replace_digits)
        self.insts = insts
        if is_train:
            print(f"[Data Info] Using the training set to build label index")
            assert label2idx is None
            # build label to index mapping. e.g., joy -> 0, fun -> 1
            idx2labels, label2idx = build_label_idx(insts)
            self.idx2labels = idx2labels
            self.label2idx = label2idx
        else:
            # for dev/test dataset we don't build label2idx, pass in label2idx argument
            assert label2idx is not None
            self.label2idx = label2idx
            check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
        self.inst_ids: List[Feature] = []
        # self.convert_instances_to_feature_tensors()

    def convert_instances_to_feature_tensors(self, word2idx: Dict[str, int]):
        self.PAD_IDX = word2idx[PAD]
        self.inst_ids = []
        for i, inst in enumerate(self.insts):
            words = inst.words
            word_ids = []
            for word in words:
                if word in word2idx:
                    word_ids.append(word2idx[word])
                else:
                    word_ids.append(word2idx[UNK])

            self.inst_ids.append(
                Feature(
                    words=word_ids,
                    seq_len=len(words),
                    label=self.label2idx[inst.label] if (inst.label and inst.label in self.label2idx.keys()) else None
                )
            )

    def read_txt(self, file: str, replace_digits) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in tqdm(enumerate(f.readlines())):
                if index == 0:
                    continue
                sparts = line.strip().split(",")
                sentence = sparts[5].replace("_comma_", ",")
                if replace_digits:
                    sentence = re.sub('\d', '0', sentence)
                # convert sentence to words, tokenize
                words = self.tokenizer(sentence)
                label = sparts[2]
                insts.append(Instance(sentence, words, label))
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.inst_ids)

    def __getitem__(self, index):
        return self.inst_ids[index]
