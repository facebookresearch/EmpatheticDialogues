from tqdm import trange, tqdm
from typing import List, Dict
import re

from .utils import build_label_idx, check_all_labels_in_dict
from .instance import Instance


class EmotionDataset():
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
        self.hist_insts = self.read_hists(file, replace_digits)
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

    def read_txt(self, file: str, replace_digits) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            for index, line in tqdm(enumerate(f.readlines())):
                if index == 0:
                    continue
                sparts = line.strip().split(",")
                ori_sentence = sparts[5].replace("_comma_", ",")
                tmp_sentence = ori_sentence
                if replace_digits:
                    tmp_sentence = re.sub('\d', '0', tmp_sentence)
                # convert sentence to words, tokenize
                words = self.tokenizer(tmp_sentence)
                label = sparts[2]
                insts.append(Instance(ori_sentence, words, label))
        print("number of sentences: {}".format(len(insts)))
        return insts

    def read_hists(self, file, replace_digits, max_hist_len=4) -> List[Instance]:
        hist_insts = []
        history = []

        df = open(file, 'r', encoding='utf-8').readlines()

        for i in trange(1, len(df)):
            cparts = df[i - 1].strip().split(",")
            sparts = df[i].strip().split(",")
            if cparts[0] == sparts[0]:
                prevsent = cparts[5].replace("_comma_", ",")
                history.append(prevsent)

                ori_sentence = " </s> ".join(history[-max_hist_len:])
                tmp_sentence = ori_sentence
                if replace_digits:
                    tmp_sentence = re.sub('\d', '0', tmp_sentence)
                # convert sentence to words, tokenize
                words = self.tokenizer(tmp_sentence)
                label = sparts[2]
                hist_insts.append(Instance(ori_sentence, words, label))
            else:
                history = []

        return hist_insts
