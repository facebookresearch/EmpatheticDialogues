# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re

import torch
from torch.utils.data import Dataset

from ed.datasets.simpler_dictionary import SimplerDictionary
from ed.datasets.tokens import get_bert_token_mapping


def get_emo(emotion):
    """
    simply convert a sentence to a torch tensor.
    """
    emod = {
        "surprised": 0,
        "excited": 1,
        "annoyed": 2,
        "proud": 3,
        "angry": 4,
        "sad": 5,
        "grateful": 6,
        "lonely": 7,
        "impressed": 8,
        "afraid": 9,
        "disgusted": 10,
        "confident": 11,
        "terrified": 12,
        "hopeful": 13,
        "anxious": 14,
        "disappointed": 15,
        "joyful": 16,
        "prepared": 17,
        "guilty": 18,
        "furious": 19,
        "nostalgic": 20,
        "jealous": 21,
        "anticipating": 22,
        "embarrassed": 23,
        "content": 24,
        "devastated": 25,
        "sentimental": 26,
        "caring": 27,
        "trusting": 28,
        "ashamed": 29,
        "apprehensive": 30,
        "faithful": 31,
    }
    temp = torch.LongTensor([[emod[emotion]]])
    return temp


def txt2vec(dic, text, fasttext_type=None):
    if hasattr(dic, "bert_tokenizer"):
        orig_mapping = get_bert_token_mapping(fasttext_type)
        mapping = dict((re.escape(k), v) for k, v in orig_mapping.items())
        pattern = re.compile("|".join(mapping.keys()))
        cleaned_text = pattern.sub(lambda m: mapping[re.escape(m.group(0))], text)
        tokenized_text = dic.bert_tokenizer.tokenize(cleaned_text)
        return dic.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    elif type(dic) is SimplerDictionary:
        return dic.txt2vec(text)
    else:
        return [dic.index(token) for token in SimplerDictionary.tokenize(text)]


def sentence_to_tensor(dic, sentence, maxlen=None, fasttext_type=None):
    """
    simply convert a sentence to a torch tensor.
    """
    indexes = txt2vec(dic, sentence, fasttext_type)
    if maxlen is not None and maxlen <= len(indexes):
        indexes = indexes[: maxlen - 1]
    if type(dic) is SimplerDictionary:
        return torch.LongTensor(indexes)
    else:
        indexes.append(dic.eos())
        return torch.LongTensor(indexes)


class EmpDataset(Dataset):
    def __init__(
        self,
        splitname,
        dic,
        data_folder,
        maxlen=100,
        history_len=1,
        reactonly=False,
        emp_loss=None,
        fasttext=None,
        fasttext_type=None,
        fasttext_path=None,
    ):
        topicmap = {
            "alt.atheism": "atheism",
            "comp.graphics": "graphics",
            "comp.os.ms-windows.misc": "windows",
            "comp.sys.ibm.pc.hardware": "pc",
            "comp.sys.mac.hardware": "mac",
            "comp.windows.x": "x",
            "misc.forsale": "sale",
            "rec.autos": "autos",
            "rec.motorcycles": "motorcycles",
            "rec.sport.baseball": "baseball",
            "rec.sport.hockey": "hockey",
            "sci.crypt": "cryptography",
            "sci.electronics": "electronics",
            "sci.med": "medicine",
            "sci.space": "space",
            "soc.religion.christian": "christian",
            "talk.politics.guns": "guns",
            "talk.politics.mideast": "mideast",
            "talk.politics.misc": "politics",
            "talk.religion.misc": "religion",
            "windows.misc": "windows",
        }
        # ^ 'windows.misc' was included for compatibility because the code below
        # splits topics on hyphens
        df = open(os.path.join(data_folder, f"{splitname}.csv")).readlines()
        newmaxlen = maxlen
        self.max_hist_len = history_len
        if fasttext is not None:
            import fastText
            assert fasttext_type is not None and fasttext_path is not None
            self.ftmodel = fastText.FastText.load_model(fasttext_path)
            newmaxlen += fasttext
            maxlen += fasttext
            if hasattr(dic, "bert_tokenizer"):
                try:
                    from pytorch_pretrained_bert import BertTokenizer
                except ImportError:
                    raise Exception(
                        "BERT rankers needs pytorch-pretrained-BERT installed. "
                        "\npip install pytorch-pretrained-bert"
                    )
                # Replace the tokenizer with a new one that won't split any of
                # the fastText labels
                new_tokenizer = BertTokenizer.from_pretrained(
                    "bert-base-cased",
                    do_lower_case=False,
                    never_split=(
                        ["[CLS]", "[MASK]"]
                        + list(get_bert_token_mapping(fasttext_type).values())
                    ),
                )
                assert new_tokenizer.vocab.keys() == dic.bert_tokenizer.vocab.keys()
                # ^ This should fail if the original tokenizer was not from the
                # 'bert-base-cased' model
                dic.bert_tokenizer = new_tokenizer
        self.reactonly = reactonly
        self.data = []
        self.ids = []
        history = []
        for i in range(1, len(df)):
            cparts = df[i - 1].strip().split(",")
            sparts = df[i].strip().split(",")
            if cparts[0] == sparts[0]:
                prevsent = cparts[5].replace("_comma_", ",")
                history.append(prevsent)
                idx = int(sparts[1])
                if not self.reactonly or ((idx % 2) == 0):
                    prev_str = " <SOC> ".join(history[-self.max_hist_len :])
                    if fasttext is not None:
                        gettop, _ = self.ftmodel.predict(
                            " </s> ".join(history[-self.max_hist_len :]), k=fasttext
                        )
                        for f in gettop:
                            prev_str = (
                                topicmap.get(
                                    f.split("_")[-1].split("-")[-1],
                                    f.split("_")[-1].split("-")[-1],
                                )
                                + " "
                                + prev_str
                            )
                    contextt = sentence_to_tensor(
                        dic, prev_str, fasttext_type=fasttext_type
                    )[:newmaxlen]
                    sent = sparts[5].replace("_comma_", ",")
                    if fasttext is not None:
                        gettop, _ = self.ftmodel.predict(sent, k=fasttext)
                        for f in gettop:
                            sent = (
                                topicmap.get(
                                    f.split("_")[-1].split("-")[-1],
                                    f.split("_")[-1].split("-")[-1],
                                )
                                + " "
                                + sent
                            )
                    label = sentence_to_tensor(dic, sent, fasttext_type=fasttext_type)[
                        :maxlen
                    ]
                    if emp_loss is None:
                        persona = torch.LongTensor([[dic[cparts[2]]]])
                        lbl_min = torch.LongTensor([[dic[sparts[2]]]])
                    else:
                        persona = get_emo(cparts[2])
                        lbl_min = torch.LongTensor([[dic[sparts[2]]]])
                    self.data.append((contextt, persona, label, lbl_min))
                    self.ids.append((sparts[0], sparts[1]))
            else:
                history = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def getid(self, index):
        return self.ids[index]
