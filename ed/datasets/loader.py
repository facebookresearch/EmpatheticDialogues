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
import tqdm
from torch.utils.data import DataLoader

from ed.datasets.empchat import EmpDataset
from ed.datasets.dailydialog import DDDataset
from ed.datasets.reddit import RedditDataset
from ed.datasets.parlai_dictionary import ParlAIDictionary
from ed.datasets.tokens import (
    get_bert_token_mapping,
    BERT_ID,
    EMPTYPERSONA_TOKEN,
    END_OF_COMMENT,
    PAD_TOKEN,
    PARLAI_PAD_TOKEN,
    START_OF_COMMENT,
    UNK_TOKEN,
)


def build_dictionary(opt):
    if opt.model == "bert":
        return build_bert_dictionary(opt)
    else:
        dict_ = torch.load(os.path.join(opt.reddit_folder, "word_dictionary"))
        dict_["iwords"] = dict_["iwords"][: opt.dict_max_words]
        dict_["iwords"].append(UNK_TOKEN)
        dict_["iwords"].append(PAD_TOKEN)
        dict_["iwords"].append(PARLAI_PAD_TOKEN)
        # Workaround for external ParlAI token
        dict_["iwords"].append(EMPTYPERSONA_TOKEN)
        dict_["iwords"].append(START_OF_COMMENT)
        dict_["iwords"].append(END_OF_COMMENT)
        dict_["words"] = {w: i for i, w in enumerate(dict_["iwords"])}
        return dict_


def build_bert_dictionary(opt):
    try:
        from pytorch_pretrained_bert import BertTokenizer
    except ImportError:
        raise Exception(
            "BERT rankers needs pytorch-pretrained-BERT installed. "
            "\npip install pytorch-pretrained-bert"
        )
    if BERT_ID != "bert-base-cased" and opt.dataset_name == "reddit":
        raise NotImplementedError(
            "Currently, only bert-base-cased can be used with reddit!"
        )
    if BERT_ID != "bert-base-cased" and opt.fasttext_type is not None:
        raise NotImplementedError(
            'Currently, "bert-base-cased" is the only BERT model for which we '
            "have defined lists of fastText labels without BERT tokens!"
        )
    is_cased = BERT_ID.split("-")[2] == "cased"
    tokenizer = BertTokenizer.from_pretrained(
        BERT_ID,
        do_lower_case=not is_cased,
        never_split=(
            ["[CLS]", "[MASK]"]
            + list(get_bert_token_mapping(opt.fasttext_type).values())
        ),
    )
    dict_ = dict()

    # Create dictionary from HuggingFace version. Note that the special tokens
    # have been replicated from build_dictionary() above, and I have used the
    # BERT token equivalence mapping suggested by ParlAI's
    # parlai/agents/bert_ranker/bert_dictionary.py, except for START_OF_COMMENT,
    # which I am setting to a token that hasn't been used before.
    if opt.dict_max_words is not None:
        logging.warning(
            "--dict-max-words will be ignored because we are using the BERT "
            "tokenizer."
        )
    dict_["iwords"] = list(tokenizer.vocab.keys())
    for orig_token, bert_token in get_bert_token_mapping(opt.fasttext_type).items():
        dict_["iwords"][tokenizer.convert_tokens_to_ids([bert_token])[0]] = orig_token
    dict_["words"] = {w: i for i, w in enumerate(dict_["iwords"])}
    dict_["wordcounts"] = None  # Not used here
    dict_["bert_tokenizer"] = tokenizer

    return dict_


class TrainEnvironment:
    def __init__(self, opt, dictionary=None):
        self.opt = opt
        self.dataset_name = opt.dataset_name
        if self.dataset_name in ["dailydialog", "empchat"]:
            if dictionary is not None:
                self.temp_dict = ParlAIDictionary.create_from_reddit_style(dictionary)
            else:
                self.dict = build_dictionary(opt)
                if EMPTYPERSONA_TOKEN not in self.dict["words"]:
                    self.dict["iwords"].append(EMPTYPERSONA_TOKEN)
                    self.dict["words"] = {
                        w: i for i, w in enumerate(self.dict["iwords"])
                    }
                self.temp_dict = ParlAIDictionary.create_from_reddit_style(self.dict)
            self.dict = dictionary or self.temp_dict.as_reddit_style_dict()
        elif self.dataset_name == "reddit":
            self.dict = dictionary or build_dictionary(opt)
            if EMPTYPERSONA_TOKEN not in self.dict["words"]:
                self.dict["iwords"].append(EMPTYPERSONA_TOKEN)
                self.dict["words"] = {w: i for i, w in enumerate(self.dict["iwords"])}
            dict_words = self.dict["words"]
            self.personas = load_personas(opt, dict_words) if opt.use_personas else {}
        else:
            raise ValueError("Dataset name unrecognized!")
        self.pad_idx = self.dict["words"][PAD_TOKEN]

    def build_reddit_dataset(self, chunk_id):
        return RedditDataset(
            self.opt.reddit_folder,
            chunk_id,
            self.dict,
            self.personas,
            max_len=self.opt.max_sent_len,
            rm_long_sent=self.opt.rm_long_sent,
            max_hist_len=self.opt.max_hist_len,
            rm_long_contexts=self.opt.rm_long_contexts,
        )

    def build_train_dataloader(self, epoch_id):
        if self.dataset_name == "empchat":
            dataset = EmpDataset(
                "train",
                self.temp_dict,
                data_folder=self.opt.empchat_folder,
                reactonly=self.opt.reactonly,
                history_len=self.opt.max_hist_len,
                emp_loss=self.opt.emp_loss,
                fasttext=self.opt.fasttext,
                fasttext_type=self.opt.fasttext_type,
                fasttext_path=self.opt.fasttext_path,
            )
            return DataLoader(
                dataset,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.no_shuffle,
                num_workers=0,
                collate_fn=self.batchify,
                pin_memory=self.opt.cuda,
            )
        elif self.dataset_name == "reddit":
            dataset = self.build_reddit_dataset(epoch_id % 999)
            return DataLoader(
                dataset,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.no_shuffle,
                num_workers=8,
                collate_fn=self.batchify,
                pin_memory=self.opt.cuda,
            )
        else:
            raise ValueError("Dataset name unrecognized!")

    def build_valid_dataloader(self, shuffle=True, test=False):
        if self.dataset_name == "dailydialog":
            splitname = "valid" if not test else "test"
            dataset = DDDataset(
                splitname=splitname,
                dic=self.temp_dict,
                data_folder=self.opt.dailydialog_folder,
                maxlen=self.opt.max_sent_len,
                history_len=self.opt.max_hist_len,
            )
        elif self.dataset_name == "empchat":
            splitname = "valid" if not test else "test"
            dataset = EmpDataset(
                splitname,
                self.temp_dict,
                data_folder=self.opt.empchat_folder,
                reactonly=self.opt.reactonly,
                history_len=self.opt.max_hist_len,
                emp_loss=self.opt.emp_loss,
                fasttext=self.opt.fasttext,
                fasttext_type=self.opt.fasttext_type,
                fasttext_path=self.opt.fasttext_path,
            )
        elif self.dataset_name == "reddit":
            dataset = self.build_reddit_dataset(999)
        else:
            raise ValueError("Dataset name unrecognized!")
        return DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            collate_fn=self.batchify,
            num_workers=0,
            shuffle=shuffle,
            pin_memory=self.opt.cuda,
        )

    def batchify(self, batch):
        input_list = list(zip(*batch))
        starttoken = (
            None if not self.opt.generate else self.dict["words"][START_OF_COMMENT]
        )
        endtoken = None if not self.opt.generate else self.dict["words"][END_OF_COMMENT]
        contexts, next_ = [
            pad(ex, self.pad_idx, gen=starttoken, endgen=endtoken)
            for ex in [input_list[0], input_list[2]]
        ]
        personas = pad(input_list[1], self.pad_idx)
        return contexts, personas, next_

    def to_words(self, tensor):
        return " ".join(self.dict["iwords"][x] for x in tensor.tolist())


def pad(tensors, padding_value=-1, gen=None, endgen=None):
    """
    Concatenate and pad the input tensors, which may be 1D or 2D. If gen and endgen are
    not None, add them as BOS/EOS tokens.
    """
    max_len = max(t.size(-1) for t in tensors)
    add_bos_eos = (gen is not None and endgen is not None)
    if tensors[0].dim() == 1:
        if add_bos_eos:
            max_len += 2
        out = torch.LongTensor(len(tensors), max_len).fill_(padding_value)
        for i, t in enumerate(tensors):
            if add_bos_eos:
                out[i, 0] = gen
                out[i, 1 : t.size(0) + 1] = t
                out[i, t.size(0) + 1] = endgen
            else:
                out[i, : t.size(0)] = t
        return out
    elif tensors[0].dim() == 2:
        max_width = max(t.size(0) for t in tensors)
        if add_bos_eos:
            max_width += 2
        out = torch.LongTensor(len(tensors), max_width, max_len).fill_(padding_value)
        for i, t in enumerate(tensors):
            if add_bos_eos:
                out[i, 0, : t.size(1)] = gen
                out[i, 1 : t.size(0) + 1, : t.size(1)] = t
                out[i, t.size(0) + 1, : t.size(1)] = endgen
            else:
                out[i, : t.size(0), : t.size(1)] = t
        return out
    else:
        raise ValueError('Input tensors must be either 1D or 2D!')


def split_persona_line(line, dictionary, max_n_personas):
    sentences_for_uid = []
    current_sentence = []
    words = line.rstrip().split()
    for w in words:
        current_sentence.append(dictionary.get(w, 0))
        if w in {".", "!", "?"}:
            sentences_for_uid.append(current_sentence)
            current_sentence = []
            if len(sentences_for_uid) == max_n_personas:
                return sentences_for_uid
    if len(current_sentence) > 0:
        sentences_for_uid.append(current_sentence)
    return sentences_for_uid


def load_personas(opt, dictionary, split_sentences=True):
    logging.info(f"Loading personas from {opt.personas}")
    if "encoded" in opt.personas:
        return torch.load(opt.personas)
    uid2personas = {}
    max_n_personas = opt.max_personas
    with open(opt.personas, "r") as f:
        for line in tqdm.tqdm(f, desc='Looping over personas'):
            if hasattr(opt, "lowercase_personas") and opt.lowercase_personas:
                line = line.lower()
            uid, personas = line.split(",", 1)
            if split_sentences:
                sentences_for_uid = split_persona_line(line, dictionary, max_n_personas)
                if len(sentences_for_uid) == 0:
                    continue
                tensor_length = max(len(sentence) for sentence in sentences_for_uid)
                persona_w = torch.LongTensor(
                    len(sentences_for_uid), tensor_length
                ).fill_(dictionary[PAD_TOKEN])
                for i, sentence in enumerate(sentences_for_uid):
                    persona_w[i, : len(sentence)] = torch.LongTensor(sentence)
                if "emptypersona" in opt and opt.emptypersona:
                    emptypersona = torch.LongTensor(1, tensor_length).fill_(
                        dictionary[PAD_TOKEN]
                    )
                    emptypersona[0] = dictionary[EMPTYPERSONA_TOKEN]
                    persona_w = torch.cat([emptypersona, persona_w])
            else:
                persona_w = torch.LongTensor(
                    [dictionary.get(w, 0) for w in personas.rstrip().split()]
                )
            uid2personas[int(uid)] = persona_w
            if opt.fast_debug and len(uid2personas) > 1000:
                break
    logging.info(f"n personas: {len(uid2personas)}")
    persona_string = opt.personas.split("/")[-1]
    dumpfn = f"../dumps/encoded-{persona_string}"
    logging.info(f"dumping encoded personas to {dumpfn}")
    torch.save(uid2personas, dumpfn)
    return uid2personas
