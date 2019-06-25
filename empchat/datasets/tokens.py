# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict
from typing import Optional


UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
PARLAI_PAD_TOKEN = "__PAD__"
EMPTYPERSONA_TOKEN = "<PER>"
START_OF_COMMENT = "<SOC>"
END_OF_COMMENT = "<EOC>"
BERT_ID = "bert-base-cased"

# Used to separate empchat utterances + responses
UNUSED_BERT_TOKEN_1 = "[unused1]"

# Both of these should actually be unused by this repo, because the first token is set
# to '__PAD__', which only exists for ParlAI compatibility, and the second token is only
# used for personas, which are no longer used
UNUSED_BERT_TOKEN_2 = "[unused2]"
UNUSED_BERT_TOKEN_3 = "[unused3]"

# Possible fastText labels that aren't BERT tokens and would thus get broken if we
# didn't map them to unused tokens
SETS_TO_BROKEN_LABELS = {
    "emo": ["anticipating", "apprehensive", "joyful", "nostalgic", "sentimental"],
    "reuters": [
        "acq",
        "alum",
        "barley",
        "bop",
        "carcass",
        "castor-oil",
        "cocoa",
        "coconut-oil",
        "copra-cake",
        "cotton-oil",
        "cpi",
        "cpu",
        "dfl",
        "dlr",
        "dmk",
        "gnp",
        "groundnut",
        "groundnut-oil",
        "hog",
        "instal-debt",
        "ipi",
        "iron-steel",
        "l-cattle",
        "lei",
        "lin-oil",
        "meal-feed",
        "money-fx",
        "money-supply",
        "naphtha",
        "nat-gas",
        "nkr",
        "nzdlr",
        "oat",
        "oilseed",
        "palladium",
        "palm-oil",
        "palmkernel",
        "pet-chem",
        "propane",
        "rand",
        "rape-oil",
        "rapeseed",
        "rye",
        "sorghum",
        "soy-meal",
        "soy-oil",
        "soybean",
        "strategic-metal",
        "sun-meal",
        "sun-oil",
        "sunseed",
        "veg-oil",
        "wpi",
        "yen",
    ],
    "twenty_newsgroups": [
        "atheism",
        "autos",
        "christian",
        "cryptography",
        "mideast",
        "pc",
    ],
}


def get_bert_token_mapping(label_set=None):
    label_set_pairs = []
    unused_token_idx = 3
    if label_set is not None:
        for label in SETS_TO_BROKEN_LABELS[label_set]:
            unused_token_idx += 1
            label_set_pairs.append((label, f"[unused{unused_token_idx:d}]"))
    return OrderedDict(
        [
            (UNK_TOKEN, "[UNK]"),
            (PAD_TOKEN, "[PAD]"),
            (PARLAI_PAD_TOKEN, UNUSED_BERT_TOKEN_2),
            (EMPTYPERSONA_TOKEN, UNUSED_BERT_TOKEN_3),
            (START_OF_COMMENT, UNUSED_BERT_TOKEN_1),
            (END_OF_COMMENT, "[SEP]"),
        ]
        + label_set_pairs
    )


def tokenize(text, split_sep: Optional[str] = " "):
    return (
        text.replace(".", " . ")
        .replace(". . .", "...")
        .replace(",", " , ")
        .replace(";", " ; ")
        .replace(":", " : ")
        .replace("!", " ! ")
        .replace("'", " ' ")
        .replace("?", " ? ")
        .replace("  ", " ")
        .replace("  ", " ")
        .strip()
        .split(split_sep)
    )
