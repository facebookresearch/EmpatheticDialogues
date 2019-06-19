# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict


UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
PARLAI_PAD_TOKEN = "__PAD__"
EMPTYPERSONA_TOKEN = "<PER>"
START_OF_COMMENT = "<SOC>"
END_OF_COMMENT = "<EOC>"
BERT_ID = "bert-base-cased"


def get_bert_token_mapping(label_set=None):

    # Used to separate empchat utterances + responses
    unused_bert_token_1 = "[unused1]"

    # Both of these should actually be unused by this repo if personas are not
    # used, because the first token is set to '__PAD__', which only exists for
    # ParlAI compatibility, and the second token is only used for personas
    unused_bert_token_2 = "[unused2]"
    unused_bert_token_3 = "[unused3]"

    # Possible fastText labels that aren't BERT tokens and would thus get broken
    # if we didn't map them to unused tokens
    sets_to_broken_labels = {
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
    label_set_pairs = []
    unused_token_idx = 3
    if label_set is not None:
        for label in sets_to_broken_labels[label_set]:
            unused_token_idx += 1
            label_set_pairs.append((label, f"[unused{unused_token_idx:d}]"))

    return OrderedDict(
        [
            (UNK_TOKEN, "[UNK]"),
            (PAD_TOKEN, "[PAD]"),
            (PARLAI_PAD_TOKEN, unused_bert_token_2),
            (EMPTYPERSONA_TOKEN, unused_bert_token_3),
            (START_OF_COMMENT, unused_bert_token_1),
            (END_OF_COMMENT, "[SEP]"),
        ]
        + label_set_pairs
    )
