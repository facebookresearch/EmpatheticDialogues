from typing import List, Tuple, Dict
from .instance import Instance

PAD = "<PAD>"
UNK = "<UNK>"


def build_label_idx(insts: List[Instance]) -> Tuple[List[str], Dict[str, int]]:
    """
    Build the mapping from label to index and index to labels.
    :param insts: list of instances.
    :return:
    """
    label2idx = {}
    idx2labels = []
    label2idx[PAD] = len(label2idx)
    idx2labels.append(PAD)
    for inst in insts:
        if inst.label not in label2idx:
            idx2labels.append(inst.label)
            label2idx[inst.label] = len(label2idx)

    label_size = len(label2idx)
    print("#labels: {}".format(label_size))
    print("label 2idx: {}".format(label2idx))
    return idx2labels, label2idx


def check_all_labels_in_dict(insts: List[Instance], label2idx: Dict[str, int]):
    for inst in insts:
        if inst.label not in label2idx:
            raise ValueError(
                f"The label {inst.label} does not exist in label2idx dict. The label might not appear in the training "
                f"set. "
            )


def build_word_idx(trains: List[Instance], devs: List[Instance], tests: List[Instance]) -> Tuple[
    Dict, List, Dict, List]:
    """
    Build the vocab 2 idx for all instances
    :param train_insts:
    :param dev_insts:
    :param test_insts:
    :return:
    """
    word2idx = dict()
    idx2word = []
    word2idx[PAD] = 0
    idx2word.append(PAD)
    word2idx[UNK] = 1
    idx2word.append(UNK)

    char2idx = {}
    idx2char = []
    char2idx[PAD] = 0
    idx2char.append(PAD)
    char2idx[UNK] = 1
    idx2char.append(UNK)

    # extract char on train, dev, test
    for inst in trains + devs + tests:
        for word in inst.words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word.append(word)

    # extract char only on train (doesn't matter for dev and test)
    for inst in trains:
        for word in inst.words:
            for c in word:
                if c not in char2idx:
                    char2idx[c] = len(idx2char)
                    idx2char.append(c)
    return word2idx, idx2word, char2idx, idx2char


def check_all_obj_is_None(objs):
    for obj in objs:
        if obj is not None:
            return False
    return [None] * len(objs)
